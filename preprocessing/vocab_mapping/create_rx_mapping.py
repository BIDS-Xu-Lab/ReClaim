#!/usr/bin/env python3
"""
NDC to RxNorm Ingredient Mapping
Creates mapping from NDC drug codes to RxNorm ingredients using OMOP vocabulary
"""

import os
import argparse
import logging
import time
import duckdb
import pandas as pd

# Import utilities
from utils import setup_logging, format_time_delta, time_execution

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create NDC to RxNorm ingredient mapping')
    parser.add_argument("--ohdsi_vocab_dir", type=str, 
                       default='standard_vocab/ohdsi_vocab_v20250827',
                       help='Directory containing OHDSI vocabulary files')
    parser.add_argument("--output_dir", type=str,
                       default='mapping',
                       help='Output directory for all mapping files')
    return parser.parse_args()


def create_rx_token_mapping(df, source_code_col, target_code_col, 
                             target_vocab, source_prefix, target_prefix, source_vocab):
    """
    Create RxNorm/NDC mapping dataframe with one-to-many aggregation using COMBSTART/COMBEND tokens
    
    This function aggregates one-to-many mappings into single rows using special combination markers.
    For example, if an NDC code maps to 2 RxNorm ingredients, they are combined as:
    <RX-COMBSTART> <RX-code1> <RX-code2> <RX-COMBEND>
    
    This approach :
    - Drug products may contain multiple active ingredients (e.g., combination drugs)
    - RxNorm hierarchy uses 'Has ingredient' relationships that group related concepts
    - The COMBSTART/COMBEND markers preserve these semantic groupings
    
    Note: NDC and RxNorm codes are numeric and do not contain dots, so no dot removal
    is performed (unlike ICD codes).
    
    By design, each source token should map to exactly one target token after aggregation.
    
    Args:
        df: DataFrame with source and target code mappings
        source_code_col: Column name containing source codes
        target_code_col: Column name containing target codes
        target_vocab: Target vocabulary name (e.g., 'RXNORM', 'RXNORM_INGR')
        source_prefix: Token prefix for source (e.g., 'NDCNUM', 'RX')
        target_prefix: Token prefix for target (e.g., 'RX')
        source_vocab: Source vocabulary name (e.g., 'NDC', 'RXNORM'). 
                     If 'source_vocab' column already exists in df, pass None to use existing.
    
    Returns:
        DataFrame with mapping columns, aggregating one-to-many mappings into single rows
        with COMBSTART/COMBEND markers for multiple targets
    """
    # Create a copy
    mapping_df = df.copy()
    
    # Add mapping-specific columns
    logging.info("%sAdding mapping-specific columns...", INDENT1)
    if source_vocab is not None:
        mapping_df['source_vocab'] = source_vocab
    mapping_df['target_vocab'] = target_vocab
    mapping_df['source_prefix'] = source_prefix
    mapping_df['target_prefix'] = target_prefix
    mapping_df['source_token'] = '<' + source_prefix + '-' + mapping_df[source_code_col] + '>'
    mapping_df['target_token_single'] = '<' + target_prefix + '-' + mapping_df[target_code_col] + '>'
    mapping_df['target_value_single'] = mapping_df[target_code_col]
    
    # Group by source token and aggregate targets (handling one-to-many mappings)
    logging.info("%sHandling one-to-many mappings...", INDENT1)
    # source_vocab is always included in groupby (either passed or already in df)
    groupby_cols = ['source_prefix', 'target_prefix', 'source_token', 'source_vocab', 'target_vocab', 'relationship_id']
    
    grouped = mapping_df.groupby(groupby_cols).agg({
        'target_token_single': lambda x: sorted(list(set(x))),  # Deduplicate then sort
        'target_value_single': lambda x: sorted(list(set(x)))   # Deduplicate then sort
    }).reset_index()
    
    # Count number of target tokens
    grouped['num_targets'] = grouped['target_token_single'].apply(len)
    
    # Create final target_token and target_value with combination markers if needed
    def create_combined_token(row):
        if row['num_targets'] >= 2:
            tokens = ' '.join(row['target_token_single'])
            return f"<{row['target_prefix']}-COMBSTART> {tokens} <{row['target_prefix']}-COMBEND>"
        else:
            return row['target_token_single'][0]
    
    def create_combined_value(row):
        if row['num_targets'] >= 2:
            values = ' '.join(row['target_value_single'])
            return f"{row['target_prefix']}-COMBSTART {values} {row['target_prefix']}-COMBEND"
        else:
            return row['target_value_single'][0]
    
    grouped['target_token'] = grouped.apply(create_combined_token, axis=1)
    grouped['target_value'] = grouped.apply(create_combined_value, axis=1)
    
    # Drop temporary columns
    mapping_df_final = grouped.drop(columns=['target_token_single', 'target_value_single'])
    
    # Verify each source token maps to exactly one target token
    source_token_counts = mapping_df_final['source_token'].value_counts()
    if (source_token_counts > 1).any():
        duplicates = source_token_counts[source_token_counts > 1]
        logging.warning("%sFound %s source tokens with multiple mappings after grouping!", INDENT1, len(duplicates))
        logging.warning("%sFirst few duplicates: %s", INDENT1, duplicates.head())
    else:
        logging.info("%sVerified: Each source token maps to exactly one target token", INDENT1)
    
    # Log statistics about one-to-many mappings
    one_to_many = (mapping_df_final['num_targets'] >= 2).sum()
    one_to_one = (mapping_df_final['num_targets'] == 1).sum()
    logging.info("%sOne-to-one mappings: %s", INDENT1, f"{one_to_one:,}")
    logging.info("%sOne-to-many mappings: %s", INDENT1, f"{one_to_many:,}")
    
    return mapping_df_final


@time_execution
def ndc_drug_to_rxnorm_drug(ohdsi_vocab_dir, output_dir):
    """
    Create NDC drug to RxNorm drug mapping using DuckDB
    
    Args:
        ohdsi_vocab_dir: Directory containing OHDSI vocabulary CSV files
        output_dir: Directory to save intermediate mapping file
    
    Returns:
        DataFrame with NDC to RxNorm drug mappings
    """
    logging.info("Creating NDC to RxNorm drug mapping...")
    
    # Build paths to CSV files
    concept_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT.csv')
    concept_relationship_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT_RELATIONSHIP.csv')
    
    logging.info("%sReading OHDSI vocabulary from: %s", INDENT1, ohdsi_vocab_dir)
    logging.info("%sConcept: %s", INDENT2, concept_path)
    logging.info("%sConcept Relationship: %s", INDENT2, concept_relationship_path)
    
    # Execute the NDC to RxNorm mapping query
    ndc_drug_to_rxnorm_drug_df = duckdb.sql(f"""
    SELECT
        c_ndc.concept_id AS ndc_drug_concept_id,
        c_ndc.concept_code AS ndc_drug_code,
        c_ndc.concept_name AS ndc_drug_name,
        c_rxnorm.concept_id AS rxnorm_drug_concept_id,
        c_rxnorm.concept_code AS rxnorm_drug_code,
        c_rxnorm.concept_name AS rxnorm_drug_name,
        cr.relationship_id
    FROM read_csv_auto('{concept_path}') c_ndc
    JOIN read_csv_auto('{concept_relationship_path}') cr
        ON cr.concept_id_1 = c_ndc.concept_id
        AND cr.relationship_id = 'Maps to'
    JOIN read_csv_auto('{concept_path}') c_rxnorm
        ON c_rxnorm.concept_id = cr.concept_id_2
        AND c_rxnorm.vocabulary_id = 'RxNorm'
        AND c_rxnorm.standard_concept = 'S'
    WHERE c_ndc.vocabulary_id = 'NDC'
    ORDER BY c_ndc.concept_code, c_rxnorm.concept_code
    """).df()
    
    total_mappings = len(ndc_drug_to_rxnorm_drug_df)
    unique_ndc = ndc_drug_to_rxnorm_drug_df['ndc_drug_code'].nunique()
    unique_rxnorm = ndc_drug_to_rxnorm_drug_df['rxnorm_drug_concept_id'].nunique()
    
    logging.info("%sTotal NDC to RxNorm mappings: %s", INDENT1, f"{total_mappings:,}")
    logging.info("%sUnique NDC codes: %s", INDENT1, f"{unique_ndc:,}")
    logging.info("%sUnique RxNorm drug concepts: %s", INDENT1, f"{unique_rxnorm:,}")
    
    # Save intermediate result
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original version (without mapping columns)
    csv_path_original = os.path.join(output_dir, 'ndc_drug_to_rxnorm_drug_original.csv')
    ndc_drug_to_rxnorm_drug_df.to_csv(csv_path_original, index=False)
    logging.info("%sSaved original data (CSV) to: %s", INDENT1, csv_path_original)

    # Create mapping version using reusable utility function
    ndc_drug_to_rxnorm_drug_mapping = create_rx_token_mapping(
        df=ndc_drug_to_rxnorm_drug_df,
        source_code_col='ndc_drug_code',
        target_code_col='rxnorm_drug_code',
        target_vocab='RXNORM',
        source_prefix='NDCNUM',
        target_prefix='RX',
        source_vocab='NDC'
    )
    
    # Save mapping version (with mapping columns)
    csv_path_mapping = os.path.join(output_dir, 'ndc_drug_to_rxnorm_drug_mapping.csv')
    ndc_drug_to_rxnorm_drug_mapping.to_csv(csv_path_mapping, index=False)
    logging.info("%sSaved mapping data (CSV) to: %s", INDENT1, csv_path_mapping)
    
    return ndc_drug_to_rxnorm_drug_df


@time_execution
def rxnorm_drug_to_rxnorm_ingredient(ohdsi_vocab_dir, output_dir):
    """
    Create RxNorm drug to RxNorm ingredient mapping using DuckDB
    
    Args:
        ohdsi_vocab_dir: Directory containing OHDSI vocabulary CSV files
        output_dir: Directory to save intermediate mapping file
    
    Returns:
        DataFrame with RxNorm drug to ingredient mappings
    """
    logging.info("Creating RxNorm drug to ingredient mapping...")
    
    # Build paths to CSV files
    concept_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT.csv')
    concept_ancestor_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT_ANCESTOR.csv')
    
    logging.info("%sReading OHDSI vocabulary from: %s", INDENT1, ohdsi_vocab_dir)
    logging.info("%sConcept: %s", INDENT2, concept_path)
    logging.info("%sConcept Ancestor: %s", INDENT2, concept_ancestor_path)
    
    # Execute the RxNorm drug to ingredient mapping query
    rxnorm_drug_to_rxnorm_ingredient_df = duckdb.sql(f"""
    -- RxNorm Ingredients
    WITH rxnorm_ingredient AS (
        SELECT
            concept_id AS rxnorm_ingredient_concept_id,
            concept_code AS rxnorm_ingredient_code,
            concept_name AS rxnorm_ingredient_name,
            vocabulary_id AS rxnorm_ingredient_vocab,
            concept_class_id AS rxnorm_ingredient_class
        FROM 
            read_csv_auto('{concept_path}') c_rxnorm
        WHERE 
            c_rxnorm.vocabulary_id = 'RxNorm' 
            AND c_rxnorm.concept_class_id = 'Ingredient'
            AND c_rxnorm.standard_concept = 'S'
            AND c_rxnorm.invalid_reason IS NULL
    )
    
    SELECT
        rd.concept_id AS rxnorm_drug_concept_id,
        rd.concept_name AS rxnorm_drug_name,
        rd.concept_code AS rxnorm_drug_code,
        rd.concept_class_id AS rxnorm_drug_class, 
        ri.rxnorm_ingredient_concept_id AS rxnorm_ingredient_concept_id,
        ri.rxnorm_ingredient_code AS rxnorm_ingredient_code,
        ri.rxnorm_ingredient_name AS rxnorm_ingredient_name,
        'Has ingredient' AS relationship_id
    FROM
        (SELECT * FROM read_csv_auto('{concept_path}') 
        WHERE vocabulary_id = 'RxNorm'
        AND standard_concept = 'S'
        AND invalid_reason IS NULL) rd
    JOIN
        read_csv_auto('{concept_ancestor_path}') ca
        ON ca.descendant_concept_id = rd.concept_id
    JOIN
        rxnorm_ingredient ri
        ON ri.rxnorm_ingredient_concept_id = ca.ancestor_concept_id
    ORDER BY rd.concept_code, ri.rxnorm_ingredient_code
    """).df()
    
    total_mappings = len(rxnorm_drug_to_rxnorm_ingredient_df)
    unique_rxnorm = rxnorm_drug_to_rxnorm_ingredient_df['rxnorm_drug_concept_id'].nunique()
    unique_ingredients = rxnorm_drug_to_rxnorm_ingredient_df['rxnorm_ingredient_code'].nunique()
    
    logging.info("%sTotal RxNorm drug to ingredient mappings: %s", INDENT1, f"{total_mappings:,}")
    logging.info("%sUnique RxNorm drug concepts: %s", INDENT1, f"{unique_rxnorm:,}")
    logging.info("%sUnique RxNorm ingredients: %s", INDENT1, f"{unique_ingredients:,}")
    
    # Save intermediate result
    os.makedirs(output_dir, exist_ok=True)

    # Save original version (without mapping columns)
    csv_path_original = os.path.join(output_dir, 'rxnorm_drug_to_rxnorm_ingredient_original.csv')
    rxnorm_drug_to_rxnorm_ingredient_df.to_csv(csv_path_original, index=False)
    logging.info("%sSaved original data (CSV) to: %s", INDENT1, csv_path_original)
    
    # Create mapping version using reusable utility function
    rxnorm_drug_to_rxnorm_ingredient_mapping = create_rx_token_mapping(
        df=rxnorm_drug_to_rxnorm_ingredient_df,
        source_code_col='rxnorm_drug_code',
        target_code_col='rxnorm_ingredient_code',
        target_vocab='RXNORM_INGR',
        source_prefix='RX',
        target_prefix='RX',
        source_vocab='RXNORM'
    )
    
    # Save mapping version (with mapping columns)
    csv_path_mapping = os.path.join(output_dir, 'rxnorm_drug_to_rxnorm_ingredient_mapping.csv')
    rxnorm_drug_to_rxnorm_ingredient_mapping.to_csv(csv_path_mapping, index=False)
    logging.info("%sSaved mapping data (CSV) to: %s", INDENT1, csv_path_mapping)
    
    return rxnorm_drug_to_rxnorm_ingredient_df


@time_execution
def ndc_drug_to_rxnorm_ingredient(ndc_drug_to_rxnorm_drug_df, rxnorm_drug_to_rxnorm_ingredient_df, output_dir):
    """
    Join NDC drug to RxNorm drug and RxNorm drug to ingredient mappings using pandas
    
    Args:
        ndc_drug_to_rxnorm_drug_df: DataFrame with NDC drug to RxNorm drug mappings
        rxnorm_drug_to_rxnorm_ingredient_df: DataFrame with RxNorm drug to ingredient mappings
        output_dir: Output directory for final mapping file
    
    Returns:
        DataFrame with NDC to RxNorm ingredient mappings
    """
    logging.info("Joining NDC to RxNorm ingredient mapping...")
    
    # Use pandas merge to join the two intermediate dataframes
    ndc_drug_to_rxnorm_ingredient_df = pd.merge(
        ndc_drug_to_rxnorm_drug_df,
        rxnorm_drug_to_rxnorm_ingredient_df,
        on=['rxnorm_drug_concept_id', 'rxnorm_drug_code', 'rxnorm_drug_name'],
        how='inner',
        suffixes=('_ndc', '_ingr')
    )
    
    # Use 'Ancestor' relationship_id from the ingredient mapping
    ndc_drug_to_rxnorm_ingredient_df['relationship_id'] = ndc_drug_to_rxnorm_ingredient_df['relationship_id_ingr']
    
    # Select final columns
    ndc_drug_to_rxnorm_ingredient_df = ndc_drug_to_rxnorm_ingredient_df[[
        'ndc_drug_code',
        'ndc_drug_name',
        'ndc_drug_concept_id',
        'rxnorm_drug_code',
        'rxnorm_drug_name',
        'rxnorm_drug_concept_id',
        'rxnorm_ingredient_code',
        'rxnorm_ingredient_name',
        'rxnorm_ingredient_concept_id',
        'relationship_id'
    ]].drop_duplicates()
    
    total_mappings = len(ndc_drug_to_rxnorm_ingredient_df)
    unique_ndc = ndc_drug_to_rxnorm_ingredient_df['ndc_drug_code'].nunique()
    unique_rxnorm = ndc_drug_to_rxnorm_ingredient_df['rxnorm_drug_code'].nunique()
    unique_ingredients = ndc_drug_to_rxnorm_ingredient_df['rxnorm_ingredient_code'].nunique()
    
    logging.info("%sTotal final mappings: %s", INDENT1, f"{total_mappings:,}")
    logging.info("%sUnique NDC codes: %s", INDENT1, f"{unique_ndc:,}")
    logging.info("%sUnique RxNorm codes: %s", INDENT1, f"{unique_rxnorm:,}")
    logging.info("%sUnique RxNorm ingredients: %s", INDENT1, f"{unique_ingredients:,}")
    
    # Save final result
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original version (without mapping columns)
    csv_path_original = os.path.join(output_dir, 'ndc_drug_to_rxnorm_ingredient_original.csv')
    ndc_drug_to_rxnorm_ingredient_df.to_csv(csv_path_original, index=False)
    logging.info("%sSaved original data (CSV) to: %s", INDENT1, csv_path_original)

    # Create mapping version using reusable utility function
    logging.info("%sAdding mapping-specific columns for final NDC to RxNorm ingredient mapping...", INDENT1)
    ndc_drug_to_rxnorm_ingredient_mapping = create_rx_token_mapping(
        df=ndc_drug_to_rxnorm_ingredient_df,
        source_code_col='ndc_drug_code',
        target_code_col='rxnorm_ingredient_code',
        target_vocab='RXNORM_INGR',
        source_prefix='NDCNUM',
        target_prefix='RX',
        source_vocab='NDC'
    )
    
    # Save mapping version (with mapping columns)
    csv_path_mapping = os.path.join(output_dir, 'ndc_drug_to_rxnorm_ingredient_mapping.csv')
    ndc_drug_to_rxnorm_ingredient_mapping.to_csv(csv_path_mapping, index=False)
    logging.info("%sSaved mapping data (CSV) to: %s", INDENT1, csv_path_mapping)


def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging()
    logging.info("NDC Drug to RxNorm Drug; and RxNorm Drug to Ingredient Mapping")
    logging.info("Log file: %s", log_file)
    
    # Parse arguments
    args = parse_args()
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sOHDSI vocabulary directory: %s", INDENT1, args.ohdsi_vocab_dir)
    logging.info("%sOutput directory: %s", INDENT1, args.output_dir)
    
    
    # Step 1: Create NDC to RxNorm drug mapping
    logging.info("********* Step 1: NDC to RxNorm Drug Mapping *********")
    ndc_drug_to_rxnorm_drug_df = ndc_drug_to_rxnorm_drug(args.ohdsi_vocab_dir, args.output_dir)
    
    # Step 2: Create RxNorm drug to ingredient mapping
    logging.info("********* Step 2: RxNorm Drug to Ingredient Mapping *********")
    rxnorm_drug_to_rxnorm_ingredient_df = rxnorm_drug_to_rxnorm_ingredient(args.ohdsi_vocab_dir, args.output_dir)
    
    # Step 3: Join to create final NDC to ingredient mapping
    logging.info("********* Step 3: Final NDC to RxNorm Ingredient Mapping *********")
    ndc_drug_to_rxnorm_ingredient(
        ndc_drug_to_rxnorm_drug_df, 
        rxnorm_drug_to_rxnorm_ingredient_df, 
        args.output_dir
    )
    
    logging.info("NDC to RxNorm mapping completed successfully!")


if __name__ == "__main__":
    main()