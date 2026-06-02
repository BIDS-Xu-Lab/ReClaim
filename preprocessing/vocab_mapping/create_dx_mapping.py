#!/usr/bin/env python3
"""
ICD9 to ICD10 Diagnosis Mapping
Creates mapping from ICD9CM to ICD10CM using CMS GEMs (General Equivalence Mappings)
"""

import os
import argparse
import logging
import time
import pandas as pd

# Import utilities
from utils import setup_logging, format_time_delta, time_execution

# Logging indentation prefixes
INDENT1 = "  "
INDENT2 = "    "


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create ICD9 to ICD10 diagnosis mapping')
    parser.add_argument("--gems_file", type=str, 
                       default='standard_vocab/diagnosis_gems_2018/2018_I9gem.txt',
                       help='Path to CMS GEMs file (ICD9 to ICD10)')
    parser.add_argument("--output_dir", type=str,
                       default='mapping',
                       help='Output directory for all mapping files')
    return parser.parse_args()


def create_diagnosis_token_mapping(df, source_code_col, target_code_col, 
                                    target_vocab, source_prefix, target_prefix, source_vocab):
    """
    Create diagnosis mapping dataframe preserving one-to-many relationships as separate rows
    
    For diagnosis codes, we keep one-to-many mappings as separate rows rather than
    combining them with COMBSTART/COMBEND tokens. This approach is appropriate because:
    
    1. Temporal ICD9/ICD10 Usage: In our longitudinal study period (2008-2024), ICD-9-CM
       was used before October 2015, and ICD-10-CM was applied after. Using COMBSTART/COMBEND
       to aggregate mappings would introduce bias by conflating codes from different time periods.
       
    2. Independent Diagnosis Code Nature: Unlike procedures which may have hierarchical
       relationships ('Is a', 'Has method'), diagnosis codes typically map with 'Maps to' or
       'Maps to approximate' relationships. Each target represents an independent valid mapping
       rather than different aspects of the same concept.
       
    3. Practical Benefits:
       - Simplifies downstream joins (no string parsing needed)
       - Maintains data in normalized form for easier querying
       - Provides flexibility for filtering specific targets or relationship types
       - Follows standard relational database practices
    
    Note: Dots are removed from ICD codes for compatibility with MarketScan data format,
    which stores diagnosis codes without decimal points (e.g., '25000' instead of '250.00').
    
    Args:
        df: DataFrame with source and target code mappings
        source_code_col: Column name containing source codes
        target_code_col: Column name containing target codes
        target_vocab: Target vocabulary name (e.g., 'ICD10CM')
        source_prefix: Token prefix for source (e.g., 'DX')
        target_prefix: Token prefix for target (e.g., 'DX')
        source_vocab: Source vocabulary name (e.g., 'ICD9CM'). 
                     If 'source_vocab' column already exists in df, pass None to use existing.
    
    Returns:
        DataFrame with mapping columns, preserving one-to-many mappings as separate rows
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
    
    # Remove dots from codes when creating tokens (if any exist)
    mapping_df['source_token'] = '<' + source_prefix + '-' + mapping_df[source_code_col].str.replace('.', '', regex=False) + '>'
    mapping_df['target_token'] = '<' + target_prefix + '-' + mapping_df[target_code_col].str.replace('.', '', regex=False) + '>'
    mapping_df['target_value'] = mapping_df[target_code_col].str.replace('.', '', regex=False)
    
    # Calculate num_targets for each source token (how many targets it maps to)
    logging.info("%sCalculating num_targets for each source token...", INDENT1)
    source_token_counts = mapping_df.groupby('source_token').size()
    mapping_df['num_targets'] = mapping_df['source_token'].map(source_token_counts)
    
    # Log statistics about one-to-many mappings
    logging.info("%sAnalyzing mapping cardinality...", INDENT1)
    one_to_many_sources = (source_token_counts > 1).sum()
    one_to_one_sources = (source_token_counts == 1).sum()
    max_targets = source_token_counts.max()
    
    logging.info("%sSource codes with one-to-one mappings: %s", INDENT1, f"{one_to_one_sources:,}")
    logging.info("%sSource codes with one-to-many mappings: %s", INDENT1, f"{one_to_many_sources:,}")
    logging.info("%sMaximum targets per source code: %s", INDENT1, f"{max_targets:,}")
    logging.info("%sTotal mapping rows: %s", INDENT1, f"{len(mapping_df):,}")
    
    # Select and order columns to match proc/rx mapping structure
    final_columns = [
        'source_prefix',
        'target_prefix',
        'source_token',
        'source_vocab',
        'target_vocab',
        'relationship_id',
        'num_targets',
        'target_token',
        'target_value'
    ]
    
    mapping_df_final = mapping_df[final_columns].copy()
    
    return mapping_df_final


@time_execution
def icd9_to_icd10(gems_file, output_dir):
    """
    Create ICD9CM to ICD10CM mapping using CMS GEMs file
    
    The GEMs file format is fixed-width:
    - Columns 1-5: ICD-9-CM code (left-justified, no decimal)
    - Columns 7-13: ICD-10-CM code (left-justified, no decimal)  
    - Columns 15-19: Flags (5 digits)
    
    Flags meaning:
    - Position 1: Approximate (0=no, 1=yes)
    - Position 2: No Map (0=valid map, 1=no acceptable GEM)
    - Position 3: Combination (0=single, 1=combination required)
    - Position 4: Scenario (0=single, 1-6=scenario number)
    - Position 5: Choice list (0=single, 1-n=choice number)
    
    Args:
        gems_file: Path to CMS GEMs file
        output_dir: Directory to save mapping file
    
    Returns:
        DataFrame with ICD9CM to ICD10CM mappings
    """
    logging.info("Creating ICD9CM to ICD10CM mapping from CMS GEMs...")
    logging.info("%sReading GEMs file: %s", INDENT1, gems_file)
    
    # Read fixed-width format file
    # Format: ICD9 (cols 0-5), space, ICD10 (cols 6-13), space, flags (cols 14-19)
    icd9_to_icd10_df = pd.read_fwf(
        gems_file,
        colspecs=[(0, 5), (6, 13), (14, 19)],
        names=['icd9_code', 'icd10_code', 'flags'],
        dtype=str
    )
    
    # Strip whitespace
    icd9_to_icd10_df['icd9_code'] = icd9_to_icd10_df['icd9_code'].str.strip()
    icd9_to_icd10_df['icd10_code'] = icd9_to_icd10_df['icd10_code'].str.strip()
    icd9_to_icd10_df['flags'] = icd9_to_icd10_df['flags'].str.strip()
    
    # Parse flags
    icd9_to_icd10_df['approximate'] = icd9_to_icd10_df['flags'].str[0]
    icd9_to_icd10_df['no_map'] = icd9_to_icd10_df['flags'].str[1]
    icd9_to_icd10_df['combination'] = icd9_to_icd10_df['flags'].str[2]
    icd9_to_icd10_df['scenario'] = icd9_to_icd10_df['flags'].str[3]
    icd9_to_icd10_df['choice_list'] = icd9_to_icd10_df['flags'].str[4]
    
    # Add relationship_id based on mapping type (for consistency with other mappings)
    def get_relationship_id(row):
        if row['no_map'] == '1':
            return 'No map'
        elif row['approximate'] == '1':
            return 'Maps to approximate'
        else:
            return 'Maps to'
    
    icd9_to_icd10_df['relationship_id'] = icd9_to_icd10_df.apply(get_relationship_id, axis=1)
    
    total_rows = len(icd9_to_icd10_df)
    logging.info("%sTotal rows in GEMs file: %s", INDENT1, f"{total_rows:,}")
    
    # Handle "no map" entries (where no_map flag = '1')
    no_map_count = (icd9_to_icd10_df['no_map'] == '1').sum()
    logging.info("%sEntries with no valid mapping (no_map=1): %s", INDENT1, f"{no_map_count:,}")
    
    # Drop rows with no valid mapping (no_map == '1')
    valid_mappings = icd9_to_icd10_df[icd9_to_icd10_df['no_map'] == '0'].copy()
    
    # Log flag statistics
    logging.info("%sFlag Statistics:", INDENT1)
    logging.info("%sApproximate mappings (flag=1): %s", INDENT2, f"{(valid_mappings['approximate'] == '1').sum():,}")
    logging.info("%sExact mappings (flag=0): %s", INDENT2, f"{(valid_mappings['approximate'] == '0').sum():,}")
    logging.info("%sCombination required (flag=1): %s", INDENT2, f"{(valid_mappings['combination'] == '1').sum():,}")
    logging.info("%sSingle code (flag=0): %s", INDENT2, f"{(valid_mappings['combination'] == '0').sum():,}")
    
    # Log relationship statistics
    logging.info("%sRelationship Statistics:", INDENT1)
    for rel_type, count in valid_mappings['relationship_id'].value_counts().items():
        logging.info("%s%s: %s", INDENT2, rel_type, f"{count:,}")
    
    total_mappings = len(valid_mappings)
    unique_icd9 = valid_mappings['icd9_code'].nunique()
    unique_icd10 = valid_mappings['icd10_code'].nunique()
    
    logging.info("%sTotal valid ICD9 to ICD10 mappings: %s", INDENT1, f"{total_mappings:,}")
    logging.info("%sUnique ICD9CM codes: %s", INDENT1, f"{unique_icd9:,}")
    logging.info("%sUnique ICD10CM codes: %s", INDENT1, f"{unique_icd10:,}")
    
    # Save result
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original version (with all flag columns)
    csv_path_original = os.path.join(output_dir, 'icd9_to_icd10_original.csv')
    valid_mappings.to_csv(csv_path_original, index=False)
    logging.info("%sSaved original data (CSV) to: %s", INDENT1, csv_path_original)
    
    # Create mapping version using reusable utility function
    icd9_to_icd10_mapping = create_diagnosis_token_mapping(
        df=valid_mappings,
        source_code_col='icd9_code',
        target_code_col='icd10_code',
        target_vocab='ICD10CM',
        source_prefix='DX',
        target_prefix='DX',
        source_vocab='ICD9CM'
    )
    
    # Save mapping version
    csv_path_mapping = os.path.join(output_dir, 'icd9_to_icd10_mapping.csv')
    icd9_to_icd10_mapping.to_csv(csv_path_mapping, index=False)
    logging.info("%sSaved mapping data (CSV) to: %s", INDENT1, csv_path_mapping)
    
    return valid_mappings


def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging()
    logging.info("ICD9CM to ICD10CM Diagnosis Mapping Generator (using CMS GEMs)")
    logging.info("Log file: %s", log_file)
    
    # Parse arguments
    args = parse_args()
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sGEMs file: %s", INDENT1, args.gems_file)
    logging.info("%sOutput directory: %s", INDENT1, args.output_dir)
    
    # Create ICD9CM to ICD10CM mapping from GEMs
    logging.info("********* ICD9CM to ICD10CM Mapping from CMS GEMs *********")
    icd9_to_icd10_df = icd9_to_icd10(args.gems_file, args.output_dir)
    
    logging.info("ICD9 to ICD10 diagnosis mapping completed successfully!")


if __name__ == "__main__":
    main()
