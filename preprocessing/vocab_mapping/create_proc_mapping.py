#!/usr/bin/env python3
"""
Procedure Code to SNOMED Mapping
Creates mappings from ICD10PCS, ICD9CM, and CPT4 to SNOMED using OMOP vocabulary
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
    parser = argparse.ArgumentParser(description='Create procedure code to SNOMED mappings')
    parser.add_argument("--ohdsi_vocab_dir", type=str, 
                       default='standard_vocab/ohdsi_vocab_v20250827',
                       help='Directory containing OHDSI vocabulary files')
    parser.add_argument("--output_dir", type=str,
                       default='mapping',
                       help='Output directory for all mapping files')
    return parser.parse_args()


def create_procedure_token_mapping(df, source_code_col, target_code_col, 
                                    target_vocab, source_prefix, target_prefix, source_vocab):
    """
    Create procedure mapping dataframe with one-to-many aggregation using COMBSTART/COMBEND tokens
    
    This function aggregates one-to-many mappings into single rows using special combination markers.
    For example, if a procedure code maps to 3 SNOMED codes, they are combined as:
    <PROC-COMBSTART> <PROC-code1> <PROC-code2> <PROC-code3> <PROC-COMBEND>
    
    This approach is appropriate for procedure codes because:
    - Procedure relationships often indicate hierarchical or method relationships ('Is a', 'Has method')
      which represent different aspects of the same procedure concept
    - Procedure codes (CPT, ICD-PCS) can overlap across vocabularies and need deduplication
    - The COMBSTART/COMBEND markers preserve semantic groupings for downstream processing
    
    Note: Dots are removed from codes for compatibility with MarketScan data format,
    which stores ICD procedure codes without decimal points.
    
    Args:
        df: DataFrame with source and target code mappings
        source_code_col: Column name containing source codes
        target_code_col: Column name containing target codes
        target_vocab: Target vocabulary name (e.g., 'SNOMED')
        source_prefix: Token prefix for source (e.g., 'PROC')
        target_prefix: Token prefix for target (e.g., 'PROC')
        source_vocab: Source vocabulary name (e.g., 'ICD10PCS', 'CPT4'). 
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
    # Remove dots from codes when creating tokens
    mapping_df['source_token'] = '<' + source_prefix + '-' + mapping_df[source_code_col].str.replace('.', '', regex=False) + '>'
    mapping_df['target_token_single'] = '<' + target_prefix + '-' + mapping_df[target_code_col].str.replace('.', '', regex=False) + '>'
    mapping_df['target_value_single'] = mapping_df[target_code_col].str.replace('.', '', regex=False)
    
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
        
        # Analyze overlap information before deduplication
        logging.info("%sOverlap Analysis:", INDENT1)
        dup_df = mapping_df_final[mapping_df_final['source_token'].isin(duplicates.index)]
        
        # Count how many duplicate tokens each vocab is involved in
        vocab_overlap_counts = dup_df.groupby('source_vocab')['source_token'].nunique()
        
        # Count total tokens per vocab (before dedup)
        vocab_total_counts = mapping_df_final.groupby('source_vocab')['source_token'].nunique()
        
        # Calculate overlap percentage for each vocab
        for vocab in vocab_overlap_counts.index:
            overlap_count = vocab_overlap_counts[vocab]
            total_count = vocab_total_counts.get(vocab, 0)
            overlap_pct = (overlap_count / total_count * 100) if total_count > 0 else 0
            logging.info("%s  %s: %s overlapping tokens out of %s total (%.2f%%)", 
                        INDENT2, vocab, f"{overlap_count:,}", f"{total_count:,}", overlap_pct)
        
        # Analyze which vocab pairs have overlaps
        vocab_pairs = dup_df.groupby('source_token')['source_vocab'].apply(lambda x: tuple(sorted(x.unique()))).value_counts()
        logging.info("%sVocab pair overlap counts:", INDENT1)
        for pair, count in vocab_pairs.items():
            logging.info("%s  %s: %s overlaps", INDENT2, pair, f"{count:,}")
        
        # Apply priority-based deduplication using source_vocab
        # Priority: CPT4 > ICD9CM > ICD10PCS (lower number = higher priority)
        priority_map = {'CPT4': 1, 'ICD9CM': 2, 'ICD10PCS': 3}
        mapping_df_final['_priority'] = mapping_df_final['source_vocab'].map(priority_map).fillna(99)
        
        # Log which duplicates will be resolved
        n_example = 5
        duplicate_tokens = duplicates.index.tolist()[:n_example]
        logging.info("%sResolving duplicates using priority (CPT4 > ICD9CM > ICD10PCS):", INDENT1)
        for token in duplicate_tokens:
            dup_rows = mapping_df_final[mapping_df_final['source_token'] == token].sort_values('_priority')
            vocabs = dup_rows['source_vocab'].tolist()
            kept_vocab = vocabs[0] if vocabs else 'unknown'
            logging.info("%s  %s -> vocabs: %s, keeping: %s", INDENT2, token, vocabs, kept_vocab)
        
        # Keep only the highest priority (lowest _priority value) for each source_token
        before_dedup = len(mapping_df_final)
        mapping_df_final = mapping_df_final.sort_values('_priority').drop_duplicates(subset=['source_token'], keep='first')
        after_dedup = len(mapping_df_final)
        mapping_df_final = mapping_df_final.drop(columns=['_priority'])
        
        logging.info("%sRemoved %s duplicate mappings using priority", INDENT1, f"{before_dedup - after_dedup:,}")
        logging.info("%sVerified: Each source token now maps to exactly one target token", INDENT1)
    else:
        logging.info("%sVerified: Each source token maps to exactly one target token", INDENT1)
    
    # Log statistics about one-to-many mappings
    one_to_many = (mapping_df_final['num_targets'] >= 2).sum()
    one_to_one = (mapping_df_final['num_targets'] == 1).sum()
    logging.info("%sOne-to-one mappings: %s", INDENT1, f"{one_to_one:,}")
    logging.info("%sOne-to-many mappings: %s", INDENT1, f"{one_to_many:,}")
    
    return mapping_df_final


@time_execution
def icd10pcs_to_snomed(ohdsi_vocab_dir, output_dir):
    """
    Create ICD10PCS to SNOMED mapping using DuckDB
    
    Args:
        ohdsi_vocab_dir: Directory containing OHDSI vocabulary CSV files
        output_dir: Directory to save mapping file
    
    Returns:
        DataFrame with ICD10PCS to SNOMED mappings
    """
    logging.info("Creating ICD10PCS to SNOMED mapping...")
    
    # Build paths to CSV files
    concept_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT.csv')
    concept_relationship_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT_RELATIONSHIP.csv')
    
    logging.info("%sReading OMOP vocabulary from: %s", INDENT1, ohdsi_vocab_dir)
    logging.info("%sConcept: %s", INDENT2, concept_path)
    logging.info("%sConcept Relationship: %s", INDENT2, concept_relationship_path)
    
    # Execute the ICD10PCS to SNOMED mapping query
    # Use INNER JOIN to get only mapped codes, DENSE_RANK() to keep only one relationship per source
    icd10pcs_snomed_df = duckdb.sql(f"""
    WITH ranked_mappings AS (
        SELECT
            c_icd10pcs.concept_id AS icd10pcs_concept_id,
            c_icd10pcs.concept_code AS icd10pcs_code,
            c_icd10pcs.concept_name AS icd10pcs_name,
            c_snomed.concept_id AS snomed_concept_id,
            c_snomed.concept_code AS snomed_code,
            c_snomed.concept_name AS snomed_name,
            cr.relationship_id,
            DENSE_RANK() OVER (
                PARTITION BY c_icd10pcs.concept_id 
                ORDER BY CASE cr.relationship_id
                    WHEN 'Is a' THEN 1
                    WHEN 'Has method' THEN 2
                    ELSE 99
                END
            ) AS rn
        FROM read_csv_auto('{concept_path}') c_icd10pcs
        JOIN read_csv_auto('{concept_relationship_path}') cr
            ON cr.concept_id_1 = c_icd10pcs.concept_id
            AND cr.relationship_id IN ('Is a', 'Has method')
        JOIN read_csv_auto('{concept_path}') c_snomed
            ON c_snomed.concept_id = cr.concept_id_2
            AND c_snomed.vocabulary_id = 'SNOMED'
            AND c_snomed.standard_concept = 'S'
        WHERE c_icd10pcs.vocabulary_id = 'ICD10PCS'
            AND LENGTH(c_icd10pcs.concept_code) = 7
    )
    SELECT
        icd10pcs_concept_id,
        icd10pcs_code,
        icd10pcs_name,
        snomed_concept_id,
        snomed_code,
        snomed_name,
        relationship_id
    FROM ranked_mappings
    WHERE rn = 1
    ORDER BY icd10pcs_code
    """).df()
    
    total_mappings = len(icd10pcs_snomed_df)
    unique_icd10pcs = icd10pcs_snomed_df['icd10pcs_code'].nunique()
    unique_snomed = icd10pcs_snomed_df['snomed_code'].nunique()
    
    logging.info("%sTotal ICD10PCS to SNOMED mappings: %s", INDENT1, f"{total_mappings:,}")
    logging.info("%sUnique ICD10PCS codes: %s", INDENT1, f"{unique_icd10pcs:,}")
    logging.info("%sUnique SNOMED codes: %s", INDENT1, f"{unique_snomed:,}")
    
    # Save result
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original version
    csv_path_original = os.path.join(output_dir, 'icd10pcs_to_snomed_procedure_original.csv')
    icd10pcs_snomed_df.to_csv(csv_path_original, index=False)
    logging.info("%sSaved original data (CSV) to: %s", INDENT1, csv_path_original)
    
    # Create mapping version using reusable utility function
    icd10pcs_snomed_mapping = create_procedure_token_mapping(
        df=icd10pcs_snomed_df,
        source_code_col='icd10pcs_code',
        target_code_col='snomed_code',
        target_vocab='SNOMED',
        source_prefix='PROC',
        target_prefix='PROC',
        source_vocab='ICD10PCS'
    )
    
    # Save mapping version
    csv_path_mapping = os.path.join(output_dir, 'icd10pcs_to_snomed_procedure_mapping.csv')
    icd10pcs_snomed_mapping.to_csv(csv_path_mapping, index=False)
    logging.info("%sSaved mapping data (CSV) to: %s", INDENT1, csv_path_mapping)
    
    return icd10pcs_snomed_df


@time_execution
def icd9cm_to_snomed(ohdsi_vocab_dir, output_dir):
    """
    Create icd9cm to SNOMED mapping using DuckDB
    
    Args:
        ohdsi_vocab_dir: Directory containing OHDSI vocabulary CSV files
        output_dir: Directory to save mapping file
    
    Returns:
        DataFrame with icd9cm to SNOMED mappings
    """
    logging.info("Creating ICD9CM to SNOMED mapping...")
    
    # Build paths to CSV files
    concept_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT.csv')
    concept_relationship_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT_RELATIONSHIP.csv')
    
    logging.info("%sReading OMOP vocabulary from: %s", INDENT1, ohdsi_vocab_dir)
    logging.info("%sConcept: %s", INDENT2, concept_path)
    logging.info("%sConcept Relationship: %s", INDENT2, concept_relationship_path)
    
    # Execute the icd9cm to SNOMED mapping query
    # Use INNER JOIN to get only mapped codes, DENSE_RANK() to keep only one relationship per source
    # Priority: 'Maps to' > 'Is a'
    icd9cm_snomed_df = duckdb.sql(f"""
    WITH ranked_mappings AS (
        SELECT
            c_icd9cm.concept_id AS icd9cm_concept_id,
            c_icd9cm.concept_code AS icd9cm_code,
            c_icd9cm.concept_name AS icd9cm_name,
            c_snomed.concept_id AS snomed_concept_id,
            c_snomed.concept_code AS snomed_code,
            c_snomed.concept_name AS snomed_name,
            cr.relationship_id,
            DENSE_RANK() OVER (
                PARTITION BY c_icd9cm.concept_id 
                ORDER BY CASE cr.relationship_id
                    WHEN 'Maps to' THEN 1
                    ELSE 99
                END
            ) AS rn
        FROM read_csv_auto('{concept_path}') c_icd9cm
        JOIN read_csv_auto('{concept_relationship_path}') cr
            ON cr.concept_id_1 = c_icd9cm.concept_id
            AND cr.relationship_id = 'Maps to'
        JOIN read_csv_auto('{concept_path}') c_snomed
            ON c_snomed.concept_id = cr.concept_id_2
            AND c_snomed.vocabulary_id = 'SNOMED'
            AND c_snomed.standard_concept = 'S'
        WHERE c_icd9cm.vocabulary_id = 'ICD9CM'
    )
    SELECT
        icd9cm_concept_id,
        icd9cm_code,
        icd9cm_name,
        snomed_concept_id,
        snomed_code,
        snomed_name,
        relationship_id
    FROM ranked_mappings
    WHERE rn = 1
    ORDER BY icd9cm_code
    """).df()
    
    total_mappings = len(icd9cm_snomed_df)
    unique_icd9cm = icd9cm_snomed_df['icd9cm_code'].nunique()
    unique_snomed = icd9cm_snomed_df['snomed_code'].nunique()
    
    logging.info("%sTotal ICD9CM to SNOMED mappings: %s", INDENT1, f"{total_mappings:,}")
    logging.info("%sUnique ICD9CM codes: %s", INDENT1, f"{unique_icd9cm:,}")
    logging.info("%sUnique SNOMED codes: %s", INDENT1, f"{unique_snomed:,}")
    
    # Save result
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original version
    csv_path_original = os.path.join(output_dir, 'icd9cm_to_snomed_procedure_original.csv')
    icd9cm_snomed_df.to_csv(csv_path_original, index=False)
    logging.info("%sSaved original data (CSV) to: %s", INDENT1, csv_path_original)
    
    # Create mapping version using reusable utility function
    icd9cm_snomed_mapping = create_procedure_token_mapping(
        df=icd9cm_snomed_df,
        source_code_col='icd9cm_code',
        target_code_col='snomed_code',
        target_vocab='SNOMED',
        source_prefix='PROC',
        target_prefix='PROC',
        source_vocab='ICD9CM'
    )
    
    # Save mapping version
    csv_path_mapping = os.path.join(output_dir, 'icd9cm_to_snomed_procedure_mapping.csv')
    icd9cm_snomed_mapping.to_csv(csv_path_mapping, index=False)
    logging.info("%sSaved mapping data (CSV) to: %s", INDENT1, csv_path_mapping)
    
    return icd9cm_snomed_df


@time_execution
def cpt_to_snomed(ohdsi_vocab_dir, output_dir):
    """
    Create CPT4 to SNOMED mapping using DuckDB
    Uses LEFT JOIN to include all CPT4 codes, with fallback for unmapped codes
    
    Args:
        ohdsi_vocab_dir: Directory containing OHDSI vocabulary CSV files
        output_dir: Directory to save mapping file
    
    Returns:
        DataFrame with CPT4 to SNOMED mappings (all CPT4 codes, with fallback for unmapped)
    """
    logging.info("Creating CPT4 to SNOMED mapping...")
    
    # Build paths to CSV files
    concept_cpt4_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT_CPT4.csv') # separated file for CPT4 per OHDSI format
    concept_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT.csv')
    concept_relationship_path = os.path.join(ohdsi_vocab_dir, 'CONCEPT_RELATIONSHIP.csv')
    
    logging.info("%sReading OMOP vocabulary from: %s", INDENT1, ohdsi_vocab_dir)
    logging.info("%sConcept: %s", INDENT2, concept_path)
    logging.info("%sConcept CPT4: %s", INDENT2, concept_cpt4_path)
    logging.info("%sConcept Relationship: %s", INDENT2, concept_relationship_path)
    
    # Execute the CPT4 to SNOMED mapping query
    # Use LEFT JOIN to keep all CPT4 codes, DENSE_RANK() to keep only one relationship per source
    # Priority: 'CPT4 - SNOMED eq' > 'Maps to' > 'Is a' > 'CPT4 - SNOMED cat'
    # Fallback: unmapped CPT4 codes use 'CPT4' + concept_code as snomed_code
    cpt_snomed_df = duckdb.sql(f"""
    WITH cpt4_len5 AS (
        SELECT concept_id, concept_code, concept_name, vocabulary_id
        FROM read_csv_auto('{concept_cpt4_path}')
        WHERE vocabulary_id = 'CPT4'
            AND LENGTH(concept_code) = 5
    ),
    ranked_mappings AS (
        SELECT
            cpt.concept_id AS cpt_concept_id,
            c_snomed.concept_id AS snomed_concept_id,
            c_snomed.concept_code AS snomed_code,
            c_snomed.concept_name AS snomed_name,
            cr.relationship_id,
            DENSE_RANK() OVER (
                PARTITION BY cpt.concept_id 
                ORDER BY CASE cr.relationship_id
                    WHEN 'CPT4 - SNOMED eq' THEN 1
                    WHEN 'Maps to' THEN 2
                    WHEN 'Is a' THEN 3
                    WHEN 'CPT4 - SNOMED cat' THEN 4
                    ELSE 99
                END
            ) AS rn
        FROM cpt4_len5 cpt
        JOIN read_csv_auto('{concept_relationship_path}') cr
            ON cpt.concept_id = cr.concept_id_1
            AND cr.relationship_id IN ('Maps to', 'CPT4 - SNOMED eq', 'Is a', 'CPT4 - SNOMED cat')
        JOIN read_csv_auto('{concept_path}') c_snomed
            ON cr.concept_id_2 = c_snomed.concept_id
            AND c_snomed.vocabulary_id = 'SNOMED'
            AND c_snomed.standard_concept = 'S'
    )
    SELECT
        cpt.concept_id AS cpt_concept_id,
        cpt.concept_code AS cpt_code,
        cpt.concept_name AS cpt_name,
        rm.snomed_concept_id,
        COALESCE(rm.snomed_code, 'CPT4' || cpt.concept_code) AS snomed_code,
        COALESCE(rm.snomed_name, cpt.concept_name) AS snomed_name,
        COALESCE(rm.relationship_id, 'Not mapped') AS relationship_id
    FROM cpt4_len5 cpt
    LEFT JOIN (SELECT * FROM ranked_mappings WHERE rn = 1) rm 
        ON cpt.concept_id = rm.cpt_concept_id
    ORDER BY cpt.concept_code
    """).df()
    
    total_rows = len(cpt_snomed_df)
    unique_cpt = cpt_snomed_df['cpt_code'].nunique()
    unique_snomed = cpt_snomed_df['snomed_code'].nunique()
    
    # Count mapped vs unmapped
    mapped_count = (cpt_snomed_df['relationship_id'] != 'Not mapped').sum()
    unmapped_count = (cpt_snomed_df['relationship_id'] == 'Not mapped').sum()
    
    logging.info("%sTotal CPT4 codes: %s", INDENT1, f"{total_rows:,}")
    logging.info("%sCPT4 codes with SNOMED mapping: %s", INDENT1, f"{mapped_count:,}")
    logging.info("%sCPT4 codes without mapping (fallback): %s", INDENT1, f"{unmapped_count:,}")
    logging.info("%sUnique CPT4 codes: %s", INDENT1, f"{unique_cpt:,}")
    logging.info("%sUnique SNOMED codes: %s", INDENT1, f"{unique_snomed:,}")
    
    # Save result
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original version
    csv_path_original = os.path.join(output_dir, 'cpt_to_snomed_procedure_original.csv')
    cpt_snomed_df.to_csv(csv_path_original, index=False)
    logging.info("%sSaved original data (CSV) to: %s", INDENT1, csv_path_original)
    
    # Create mapping version using reusable utility function
    cpt_snomed_mapping = create_procedure_token_mapping(
        df=cpt_snomed_df,
        source_code_col='cpt_code',
        target_code_col='snomed_code',
        target_vocab='SNOMED',
        source_prefix='PROC',
        target_prefix='PROC',
        source_vocab='CPT4'
    )
    
    # Save mapping version
    csv_path_mapping = os.path.join(output_dir, 'cpt_to_snomed_procedure_mapping.csv')
    cpt_snomed_mapping.to_csv(csv_path_mapping, index=False)
    logging.info("%sSaved mapping data (CSV) to: %s", INDENT1, csv_path_mapping)
    
    return cpt_snomed_df


@time_execution
def analyze_code_overlaps(icd10pcs_snomed_df, icd9cm_snomed_df, cpt_snomed_df):
    """
    Analyze overlaps between source codes and their SNOMED mappings
    
    Args:
        icd10pcs_snomed_df: DataFrame with ICD10PCS to SNOMED mappings
        icd9cm_snomed_df: DataFrame with icd9cm to SNOMED mappings
        cpt_snomed_df: DataFrame with CPT4 to SNOMED mappings
    
    Returns:
        float: Overlap rate (proportion of overlapping codes relative to total unique codes)
    """
    logging.info("Analyzing code overlaps...")
    
    # Extract source codes from each mapping
    icd10pcs_codes = set(icd10pcs_snomed_df['icd10pcs_code'].unique())
    icd9cm_codes = set(icd9cm_snomed_df['icd9cm_code'].unique())
    cpt_codes = set(cpt_snomed_df['cpt_code'].unique())
    
    # Check for source code overlaps (should be none since they're different vocabularies)
    
    logging.info("%sSource Code Overlap Analysis:", INDENT1)
    logging.info("%sTotal ICD10PCS codes: %s", INDENT2, f"{len(icd10pcs_codes):,}")
    logging.info("%sTotal ICD9CM codes: %s", INDENT2, f"{len(icd9cm_codes):,}")
    logging.info("%sTotal CPT4 codes: %s", INDENT2, f"{len(cpt_codes):,}")
    
    icd10pcs_icd9cm_overlap = icd10pcs_codes & icd9cm_codes
    icd10pcs_cpt_overlap = icd10pcs_codes & cpt_codes
    icd9cm_cpt_overlap = icd9cm_codes & cpt_codes
    
    logging.info("%sICD10PCS X ICD9CM: %s codes", INDENT2, f"{len(icd10pcs_icd9cm_overlap):,}")
    logging.info("%sICD10PCS X CPT4: %s codes", INDENT2, f"{len(icd10pcs_cpt_overlap):,}")
    logging.info("%sICD9CM X CPT4: %s codes", INDENT2, f"{len(icd9cm_cpt_overlap):,}")
    
    # Calculate overlap rate
    total_unique_codes = len(icd10pcs_codes | icd9cm_codes | cpt_codes)
    total_overlapping_codes = (len(icd10pcs_icd9cm_overlap) + len(icd10pcs_cpt_overlap) +
                               len(icd9cm_cpt_overlap))
    overlap_rate = total_overlapping_codes / total_unique_codes if total_unique_codes > 0 else 0.0
    
    logging.info("%sTotal unique codes across all sources: %s", INDENT2, f"{total_unique_codes:,}")
    logging.info("%sTotal overlapping codes: %s", INDENT2, f"{total_overlapping_codes:,}")
    logging.info("%sSource code overlap rate: %s (%s%%)", INDENT2, f"{overlap_rate:.6f}", f"{overlap_rate*100:.4f}")
    
    if overlap_rate == 0:
        logging.info("%sNo source code overlaps detected - vocabularies are disjoint", INDENT2)
    else:
        logging.info("%sSource code overlaps detected!", INDENT2)
    
    # Get unique SNOMED codes from each source
    icd10pcs_snomed_set = set(icd10pcs_snomed_df['snomed_code'].unique())
    icd9cm_snomed_set = set(icd9cm_snomed_df['snomed_code'].unique())
    cpt_snomed_set = set(cpt_snomed_df['snomed_code'].unique())
    
    # Check for SNOMED code overlaps
    
    logging.info("%sSNOMED Code Overlap Analysis:", INDENT1)
    logging.info("%sUnique SNOMED codes from ICD10PCS: %s", INDENT2, f"{len(icd10pcs_snomed_set):,}")
    logging.info("%sUnique SNOMED codes from ICD9CM: %s", INDENT2, f"{len(icd9cm_snomed_set):,}")
    logging.info("%sUnique SNOMED codes from CPT4: %s", INDENT2, f"{len(cpt_snomed_set):,}")
    
    icd10pcs_icd9cm_snomed_overlap = icd10pcs_snomed_set & icd9cm_snomed_set
    icd10pcs_cpt_snomed_overlap = icd10pcs_snomed_set & cpt_snomed_set
    icd9cm_cpt_snomed_overlap = icd9cm_snomed_set & cpt_snomed_set

    all_snomed_overlap = icd10pcs_snomed_set & icd9cm_snomed_set & cpt_snomed_set
    
    logging.info("%sSNOMED overlap (ICD10PCS X ICD9CM): %s codes", INDENT2, f"{len(icd10pcs_icd9cm_snomed_overlap):,}")
    logging.info("%sSNOMED overlap (ICD10PCS X CPT4): %s codes", INDENT2, f"{len(icd10pcs_cpt_snomed_overlap):,}")
    logging.info("%sSNOMED overlap (ICD9CM X CPT4): %s codes", INDENT2, f"{len(icd9cm_cpt_snomed_overlap):,}")
    logging.info("%sSNOMED overlap (all): %s codes", INDENT2, f"{len(all_snomed_overlap):,}")
    
    # Calculate unique SNOMED codes per source (exclusive to that source)
    icd10pcs_only = icd10pcs_snomed_set - icd9cm_snomed_set - cpt_snomed_set
    icd9cm_only = icd9cm_snomed_set - icd10pcs_snomed_set - cpt_snomed_set
    cpt_only = cpt_snomed_set - icd10pcs_snomed_set - icd9cm_snomed_set
    
    
    logging.info("%sExclusive SNOMED codes (not shared with other sources):", INDENT1)
    logging.info("%sICD10PCS only: %s codes", INDENT2, f"{len(icd10pcs_only):,}")
    logging.info("%sICD9CM only: %s codes", INDENT2, f"{len(icd9cm_only):,}")
    logging.info("%sCPT4 only: %s codes", INDENT2, f"{len(cpt_only):,}")
    
    return overlap_rate


@time_execution
def aggregate_proc_to_snomed(icd10pcs_snomed_df, icd9cm_snomed_df, cpt_snomed_df, output_dir, overlap_rate):
    """
    Aggregate SNOMED codes from all mappings and create combined mapping
    
    Args:
        icd10pcs_snomed_df: DataFrame with ICD10PCS to SNOMED mappings
        icd9cm_snomed_df: DataFrame with icd9cm to SNOMED mappings
        cpt_snomed_df: DataFrame with CPT4 to SNOMED mappings
        output_dir: Directory to save aggregated results
        overlap_rate: Source code overlap rate (proportion of overlapping codes)
    
    Returns:
        DataFrame with all unique SNOMED codes
    """
    logging.info("Aggregating SNOMED codes from all mappings...")
    
    # Define overlap threshold
    OVERLAP_THRESHOLD = 0.001
    
    if overlap_rate < OVERLAP_THRESHOLD:
        logging.info("%sOverlap rate (%s) is below threshold (%s)", INDENT1, f"{overlap_rate:.6f}", OVERLAP_THRESHOLD)
        logging.info("%sUsing concatenation strategy (source codes are disjoint)", INDENT1)
        
        # Extract source codes and SNOMED codes from each mapping
        # Keep source_vocab for tracking, rename to 'proc_code' for unified mapping
        icd10pcs_mapping = icd10pcs_snomed_df[['icd10pcs_code', 'snomed_code', 'snomed_name', 'snomed_concept_id', 'relationship_id']].copy()
        icd10pcs_mapping['source_vocab'] = 'ICD10PCS'
        icd10pcs_mapping['proc_code'] = icd10pcs_mapping['icd10pcs_code']
        icd10pcs_mapping = icd10pcs_mapping.drop(columns=['icd10pcs_code'])
        
        icd9cm_mapping = icd9cm_snomed_df[['icd9cm_code', 'snomed_code', 'snomed_name', 'snomed_concept_id', 'relationship_id']].copy()
        icd9cm_mapping['source_vocab'] = 'ICD9CM'
        icd9cm_mapping['proc_code'] = icd9cm_mapping['icd9cm_code']
        icd9cm_mapping = icd9cm_mapping.drop(columns=['icd9cm_code'])
        
        cpt_mapping = cpt_snomed_df[['cpt_code', 'snomed_code', 'snomed_name', 'snomed_concept_id', 'relationship_id']].copy()
        cpt_mapping['source_vocab'] = 'CPT4'
        cpt_mapping['proc_code'] = cpt_mapping['cpt_code']
        cpt_mapping = cpt_mapping.drop(columns=['cpt_code'])
        
        # Concatenate all mappings
        proc_to_snomed = pd.concat([
            icd10pcs_mapping,
            icd9cm_mapping,
            cpt_mapping
        ], ignore_index=True)
        
    else:
        # Overlap rate is too high - exit with error
        error_msg = (
            f"\n{'='*80}\n"
            f"ERROR: Source code overlap rate ({overlap_rate:.6f} = {overlap_rate*100:.4f}%) exceeds threshold ({OVERLAP_THRESHOLD})\n"
            f"{'='*80}\n"
            f"Code overlaps exist between source vocabularies (ICD10PCS, ICD9CM, CPT4).\n"
            f"The mapping needs to be done separately for procedures from different coding sources.\n"
            f"{'='*80}\n"
        )
        logging.error(error_msg)
        import sys
        sys.exit(1)
    
    # Calculate statistics
    total_mappings = len(proc_to_snomed)
    unique_proc_codes = proc_to_snomed['proc_code'].nunique()
    distinct_target_codes = proc_to_snomed['snomed_code'].nunique()
    
    # Count actual SNOMED mappings vs CPT4 fallbacks
    # CPT4 fallbacks have relationship_id = 'Not mapped' and snomed_code starts with 'CPT4'
    is_actual_snomed = proc_to_snomed['relationship_id'] != 'Not mapped'
    actual_snomed_count = is_actual_snomed.sum()
    cpt4_fallback_count = (~is_actual_snomed).sum()
    
    # Calculate percentages
    actual_snomed_pct = (actual_snomed_count / total_mappings * 100) if total_mappings > 0 else 0
    cpt4_fallback_pct = (cpt4_fallback_count / total_mappings * 100) if total_mappings > 0 else 0
    
    # Count distinct target codes by type
    actual_snomed_distinct = proc_to_snomed[is_actual_snomed]['snomed_code'].nunique()
    cpt4_fallback_distinct = proc_to_snomed[~is_actual_snomed]['snomed_code'].nunique()
    
    logging.info("%sOverall Statistics:", INDENT1)
    logging.info("%sTotal procedure to target mapping rows: %s", INDENT2, f"{total_mappings:,}")
    logging.info("%sUnique procedure codes: %s", INDENT2, f"{unique_proc_codes:,}")
    logging.info("%sDistinct target codes (SNOMED + CPT4 fallback): %s", INDENT2, f"{distinct_target_codes:,}")
    
    logging.info("%sMapping Breakdown:", INDENT1)
    logging.info("%sMapped to actual SNOMED: %s rows (%.2f%%)", INDENT2, f"{actual_snomed_count:,}", actual_snomed_pct)
    logging.info("%sCPT4 fallback (Not mapped): %s rows (%.2f%%)", INDENT2, f"{cpt4_fallback_count:,}", cpt4_fallback_pct)
    logging.info("%sDistinct actual SNOMED codes: %s", INDENT2, f"{actual_snomed_distinct:,}")
    logging.info("%sDistinct CPT4 fallback codes: %s", INDENT2, f"{cpt4_fallback_distinct:,}")
    
    # Log statistics by source vocabulary
    proc_unique = proc_to_snomed.drop_duplicates(subset=['proc_code', 'source_vocab'])
    logging.info("%sStatistics by Source Vocabulary:", INDENT1)
    for vocab in ['ICD10PCS', 'ICD9CM', 'CPT4']:
        vocab_df = proc_to_snomed[proc_to_snomed['source_vocab'] == vocab]
        vocab_df_unique = proc_unique[proc_unique['source_vocab'] == vocab]
        vocab_total_unique = len(vocab_df_unique)
        vocab_mapped = (vocab_df['relationship_id'] != 'Not mapped').sum()
        vocab_unmapped = (vocab_df['relationship_id'] == 'Not mapped').sum()
        vocab_mapped_pct = (vocab_mapped / len(vocab_df) * 100) if len(vocab_df) > 0 else 0
        logging.info("%s%s: %s unique codes (%s mapped to SNOMED [%.2f%%], %s fallback)", 
                    INDENT2, vocab, f"{vocab_total_unique:,}", 
                    f"{vocab_mapped:,}", vocab_mapped_pct, f"{vocab_unmapped:,}")
    
    # Save unique SNOMED codes
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'proc_to_snomed_original.csv')
    proc_to_snomed.to_csv(csv_path, index=False)
    
    logging.info("%sSaved unique SNOMED codes to: %s", INDENT1, csv_path)
    
    # Create combined mapping using create_procedure_token_mapping with unified PROC prefix
    logging.info("%sCreating combined PROC to SNOMED mapping...", INDENT1)
    proc_to_snomed_mapping = create_procedure_token_mapping(
        df=proc_to_snomed,
        source_code_col='proc_code',
        target_code_col='snomed_code',
        target_vocab='SNOMED',
        source_prefix='PROC',
        target_prefix='PROC',
        source_vocab=None  # source_vocab column already exists in df
    )
    
    
    # Save combined mapping
    csv_path_mapping = os.path.join(output_dir, 'proc_to_snomed_mapping.csv')
    proc_to_snomed_mapping.to_csv(csv_path_mapping, index=False)
    logging.info("%sSaved combined PROC to SNOMED mapping to: %s", INDENT1, csv_path_mapping)


def main():
    """Main execution function"""
    # Set up logging
    log_file = setup_logging()
    
    logging.info("Procedure Code to SNOMED Mapping Generator")
    logging.info("Log file: %s", log_file)
    
    # Parse arguments
    args = parse_args()
    
    # Log configuration
    logging.info("Configuration:")
    logging.info("%sOHDSI vocabulary directory: %s", INDENT1, args.ohdsi_vocab_dir)
    logging.info("%sOutput directory: %s", INDENT1, args.output_dir)
    
    # Step 1: Create ICD10PCS to SNOMED mapping
    logging.info("********* Step 1: ICD10PCS to SNOMED Mapping *********")
    icd10pcs_snomed_df = icd10pcs_to_snomed(args.ohdsi_vocab_dir, args.output_dir)
    
    # Step 2: Create ICD9CM to SNOMED mapping  
    logging.info("********* Step 2: ICD9CM to SNOMED Mapping *********")
    icd9cm_snomed_df = icd9cm_to_snomed(args.ohdsi_vocab_dir, args.output_dir)
    
    # Step 3: Create CPT4 to SNOMED mapping
    logging.info("********* Step 3: CPT4 to SNOMED Mapping *********")
    cpt_snomed_df = cpt_to_snomed(args.ohdsi_vocab_dir, args.output_dir)
    
    # Step 4: Analyze code overlaps
    logging.info("********* Step 4: Analyzing Code Overlaps *********")
    overlap_rate = analyze_code_overlaps(
        icd10pcs_snomed_df, icd9cm_snomed_df, cpt_snomed_df
    )
    
    # Step 5: Aggregate SNOMED codes
    logging.info("********* Step 5: Aggregating SNOMED Codes *********")
    aggregate_proc_to_snomed(
        icd10pcs_snomed_df, icd9cm_snomed_df, cpt_snomed_df, args.output_dir, overlap_rate
    )
    
    logging.info("Procedure code to SNOMED mapping completed successfully!")


if __name__ == "__main__":
    main()
