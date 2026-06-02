import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from datetime import datetime
import argparse
import logging
from utils import setup_logging, time_execution


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cohort extraction and processing for antidiabetic drug study')
    parser.add_argument('--version', type=int, 
                       default=6,  # 4, 5, 6
                       help='Data version')
    return parser.parse_args()


# Parse arguments
args = parse_args()

# Set input and output paths
working_dir = os.getcwd()
intermediate_result_dir = os.path.join(working_dir, 'intermediate', f'v{args.version}')
code_set_folder_dir = os.path.join(working_dir, 'lookup_table')
output_dir = os.path.join(intermediate_result_dir, 'final_cohort')

demo_file_dir = os.path.join(intermediate_result_dir, 'df_processed', 'demo_df')
icd_codes_file = os.path.join(code_set_folder_dir, 'outcome_cov_icd.csv')
dx_dir = os.path.join(intermediate_result_dir, 'df_processed', 'dx_df')
med_dir = os.path.join(intermediate_result_dir, 'df_processed', 'med_df')
core_dir = os.path.join(intermediate_result_dir, 'df_processed', 'core_df')
# lab_dir = ' '
# vital_dir = ' '

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define year range
years = range(2018, 2025)

# Define antidiabetic drug classes and keywords
ANTIDIABETIC_DRUGS = {
    'glp1': ['Albiglutide', 'Dulaglutide', 'Exenatide', 'Liraglutide', 'Lixisenatide', 'Semaglutide', 'Tirzepatide'],
    'sglt2': ['Canagliflozin', 'Dapagliflozin', 'Empagliflozin', 'Ertugliflozin', 'Sotagliflozin', 'Bexagliflozin'],
    'dpp4': ['Sitagliptin', 'Saxagliptin', 'Alogliptin', 'Linagliptin']
}

# Define file types and their required columns (in lowercase)
file_types = {
    'Dx': ['pid', 'dx_date', 'code'],
    'Med': ['pid', 'start_date', 'medication_name'],
    # Comment out other tables
    'Core': ['pid', 'visit_date', 'patient_class'],
    'Lab': ['pid', 'result_loinc_code', 'result_num', 'result_date'],
    'Vital': ['pid', 'vital_name', 'result_vital', 'recorded_time']
}

# Define date columns for each file type
date_columns = {
    'Dx': ['dx_date'],
    'Med': ['start_date'],
    # Comment out other tables
    'Core': ['visit_date'],
    'Lab': ['result_date'],
    'Vital': ['recorded_time']
}

def convert_date_columns(df, date_cols):
    """
    Convert date columns to datetime and extract date only
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    date_cols : list
        List of date column names to convert
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with converted date columns
    """
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), errors='coerce').dt.date
    return df

def read_and_combine_files(file_type, year):
    """
    Read and combine files for a specific year and file type
    
    Parameters:
    -----------
    file_type : str
        Type of file to process (Dx, Core, Med, Lab, or Vital)
    year : int
        Year to process
        
    Returns:
    --------
    pandas.DataFrame or None
        Combined dataframe for the specified year and file type, or None if no files found
    """
    logging.info(f"Processing {file_type} files for year {year}...")
    
    # Get all matching files for the year
    pattern = f"{input_base_dir}/{year}/{file_type}_*.csv"
    files = sorted(glob.glob(pattern))
    
    if not files:
        logging.info(f"No files found for {file_type} in {year}")
        return None
    
    # Read and combine all files
    dfs = []
    for file in files:
        try:
            # Read the file with all columns
            df = pd.read_csv(file, low_memory=False)
            
            # Convert column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Get the intersection of required columns and available columns
            available_cols = [col for col in file_types[file_type] if col in df.columns]
            
            if not available_cols:
                logging.warning(f"No required columns found in {file}")
                continue
            
            # Select only the required columns
            df = df[available_cols]
            
            # Convert date columns
            df = convert_date_columns(df, date_columns[file_type])
            
            # Convert pid to string type
            if 'pid' in df.columns:
                df['pid'] = df['pid'].astype(str)
                
            dfs.append(df)
            logging.info(f"Successfully read {file}")
        except Exception as e:
            logging.error(f"Error reading {file}: {str(e)}")
    
    if not dfs:
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

def process_all_years():
    """
    Process all years for all file types and return separate DataFrames for each type
    
    Returns:
    --------
    tuple
        Tuple containing DataFrames for each file type:
        - dx_df: DataFrame containing diagnosis data with columns:
            * pid: patient identifier (string type)
            * dx_date: diagnosis date
            * code: diagnosis code
        
        - core_df: DataFrame containing core visit data with columns:
            * pid: patient identifier (string type)
            * visit_date: visit date
            * patient_class: patient class information
        
        - med_df: DataFrame containing medication data with columns:
            * pid: patient identifier (string type)
            * start_date: medication start date
            * medication_name: name of the medication
        
        - lab_df: DataFrame containing laboratory test data with columns:
            * pid: patient identifier (string type)
            * result_loinc_code: LOINC code for the test
            * result_num: numerical result value
            * result_date: date of the test
        
        - vital_df: DataFrame containing vital signs data with columns:
            * pid: patient identifier (string type)
            * vital_name: name of the vital sign
            * result_vital: vital sign measurement value
            * recorded_time: time of measurement
    """
    # Initialize DataFrames for each type
    dx_df = None
    core_df = None
    med_df = None
    lab_df = None
    vital_df = None
    
    # Process each file type
    for file_type in file_types.keys():
        logging.info(f"Processing {file_type} files...")
        yearly_dfs = []
        
        for year in years:
            df = read_and_combine_files(file_type, year)
            if df is not None:
                yearly_dfs.append(df)
        
        if yearly_dfs:
            # Combine all years' data
            combined_df = pd.concat(yearly_dfs, ignore_index=True)
            
            # Assign to the appropriate DataFrame variable
            if file_type == 'Dx':
                dx_df = combined_df
            elif file_type == 'Core':
                core_df = combined_df
            elif file_type == 'Med':
                med_df = combined_df
            elif file_type == 'Lab':
                lab_df = combined_df
            elif file_type == 'Vital':
                vital_df = combined_df
    
    # Return all DataFrames to maintain compatibility
    return dx_df, core_df, med_df, lab_df, vital_df

@time_execution
def process_antidiabetic_cohort(med_df):
    """
    Process medication data to identify and categorize patients taking antidiabetic drugs
    
    Parameters:
    -----------
    med_df : pandas.DataFrame
        Medication dataframe containing columns:
        - pid: patient identifier
        - start_date: medication start date
        - medication_name: name of the medication
        
    Returns:
    --------
    pandas.DataFrame
        Final cohort dataframe with one row per patient, containing:
        - pid: patient identifier
        - entry_date: date of first antidiabetic medication
        - medication_name: name of the medication
        - drug_name: keyword used to identify the drug
        - drug_class: class of drug (glp1, sglt2, or dpp4)
    """
    # Check if medication data is available
    if med_df is None:
        logging.warning("No medication data available.")
        return None
    
    # Rename date column for consistency
    if 'start_date' in med_df.columns:
        med_df = med_df.rename(columns={'start_date': 'entry_date'})
    
    # Filter for medications after 2019
    med_df['entry_date'] = pd.to_datetime(med_df['entry_date'].astype(str)).dt.date  # WEIPENG ADD
    med_df = med_df[med_df['entry_date'] >= pd.to_datetime('2019-01-01').date()]
    logging.info(f"Total medication records after 2019: {med_df.shape[0]:,}")
    
    # Initialize an empty list to store all matching records
    all_drug_records = []
    
    # Search for each drug keyword in medication_name column
    for drug_class, keywords in ANTIDIABETIC_DRUGS.items():
        for keyword in keywords:
            # Find records containing the keyword (case-insensitive)
            matching_records = med_df[med_df['medication_name'].str.contains(keyword, case=False, na=False)].copy()
            
            if not matching_records.empty:
                # Add drug_name and drug_class columns
                matching_records['drug_name'] = keyword
                matching_records['drug_class'] = drug_class
                all_drug_records.append(matching_records)
                logging.info(f"Found {matching_records.shape[0]:,} records for {keyword} ({drug_class})")
    
    # Combine all matching records
    if not all_drug_records:
        logging.warning("No matching drug records found.")
        return None
        
    drug_df = pd.concat(all_drug_records, ignore_index=True)
    logging.info(f"Total drug records found: {drug_df.shape[0]:,} for {drug_df['pid'].nunique():,} unique patients")
    
    # Keep only the earliest record for each patient
    drug_df = drug_df.sort_values('entry_date')
    
    # Check for patients with multiple drug classes on the same earliest date
    earliest_dates = drug_df.groupby('pid')['entry_date'].min().reset_index()
    earliest_records = pd.merge(drug_df, earliest_dates, on=['pid', 'entry_date'])
    
    # Count drug classes per patient on their earliest date
    patient_class_counts = earliest_records.groupby('pid')['drug_class'].nunique()
    
    # Identify patients with multiple drug classes on the same date
    multi_class_patients = patient_class_counts[patient_class_counts > 1].index.tolist()
    logging.info(f"Patients with multiple drug classes on the same first date: {len(multi_class_patients):,}")
    
    # Remove patients with multiple drug classes on the same date
    clean_drug_df = earliest_records[~earliest_records['pid'].isin(multi_class_patients)]
    
    # Keep only the first record for each patient (in case of multiple records with same drug class on same date)
    final_cohort = clean_drug_df.drop_duplicates(subset=['pid'], keep='first')
    
    return final_cohort

@time_execution
def process_demographics(cohort_df, demo_file=demo_file_dir):
    """
    Process demographic data and join it with the cohort
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    demo_file : str
        Path to the demographic data file
        
    Returns:
    --------
    pandas.DataFrame
        Joined dataframe with demographic information and age at entry
    """
    logging.info(f"Processing demographic data from {demo_file}")
    
    # Check if cohort data is available
    if cohort_df is None or cohort_df.empty:
        logging.warning("No cohort data available.")
        return None
    
    # Record initial cohort size
    initial_size = cohort_df.shape[0]
    logging.info(f"Initial cohort size: {initial_size:,} patients")
    
    try:
        # Read demographic data
        demo_df = pd.read_parquet(demo_file)
        
        # Convert pid to string in demographic data
        if 'pid' in demo_df.columns:
            demo_df['pid'] = demo_df['pid'].astype(str)
        
        # Convert date columns to datetime and keep only date part
        date_columns = ['birth_date', 'death_date']
        for col in date_columns:
            if col in demo_df.columns:
                demo_df[col] = pd.to_datetime(demo_df[col].astype(str), errors='coerce').dt.date
        
        # Deduplicate demographics to one row per pid to avoid 1-to-many expansion
        pre_rows, pre_unique = demo_df.shape[0], demo_df['pid'].nunique()
        if pre_rows != pre_unique:
            logging.info(f"Demographics has duplicates: rows={pre_rows:,}, unique_pid={pre_unique:,}. Deduplicating by pid.")
            demo_df = demo_df.sort_values(['pid']).drop_duplicates(subset=['pid'], keep='first')
            logging.info(f"After deduplication: {demo_df.shape[0]:,} rows")

        # Join cohort with demographic data
        merged_df = pd.merge(cohort_df, demo_df, on='pid', how='left')
        logging.info(f"After joining with demographics: {merged_df.shape[0]:,} patients")
        logging.info(f"Post-join: rows={merged_df.shape[0]:,}, unique_pid={merged_df['pid'].nunique():,}")
        
        # Check if any patients are missing after join
        if merged_df.shape[0] != cohort_df.shape[0]:
            logging.warning(f"{cohort_df.shape[0] - merged_df.shape[0]:,} patients from cohort not found in demographics data")
        
        # Process birth_date
        if 'birth_date' in merged_df.columns:
            # Remove records with missing birth date
            has_birth_date = merged_df.shape[0]
            merged_df = merged_df.dropna(subset=['birth_date'])
            logging.info(f"Removed {has_birth_date - merged_df.shape[0]:,} patients with missing birth date")
            
            # Calculate age at entry
            merged_df['entry_date'] = pd.to_datetime(merged_df['entry_date'].astype(str))
            merged_df['age_at_entry'] = ((merged_df['entry_date'] - pd.to_datetime(merged_df['birth_date'].astype(str))).dt.days / 365.25).round(1)
            
            # Remove patients younger than 18 at entry
            adult_count = merged_df.shape[0]
            merged_df = merged_df[merged_df['age_at_entry'] >= 18]
            logging.info(f"Removed {adult_count - merged_df.shape[0]:,} patients younger than 18 years")
        else:
            logging.warning("birth_date column not found in demographic data")
        
        # Final cohort size
        logging.info(f"Final cohort after demographic processing: {merged_df.shape[0]:,} patients")
        logging.info(f"Total reduction: {initial_size - merged_df.shape[0]:,} patients ({(initial_size - merged_df.shape[0])/initial_size*100:.2f}%)")
        
        return merged_df
        
    except Exception as e:
        logging.error(f"Error processing demographic data: {str(e)}")
        return None

@time_execution
def process_baseline_medications(cohort_df, med_df):
    """
    Process baseline medication data to identify antidiabetic drug usage during baseline period
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid, entry_date and drug_class columns
    med_df : pandas.DataFrame
        Medication dataframe with pid, start_date and medication_name columns
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with baseline medication flags and filtered to exclude patients
        who used same class of antidiabetic drugs during baseline period
    """
    logging.info("Processing baseline medications...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or med_df is None or med_df.empty:
        logging.warning("Missing input data for baseline medication processing")
        return cohort_df
    
    # Record initial cohort size
    initial_size = cohort_df.shape[0]
    logging.info(f"Initial cohort size before baseline medication filtering: {initial_size:,} patients")
    
    # Ensure we have entry_date in correct format
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Make sure med_df has the right date column name
    if 'start_date' in med_df.columns and 'entry_date' not in med_df.columns:
        med_df = med_df.rename(columns={'start_date': 'med_date'})
    elif 'entry_date' in med_df.columns:
        med_df = med_df.rename(columns={'entry_date': 'med_date'})
    
    # Convert med_date to datetime
    med_df['med_date'] = pd.to_datetime(med_df['med_date'].astype(str))
    
    # Initialize baseline flags in cohort dataframe
    cohort_df['baseline_glp1'] = 0
    cohort_df['baseline_sglt2'] = 0
    cohort_df['baseline_dpp4'] = 0
    
    # Create a copy of med_df for optimization
    working_med_df = med_df[['pid', 'med_date', 'medication_name']].copy()
    
    # Create a list to store filtered patients
    patients_to_keep = []
    
    # Process each drug class
    for drug_class, keywords in ANTIDIABETIC_DRUGS.items():
        # Create flag for this class in medication data
        working_med_df[f'is_{drug_class}'] = False
        
        # Mark medications belonging to this class
        for keyword in keywords:
            working_med_df[f'is_{drug_class}'] |= working_med_df['medication_name'].str.contains(keyword, case=False, na=False)
    
    # Perform a join to get cohort data alongside med data
    # This creates a cartesian product for each patient's meds
    joined_df = pd.merge(
        cohort_df[['pid', 'entry_date', 'drug_class']], 
        working_med_df,
        on='pid',
        how='left'
    )
    
    # Filter for baseline period
    joined_df['is_baseline'] = (
        (joined_df['med_date'] >= (joined_df['entry_date'] - pd.Timedelta(days=365))) & 
        (joined_df['med_date'] < joined_df['entry_date'])
    )
    baseline_meds = joined_df[joined_df['is_baseline']]
    
    # Group by patient and aggregate drug flags
    baseline_flags = baseline_meds.groupby('pid').agg({
        'is_glp1': 'any',
        'is_sglt2': 'any',
        'is_dpp4': 'any'
    }).reset_index()
    
    # Rename columns for clarity
    baseline_flags.rename(columns={
        'is_glp1': 'baseline_glp1',
        'is_sglt2': 'baseline_sglt2',
        'is_dpp4': 'baseline_dpp4'
    }, inplace=True)
    
    # Convert boolean to integer (0/1)
    for col in ['baseline_glp1', 'baseline_sglt2', 'baseline_dpp4']:
        baseline_flags[col] = baseline_flags[col].astype(int)
    
    # Merge baseline flags back to cohort
    result_df = pd.merge(cohort_df, baseline_flags, on='pid', how='left', suffixes=('', '_new'))
    
    # Update baseline columns, handling NaNs from the left join
    for col in ['baseline_glp1', 'baseline_sglt2', 'baseline_dpp4']:
        new_col = f"{col}_new"
        if new_col in result_df.columns:
            result_df[col] = result_df[new_col].fillna(0).astype(int)
            result_df.drop(columns=[new_col], inplace=True)
    
    # Report baseline usage
    for drug_class in ['glp1', 'sglt2', 'dpp4']:
        baseline_count = result_df[f'baseline_{drug_class}'].sum()
        logging.info(f"Patients with baseline {drug_class} usage: {baseline_count:,}")
    
    # Identify patients who used any antidiabetic drug during baseline
    result_df['used_any_antidiabetic'] = (
        (result_df['baseline_glp1'] == 1) | 
        (result_df['baseline_sglt2'] == 1) | 
        (result_df['baseline_dpp4'] == 1)
    )
    
    # Remove patients who used any antidiabetic drugs during baseline
    filtered_cohort = result_df[~result_df['used_any_antidiabetic']].drop(columns=['used_any_antidiabetic'])
    
    # Report filtering results
    removed_count = result_df['used_any_antidiabetic'].sum()
    logging.info(f"Removed {removed_count:,} patients who used any antidiabetic drug during baseline")
    logging.info(f"Final cohort after baseline filtering: {filtered_cohort.shape[0]:,} patients")
    logging.info(f"Total reduction: {initial_size - filtered_cohort.shape[0]:,} patients ({removed_count/initial_size*100:.2f}%)")
    
    return filtered_cohort

@time_execution
def process_baseline_conditions(cohort_df, dx_df, icd_file=icd_codes_file):
    """
    Process baseline diagnosis data to identify and filter patients with ESRD, T1D, and T2D
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    dx_df : pandas.DataFrame
        Diagnosis dataframe with pid, dx_date, and code columns
    icd_file : str
        Path to the ICD codes file defining conditions
        
    Returns:
    --------
    pandas.DataFrame
        Filtered cohort dataframe
    """
    logging.info("Processing baseline conditions...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or dx_df is None or dx_df.empty:
        logging.warning("Missing input data for baseline condition processing")
        return cohort_df
    
    # Record initial cohort size
    initial_size = cohort_df.shape[0]
    logging.info(f"Initial cohort size before condition filtering: {initial_size:,} patients")
    
    # Ensure we have entry_date in correct format
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Convert dx_date to datetime
    dx_df['dx_date'] = pd.to_datetime(dx_df['dx_date'].astype(str))
    
    # Read ICD codes file to get condition definitions
    try:
        # Read ICD codes file
        icd_codes = pd.read_csv(icd_file)
        # Extract covariate conditions
        cov_codes = icd_codes[icd_codes['cat'] == 'Covariate']
        
        # Get ICD-10 codes for each condition
        # Using column indexing with brackets syntax to handle hyphen in column name
        esrd_codes = cov_codes[cov_codes['cluster'] == 'ESRD']['icd-10'].tolist()
        t1d_codes = cov_codes[cov_codes['cluster'] == 'T1D']['icd-10'].tolist()
        t2d_codes = cov_codes[cov_codes['cluster'] == 'T2D']['icd-10'].tolist()
        
        logging.info(f"Found {len(esrd_codes)} ESRD codes, {len(t1d_codes)} T1D codes, and {len(t2d_codes)} T2D codes")
    except Exception as e:
        logging.error(f"Error reading ICD codes file: {str(e)}")
        # Default codes if file cannot be read
        esrd_codes = ['N18.5', 'N18.6', 'R88.0', 'T82.4', 'T85.611', 'T85.621', 'T85.631', 'T85.71', 'T86.1', 'Y84.1']
        t1d_codes = ['E10']
        t2d_codes = ['E11']
        logging.info("Using default codes instead")
    
    # Initialize baseline condition flags in cohort dataframe
    cohort_df['baseline_ESRD'] = 0
    cohort_df['baseline_T1D'] = 0
    cohort_df['baseline_T2D'] = 0
    
    # Perform a join to get cohort data alongside dx data
    joined_df = pd.merge(
        cohort_df[['pid', 'entry_date']], 
        dx_df[['pid', 'dx_date', 'code']],
        on='pid',
        how='left'
    )
    
    # Filter for baseline period
    joined_df['is_baseline'] = (
        (joined_df['dx_date'] >= (joined_df['entry_date'] - pd.Timedelta(days=365))) & 
        (joined_df['dx_date'] <= joined_df['entry_date']))
    baseline_dx = joined_df[joined_df['is_baseline']].copy()
    
    # Function to check if code starts with any code in the list
    def code_matches(code, code_list):
        if pd.isna(code):
            return False
        code_str = str(code).upper()
        for icd_code in code_list:
            if code_str.startswith(icd_code):
                return True
        return False
    
    # Apply vectorized operations to identify conditions
    # Create condition flags
    baseline_dx.loc[:, 'is_ESRD'] = baseline_dx['code'].apply(lambda x: code_matches(x, esrd_codes))
    baseline_dx.loc[:, 'is_T1D'] = baseline_dx['code'].apply(lambda x: code_matches(x, t1d_codes))
    baseline_dx.loc[:, 'is_T2D'] = baseline_dx['code'].apply(lambda x: code_matches(x, t2d_codes))
    
    # Group by patient and aggregate condition flags
    condition_flags = baseline_dx.groupby('pid').agg({
        'is_ESRD': 'any',
        'is_T1D': 'any',
        'is_T2D': 'any'
    }).reset_index()
    
    # Rename columns for clarity
    condition_flags.rename(columns={
        'is_ESRD': 'baseline_ESRD',
        'is_T1D': 'baseline_T1D',
        'is_T2D': 'baseline_T2D'
    }, inplace=True)
    
    # Convert boolean to integer (0/1)
    for col in ['baseline_ESRD', 'baseline_T1D', 'baseline_T2D']:
        condition_flags[col] = condition_flags[col].astype(int)
    
    # Merge condition flags back to cohort
    result_df = pd.merge(cohort_df, condition_flags, on='pid', how='left', suffixes=('', '_new'))
    
    # Update condition columns, handling NaNs from the left join
    for col in ['baseline_ESRD', 'baseline_T1D', 'baseline_T2D']:
        new_col = f"{col}_new"
        if new_col in result_df.columns:
            result_df[col] = result_df[new_col].fillna(0).astype(int)
            result_df.drop(columns=[new_col], inplace=True)
    
    # Report baseline condition statistics
    for condition in ['ESRD', 'T1D', 'T2D']:
        count = result_df[f'baseline_{condition}'].sum()
        logging.info(f"Patients with baseline {condition}: {count:,} ({count/result_df.shape[0]*100:.2f}%)")
    
    # Apply filtering rules:
    # 1. Remove patients with ESRD
    # 2. Remove patients with T1D
    # 3. Keep only patients with T2D
    filtered_cohort = result_df[
        (result_df['baseline_ESRD'] == 0) & 
        (result_df['baseline_T1D'] == 0) & 
        (result_df['baseline_T2D'] == 1)
    ]
    
    # Report filtering results
    removed_count = initial_size - filtered_cohort.shape[0]
    logging.info(f"Removed {removed_count:,} patients based on condition criteria")
    logging.info(f"Final cohort after condition filtering: {filtered_cohort.shape[0]:,} patients")
    logging.info(f"Total reduction: {removed_count:,} patients ({removed_count/initial_size*100:.2f}%)")
    
    return filtered_cohort

@time_execution
def process_baseline_visits(cohort_df, core_df):
    """
    Check if each patient has any visit records during the baseline period
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    core_df : pandas.DataFrame
        Core visit dataframe with pid and visit_date columns
        
    Returns:
    --------
    pandas.DataFrame
        Filtered cohort dataframe containing only patients with at least one visit during baseline
    """
    logging.info("Checking for baseline visits...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty:
        logging.warning("Missing cohort data for baseline visit checking")
        return cohort_df
    
    # Record initial cohort size
    initial_size = cohort_df.shape[0]
    logging.info(f"Checking baseline visits for {initial_size:,} patients")
    
    if core_df is None or core_df.empty:
        logging.warning("No core visit data available, no patients can be kept")
        return pd.DataFrame(columns=cohort_df.columns)  # Return empty dataframe with same structure
    
    # Ensure we have entry_date in correct format
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Convert visit_date to datetime
    if 'visit_date' in core_df.columns:
        if not pd.api.types.is_datetime64_dtype(core_df['visit_date']):
            core_df['visit_date'] = pd.to_datetime(core_df['visit_date'].astype(str))
    else:
        logging.error("'visit_date' column not found in core_df")
        return pd.DataFrame(columns=cohort_df.columns)  # Return empty dataframe with same structure
    
    # Initialize has_baseline_visit column
    cohort_df['has_baseline_visit'] = 0
    
    # Perform a join to get cohort data alongside core visit data
    joined_df = pd.merge(
        cohort_df[['pid', 'entry_date']], 
        core_df[['pid', 'visit_date']],
        on='pid',
        how='left'
    )
    
    # Filter for baseline period
    baseline_visits = joined_df[
        (joined_df['visit_date'] >= (joined_df['entry_date'] - pd.Timedelta(days=365))) & 
        (joined_df['visit_date'] < joined_df['entry_date'])
    ]
    
    # Group by patient to get patients with baseline visits
    patients_with_visits = []
    if not baseline_visits.empty:
        patients_with_visits = baseline_visits['pid'].unique()
        
        # Set has_baseline_visit flag to 1 for these patients
        cohort_df.loc[cohort_df['pid'].isin(patients_with_visits), 'has_baseline_visit'] = 1
        
        # Report stats
        visit_count = cohort_df['has_baseline_visit'].sum()
        logging.info(f"Found {visit_count:,} patients with visits during baseline period ({visit_count/len(cohort_df)*100:.2f}%)")
    else:
        logging.info("No patients found with baseline visits")
    
    # Filter to keep only patients with baseline visits
    filtered_cohort = cohort_df[cohort_df['has_baseline_visit'] == 1]
    
    # Report filtering results
    removed_count = initial_size - filtered_cohort.shape[0]
    logging.info(f"Removed {removed_count:,} patients without baseline visits")
    logging.info(f"Final cohort after baseline visit filtering: {filtered_cohort.shape[0]:,} patients")
    logging.info(f"Total reduction: {removed_count:,} patients ({removed_count/initial_size*100:.2f}%)")
    
    return filtered_cohort

@time_execution
def process_mental_outcomes(cohort_df, dx_df, icd_file=icd_codes_file):
    """
    Process mental health outcomes based on diagnosis records
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    dx_df : pandas.DataFrame
        Diagnosis dataframe with pid, dx_date, and code columns
    icd_file : str
        Path to the ICD codes file defining mental health conditions
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with mental health outcomes added:
        - visits_{cluster}: date of first diagnosis during follow-up
        - visits_binary_{cluster}: binary flag (0/1) for diagnosis during follow-up
        - pre_visits_{cluster}: binary flag (0/1) for diagnosis during baseline
    """
    logging.info("Processing mental health outcomes...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or dx_df is None or dx_df.empty:
        logging.warning("Missing input data for mental health outcome processing")
        return cohort_df
    
    # Create a complete copy of the cohort dataframe
    cohort_df = cohort_df.copy()
    
    # Ensure pid is string type in both dataframes to avoid merge errors
    cohort_df['pid'] = cohort_df['pid'].astype(str)
    dx_df['pid'] = dx_df['pid'].astype(str)
    
    # Read ICD codes file to get Mental outcome definitions
    try:
        # Read ICD codes file
        icd_codes = pd.read_csv(icd_file)
        
        # Extract mental health outcome conditions (cat = 'Outcome' and cat_sub = 'Mental')
        mental_codes = icd_codes[(icd_codes['cat'] == 'Outcome') & (icd_codes['cat_sub'] == 'Mental')]
        
        # Get unique clusters (different mental health conditions)
        mental_clusters = mental_codes['cluster'].unique()
        
        logging.info(f"Found {len(mental_clusters)} mental health condition clusters")
        
        # Create a dictionary to store ICD-10 codes for each cluster
        mental_icd_dict = {}
        for cluster in mental_clusters:
            # Get ICD-10 codes for this cluster
            codes = mental_codes[mental_codes['cluster'] == cluster]['icd-10'].tolist()
            mental_icd_dict[cluster] = codes
            logging.info(f"  - {cluster}: {len(codes)} ICD-10 codes")
        
    except Exception as e:
        logging.error(f"Error reading ICD codes file: {str(e)}")
        # Return original cohort if we can't process outcomes
        return cohort_df
    
    # Ensure we have entry_date in correct format
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Convert dx_date to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(dx_df['dx_date']):
        dx_df['dx_date'] = pd.to_datetime(dx_df['dx_date'].astype(str))
    
    # Merge dx_df with cohort_df for processing
    cohort_dx_df = pd.merge(dx_df, cohort_df[['pid', 'entry_date']], on='pid', how='inner')
    
    # Process each mental health condition cluster
    for cluster, icd_codes in mental_icd_dict.items():
        logging.info(f"Processing {cluster} outcomes...")
        
        #############################################################
        # FOLLOW-UP PERIOD: Find diagnoses after the entry date
        #############################################################
        # Filter for records after entry date matching the current cluster's ICD codes
        cluster_records = cohort_dx_df[
            (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date']) &
            (cohort_dx_df['code'].str.startswith(tuple(icd_codes), na=False))
        ]
        
        # Find earliest diagnosis date for each patient
        if not cluster_records.empty:
            idx = cluster_records.groupby('pid')['dx_date'].idxmin()
            earliest_dates = cluster_records.loc[idx, ['pid', 'dx_date']].set_index('pid')['dx_date']
        else:
            earliest_dates = pd.Series(dtype='datetime64[ns]')
        
        # Add outcome columns to cohort dataframe - both date and binary indicator
        cohort_df[f'visits_{cluster}'] = cohort_df['pid'].map(earliest_dates).dt.date
        cohort_df[f'visits_binary_{cluster}'] = cohort_df[f'visits_{cluster}'].notna().astype(int)
        
        #############################################################
        # BASELINE PERIOD: Find diagnoses in the year before entry date
        #############################################################
        # Filter for records within 1 year before entry date matching the current cluster's ICD codes
        pre_records = cohort_dx_df[
            (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date'] - pd.Timedelta(days=365)) &
            (cohort_dx_df['dx_date'] <= cohort_dx_df['entry_date']) &
            (cohort_dx_df['code'].str.startswith(tuple(icd_codes), na=False))
        ]
        
        # Set baseline flag to 1 for patients with any matching records during baseline
        cohort_df[f'pre_visits_{cluster}'] = cohort_df['pid'].isin(pre_records['pid']).astype(int)
        
        # Report statistics for this cluster
        outcome_count = cohort_df[f'visits_binary_{cluster}'].sum()
        logging.info(f"  - Found {outcome_count:,} patients with {cluster} outcome in follow-up period ({outcome_count/len(cohort_df)*100:.2f}%)")
        
        baseline_count = cohort_df[f'pre_visits_{cluster}'].sum()
        logging.info(f"  - Found {baseline_count:,} patients with {cluster} in baseline period ({baseline_count/len(cohort_df)*100:.2f}%)")
    
    return cohort_df

def process_gi_outcomes(cohort_df, dx_df, icd_file=icd_codes_file):
    """
    Process gastrointestinal (GI) outcomes based on diagnosis records
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    dx_df : pandas.DataFrame
        Diagnosis dataframe with pid, dx_date, and code columns
    icd_file : str
        Path to the ICD codes file defining GI conditions
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with GI outcomes added:
        - visits_{cluster}: date of first diagnosis during follow-up
        - visits_binary_{cluster}: binary flag (0/1) for diagnosis during follow-up
        - pre_visits_{cluster}: binary flag (0/1) for diagnosis during baseline
    """
    logging.info("Processing gastrointestinal (GI) outcomes...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or dx_df is None or dx_df.empty:
        logging.info("Missing input data for GI outcome processing")
        return cohort_df
    
    # Create a complete copy of the cohort dataframe
    cohort_df = cohort_df.copy()
    
    # Ensure pid is string type in both dataframes to avoid merge errors
    cohort_df['pid'] = cohort_df['pid'].astype(str)
    dx_df['pid'] = dx_df['pid'].astype(str)
    
    # Read ICD codes file to get GI outcome definitions
    try:
        # Read ICD codes file
        icd_codes = pd.read_csv(icd_file)
        
        # Extract GI outcome conditions (cat = 'Outcome' and cat_sub = 'GI')
        gi_codes = icd_codes[(icd_codes['cat'] == 'Outcome') & (icd_codes['cat_sub'] == 'GI')]
        
        # Get unique clusters (different GI conditions)
        gi_clusters = gi_codes['cluster'].unique()
        
        logging.info(f"Found {len(gi_clusters)} GI condition clusters")
        
        # Create a dictionary to store ICD-10 codes for each cluster
        gi_icd_dict = {}
        for cluster in gi_clusters:
            # Get ICD-10 codes for this cluster
            codes = gi_codes[gi_codes['cluster'] == cluster]['icd-10'].tolist()
            gi_icd_dict[cluster] = codes
            logging.info(f"  - {cluster}: {len(codes)} ICD-10 codes")
        
    except Exception as e:
        logging.info(f"Error reading ICD codes file: {str(e)}")
        # Return original cohort if we can't process outcomes
        return cohort_df
    
    # Ensure we have entry_date in correct format
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Convert dx_date to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(dx_df['dx_date']):
        dx_df['dx_date'] = pd.to_datetime(dx_df['dx_date'].astype(str))
    
    # Merge dx_df with cohort_df for processing
    cohort_dx_df = pd.merge(dx_df, cohort_df[['pid', 'entry_date']], on='pid', how='inner')
    
    # Process each GI condition cluster
    for cluster, icd_codes in gi_icd_dict.items():
        logging.info(f"Processing {cluster} outcomes...")
        
        #############################################################
        # FOLLOW-UP PERIOD: Find diagnoses after the entry date
        #############################################################
        # Filter for records after entry date matching the current cluster's ICD codes
        cluster_records = cohort_dx_df[
            (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date']) &
            (cohort_dx_df['code'].str.startswith(tuple(icd_codes), na=False))
        ]
        
        # Find earliest diagnosis date for each patient
        if not cluster_records.empty:
            idx = cluster_records.groupby('pid')['dx_date'].idxmin()
            earliest_dates = cluster_records.loc[idx, ['pid', 'dx_date']].set_index('pid')['dx_date']
        else:
            earliest_dates = pd.Series(dtype='datetime64[ns]')
        
        # Add outcome columns to cohort dataframe - both date and binary indicator
        cohort_df[f'visits_{cluster}'] = cohort_df['pid'].map(earliest_dates).dt.date
        cohort_df[f'visits_binary_{cluster}'] = cohort_df[f'visits_{cluster}'].notna().astype(int)
        
        #############################################################
        # BASELINE PERIOD: Find diagnoses in the year before entry date
        #############################################################
        # Filter for records within 1 year before entry date matching the current cluster's ICD codes
        pre_records = cohort_dx_df[
            (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date'] - pd.Timedelta(days=365)) &
            (cohort_dx_df['dx_date'] <= cohort_dx_df['entry_date']) &
            (cohort_dx_df['code'].str.startswith(tuple(icd_codes), na=False))
        ]
        
        # Set baseline flag to 1 for patients with any matching records during baseline
        cohort_df[f'pre_visits_{cluster}'] = cohort_df['pid'].isin(pre_records['pid']).astype(int)
        
        # Report statistics for this cluster
        outcome_count = cohort_df[f'visits_binary_{cluster}'].sum()
        logging.info(f"  - Found {outcome_count:,} patients with {cluster} outcome in follow-up period ({outcome_count/len(cohort_df)*100:.2f}%)")
        
        baseline_count = cohort_df[f'pre_visits_{cluster}'].sum()
        logging.info(f"  - Found {baseline_count:,} patients with {cluster} in baseline period ({baseline_count/len(cohort_df)*100:.2f}%)")
    
    return cohort_df

def process_cardiac_outcomes(cohort_df, dx_df, core_df, icd_file=icd_codes_file):
    """
    Process cardiac outcomes based on diagnosis records and special criteria
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid, entry_date and death_date columns
    dx_df : pandas.DataFrame
        Diagnosis dataframe with pid, dx_date, and code columns
    core_df : pandas.DataFrame
        Core visit dataframe with pid, visit_date, and patient_class columns
    icd_file : str
        Path to the ICD codes file defining cardiac conditions
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with cardiac outcomes added:
        - visits_{outcome}: date of first diagnosis during follow-up
        - visits_binary_{outcome}: binary flag (0/1) for diagnosis during follow-up
        - pre_visits_{outcome}: binary flag (0/1) for diagnosis during baseline
        
        Where outcome is one of:
        - non_fatal_mi: Non-fatal myocardial infarction identified by ICD-10 codes during follow-up period
        - non_fatal_stroke: Non-fatal stroke identified by ICD-10 codes during follow-up period
        - cv_death: Cardiovascular death identified by ICD-10 codes, requires patient to have a death_date record within 30 days after the cardiovascular diagnosis
        - unstable_angina: Unstable angina identified by ICD-10 codes, requires an inpatient hospitalization record within 1 day before or after the diagnosis date
    """
    logging.info("Processing cardiac outcomes...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or dx_df is None or dx_df.empty or core_df is None or core_df.empty:
        logging.info("Missing input data for cardiac outcome processing")
        return cohort_df
    
    # Create a complete copy of the cohort dataframe
    cohort_df = cohort_df.copy()
    
    # Ensure pid is string type in all dataframes to avoid merge errors
    cohort_df['pid'] = cohort_df['pid'].astype(str)
    dx_df['pid'] = dx_df['pid'].astype(str)
    core_df['pid'] = core_df['pid'].astype(str)
    
    # Ensure we have entry_date and death_date in correct format
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    if 'death_date' in cohort_df.columns:
        cohort_df['death_date'] = pd.to_datetime(cohort_df['death_date'].astype(str))
    else:
        # If death_date is missing, add it as NA
        cohort_df['death_date'] = pd.NaT
    
    # Convert dx_date to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(dx_df['dx_date']):
        dx_df['dx_date'] = pd.to_datetime(dx_df['dx_date'].astype(str))
    
    # Convert visit_date to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(core_df['visit_date']):
        core_df['visit_date'] = pd.to_datetime(core_df['visit_date'].astype(str))
    
    # Read ICD codes file to get Cardiac outcome definitions
    try:
        # Read ICD codes file
        icd_codes = pd.read_csv(icd_file)
        
        # Extract Cardiac outcome conditions (cat = 'Outcome' and cat_sub = 'Cardiac')
        cardiac_codes = icd_codes[(icd_codes['cat'] == 'Outcome') & (icd_codes['cat_sub'] == 'Cardiac')]
        
        # Get specific cardiac condition clusters
        non_fatal_mi_codes = cardiac_codes[cardiac_codes['cluster'] == 'Non-fatal_mayocardial_infarction']['icd-10'].tolist()
        non_fatal_stroke_codes = cardiac_codes[cardiac_codes['cluster'] == 'Non-fatal_stroke']['icd-10'].tolist()
        cv_death_codes = cardiac_codes[cardiac_codes['cluster'] == 'CV_Death']['icd-10'].tolist()
        cv_event_codes = cardiac_codes[cardiac_codes['cluster'] == 'CV_Event_code']['icd-10'].tolist()
        unstable_angina_codes = cardiac_codes[cardiac_codes['cluster'] == 'unstable_angina']['icd-10'].tolist()
        
        logging.info(f"Found cardiac condition codes:")
        logging.info(f"  - Non-fatal MI: {len(non_fatal_mi_codes)} codes")
        logging.info(f"  - Non-fatal Stroke: {len(non_fatal_stroke_codes)} codes")
        logging.info(f"  - CV Death: {len(cv_death_codes)} codes")
        logging.info(f"  - CV Event: {len(cv_event_codes)} codes")
        logging.info(f"  - Unstable Angina: {len(unstable_angina_codes)} codes")
        
    except Exception as e:
        logging.info(f"Error reading ICD codes file: {str(e)}")
        # Return original cohort if we can't process outcomes
        return cohort_df
    
    # Merge dx_df with cohort_df for processing
    cohort_dx_df = pd.merge(dx_df, cohort_df[['pid', 'entry_date', 'death_date']], on='pid', how='inner')
    
    # Initialize outcome columns
    cardiac_outcomes = ['non_fatal_mi', 'non_fatal_stroke', 'cv_death', 'unstable_angina']
    
    #############################################################
    # NON-FATAL MI: Process Myocardial Infarction outcomes
    #############################################################
    logging.info("Processing Non-fatal MI outcomes...")
    
    # FOLLOW-UP PERIOD: Find MI diagnoses after entry date
    mi_records = cohort_dx_df[
        (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date']) &
        (cohort_dx_df['code'].str.startswith(tuple(non_fatal_mi_codes), na=False))
    ]
    
    if not mi_records.empty:
        idx = mi_records.groupby('pid')['dx_date'].idxmin()
        earliest_mi_dates = mi_records.loc[idx, ['pid', 'dx_date']].set_index('pid')['dx_date']
        
        # Add outcome columns - both date and binary indicator
        cohort_df['visits_non_fatal_mi'] = cohort_df['pid'].map(earliest_mi_dates).dt.date
        cohort_df['visits_binary_non_fatal_mi'] = cohort_df['visits_non_fatal_mi'].notna().astype(int)
        
        mi_count = cohort_df['visits_binary_non_fatal_mi'].sum()
        logging.info(f"  - Found {mi_count} patients with Non-fatal MI in follow-up period ({mi_count/len(cohort_df)*100:.2f}%)")
    else:
        cohort_df['visits_non_fatal_mi'] = pd.NaT
        cohort_df['visits_binary_non_fatal_mi'] = 0
    
    # BASELINE PERIOD: Find MI diagnoses in the year before entry date
    pre_mi_records = cohort_dx_df[
        (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date'] - pd.Timedelta(days=365)) &
        (cohort_dx_df['dx_date'] <= cohort_dx_df['entry_date']) &
        (cohort_dx_df['code'].str.startswith(tuple(non_fatal_mi_codes), na=False))
    ]
    
    # Set baseline flag based on existence of records
    cohort_df['pre_visits_non_fatal_mi'] = cohort_df['pid'].isin(pre_mi_records['pid']).astype(int)
    
    baseline_mi_count = cohort_df['pre_visits_non_fatal_mi'].sum()
    logging.info(f"  - Found {baseline_mi_count} patients with Non-fatal MI in baseline period ({baseline_mi_count/len(cohort_df)*100:.2f}%)")
    
    #############################################################
    # NON-FATAL STROKE: Process Stroke outcomes
    #############################################################
    logging.info("Processing Non-fatal Stroke outcomes...")
    
    # FOLLOW-UP PERIOD: Find Stroke diagnoses after entry date
    stroke_records = cohort_dx_df[
        (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date']) &
        (cohort_dx_df['code'].str.startswith(tuple(non_fatal_stroke_codes), na=False))
    ]
    
    if not stroke_records.empty:
        idx = stroke_records.groupby('pid')['dx_date'].idxmin()
        earliest_stroke_dates = stroke_records.loc[idx, ['pid', 'dx_date']].set_index('pid')['dx_date']
        
        # Add outcome columns - both date and binary indicator
        cohort_df['visits_non_fatal_stroke'] = cohort_df['pid'].map(earliest_stroke_dates).dt.date
        cohort_df['visits_binary_non_fatal_stroke'] = cohort_df['visits_non_fatal_stroke'].notna().astype(int)
        
        stroke_count = cohort_df['visits_binary_non_fatal_stroke'].sum()
        logging.info(f"  - Found {stroke_count} patients with Non-fatal Stroke in follow-up period ({stroke_count/len(cohort_df)*100:.2f}%)")
    else:
        cohort_df['visits_non_fatal_stroke'] = pd.NaT
        cohort_df['visits_binary_non_fatal_stroke'] = 0
    
    # BASELINE PERIOD: Find Stroke diagnoses in the year before entry date
    pre_stroke_records = cohort_dx_df[
        (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date'] - pd.Timedelta(days=365)) &
        (cohort_dx_df['dx_date'] <= cohort_dx_df['entry_date']) &
        (cohort_dx_df['code'].str.startswith(tuple(non_fatal_stroke_codes), na=False))
    ]
    
    # Set baseline flag based on existence of records
    cohort_df['pre_visits_non_fatal_stroke'] = cohort_df['pid'].isin(pre_stroke_records['pid']).astype(int)
    
    baseline_stroke_count = cohort_df['pre_visits_non_fatal_stroke'].sum()
    logging.info(f"  - Found {baseline_stroke_count} patients with Non-fatal Stroke in baseline period ({baseline_stroke_count/len(cohort_df)*100:.2f}%)")
    
    #############################################################
    # CV DEATH: Process Cardiovascular Death outcomes
    # Special case: Requires either CV death code or CV event followed by death
    #############################################################
    logging.info("Processing CV Death outcomes...")
    
    # FOLLOW-UP PERIOD - Part 1: Direct CV death codes
    cv_death_records = cohort_dx_df[
        (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date']) &
        (cohort_dx_df['code'].str.startswith(tuple(cv_death_codes), na=False))
    ]
    
    cv_death_direct = None
    if not cv_death_records.empty:
        idx = cv_death_records.groupby('pid')['dx_date'].idxmin()
        cv_death_direct = cv_death_records.loc[idx, ['pid', 'dx_date']].set_index('pid')
    
    # FOLLOW-UP PERIOD - Part 2: CV event followed by death within 30 days
    cv_event_records = cohort_dx_df[
        (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date']) &
        (cohort_dx_df['code'].str.startswith(tuple(cv_event_codes), na=False))
    ]
    
    cv_death_from_events = None
    if not cv_event_records.empty and 'death_date' in cohort_df.columns:
        # Add death dates to event records
        cv_events_with_death = cv_event_records.copy()
        
        # Calculate days between event and death
        cv_events_with_death['days_to_death'] = (cv_events_with_death['death_date'] - cv_events_with_death['dx_date']).dt.days
        
        # Filter for events followed by death within 30 days
        cv_death_events = cv_events_with_death[
            (cv_events_with_death['days_to_death'] > 0) & 
            (cv_events_with_death['days_to_death'] <= 30)
        ]
        
        if not cv_death_events.empty:
            idx = cv_death_events.groupby('pid')['death_date'].idxmin()
            cv_death_from_events = cv_death_events.loc[idx, ['pid', 'death_date']].rename(
                columns={'death_date': 'dx_date'}).set_index('pid')
    
    # Combine both sources of CV death for follow-up
    cv_death_final = None
    if cv_death_direct is not None and cv_death_from_events is not None:
        cv_death_combined = pd.concat([
            cv_death_direct['dx_date'],
            cv_death_from_events['dx_date']
        ]).to_frame('dx_date')
        
        cv_death_combined = cv_death_combined.reset_index().drop_duplicates(subset=['pid']).set_index('pid')
        cv_death_final = cv_death_combined.groupby('pid').min()
    elif cv_death_direct is not None:
        cv_death_final = cv_death_direct[['dx_date']]
    elif cv_death_from_events is not None:
        cv_death_final = cv_death_from_events[['dx_date']]
    
    # Add CV death outcomes to cohort - both date and binary indicator
    if cv_death_final is not None:
        cohort_df['visits_cv_death'] = cohort_df['pid'].map(cv_death_final['dx_date']).dt.date
        cohort_df['visits_binary_cv_death'] = cohort_df['visits_cv_death'].notna().astype(int)
        
        cv_death_count = cohort_df['visits_binary_cv_death'].sum()
        logging.info(f"  - Found {cv_death_count} patients with CV Death in follow-up period ({cv_death_count/len(cohort_df)*100:.2f}%)")
    else:
        cohort_df['visits_cv_death'] = pd.NaT
        cohort_df['visits_binary_cv_death'] = 0
    
    # Remove baseline CV death check since patients who died during baseline period
    # would not be in the cohort (they couldn't have prescription records after death)
    # For consistency with other outcomes, still initialize the column but set all to 0
    cohort_df['pre_visits_cv_death'] = 0
    logging.info(f"  - Baseline CV death check skipped - patients deceased during baseline would not be in cohort")
    
    #############################################################
    # UNSTABLE ANGINA: Process Unstable Angina outcomes
    # Special case: Requires both diagnosis and inpatient visit
    #############################################################
    logging.info("Processing Unstable Angina outcomes...")
    
    # FOLLOW-UP PERIOD: Find Angina diagnoses after entry date
    unstable_angina_records = cohort_dx_df[
        (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date']) &
        (cohort_dx_df['code'].str.startswith(tuple(unstable_angina_codes), na=False))
    ]
    
    if not unstable_angina_records.empty:
        # Create date range for each angina diagnosis (±1 day)
        unstable_angina_records = unstable_angina_records.copy()
        unstable_angina_records.loc[:, 'dx_date_minus1'] = unstable_angina_records['dx_date'] - pd.Timedelta(days=1)
        unstable_angina_records.loc[:, 'dx_date_plus1'] = unstable_angina_records['dx_date'] + pd.Timedelta(days=1)
        
        # Prepare core data - filter for inpatient visits only
        inpatient_visits = core_df[core_df['patient_class'].str.lower() == 'inpatient'].copy()
        
        # Merge angina diagnoses with inpatient visits to find matches within ±1 day
        unstable_angina_records = unstable_angina_records.sort_values('dx_date')
        inpatient_visits = inpatient_visits.sort_values('visit_date')
        
        # First, find visits that occurred on or after the diagnosis date - 1 day
        visits_after = pd.merge_asof(
            unstable_angina_records[['pid', 'dx_date', 'dx_date_minus1', 'dx_date_plus1']],
            inpatient_visits[['pid', 'visit_date']],
            left_on='dx_date_minus1',
            right_on='visit_date',
            by='pid',
            direction='forward'
        )
        
        # Then filter for visits that occurred before or on diagnosis date + 1 day
        valid_visits = visits_after[visits_after['visit_date'] <= visits_after['dx_date_plus1']]
        
        if not valid_visits.empty:
            # Get earliest qualifying visit for each patient
            idx = valid_visits.groupby('pid')['dx_date'].idxmin()
            unstable_angina_dates = valid_visits.loc[idx, ['pid', 'dx_date']].set_index('pid')['dx_date']
            
            # Add outcome columns - both date and binary indicator
            cohort_df['visits_unstable_angina'] = cohort_df['pid'].map(unstable_angina_dates).dt.date
            cohort_df['visits_binary_unstable_angina'] = cohort_df['visits_unstable_angina'].notna().astype(int)
            
            angina_count = cohort_df['visits_binary_unstable_angina'].sum()
            logging.info(f"  - Found {angina_count} patients with Unstable Angina in follow-up period ({angina_count/len(cohort_df)*100:.2f}%)")
        else:
            cohort_df['visits_unstable_angina'] = pd.NaT
            cohort_df['visits_binary_unstable_angina'] = 0
    else:
        cohort_df['visits_unstable_angina'] = pd.NaT
        cohort_df['visits_binary_unstable_angina'] = 0
    
    # BASELINE PERIOD: Find Angina diagnoses in the year before entry date
    baseline_angina = cohort_dx_df[
        (cohort_dx_df['dx_date'] > cohort_dx_df['entry_date'] - pd.Timedelta(days=365)) &
        (cohort_dx_df['dx_date'] <= cohort_dx_df['entry_date']) &
        (cohort_dx_df['code'].str.startswith(tuple(unstable_angina_codes), na=False))
    ]
    
    if not baseline_angina.empty:
        # Create date range for each baseline angina diagnosis (±1 day)
        baseline_angina = baseline_angina.copy()
        baseline_angina.loc[:, 'dx_date_minus1'] = baseline_angina['dx_date'] - pd.Timedelta(days=1)
        baseline_angina.loc[:, 'dx_date_plus1'] = baseline_angina['dx_date'] + pd.Timedelta(days=1)
        
        # Prepare core data for baseline period - filter for inpatient visits only
        baseline_inpatient = core_df[
            (core_df['patient_class'].str.lower() == 'inpatient') & 
            (core_df['visit_date'] >= (cohort_df['entry_date'].min() - pd.Timedelta(days=366))) &
            (core_df['visit_date'] <= cohort_df['entry_date'].max())
        ]
        
        if not baseline_inpatient.empty:
            # Sort for merge_asof
            baseline_angina = baseline_angina.sort_values('dx_date')
            baseline_inpatient = baseline_inpatient.sort_values('visit_date')
            
            # Find visits that occurred on or after the diagnosis date - 1 day
            baseline_visits_after = pd.merge_asof(
                baseline_angina[['pid', 'dx_date', 'dx_date_minus1', 'dx_date_plus1']],
                baseline_inpatient[['pid', 'visit_date']],
                left_on='dx_date_minus1',
                right_on='visit_date',
                by='pid',
                direction='forward'
            )
            
            # Then filter for visits that occurred before or on diagnosis date + 1 day
            baseline_valid_visits = baseline_visits_after[baseline_visits_after['visit_date'] <= baseline_visits_after['dx_date_plus1']]
            
            # Set baseline flag based on existence of records
            cohort_df['pre_visits_unstable_angina'] = cohort_df['pid'].isin(baseline_valid_visits['pid']).astype(int)
        else:
            cohort_df['pre_visits_unstable_angina'] = 0
    else:
        cohort_df['pre_visits_unstable_angina'] = 0
    
    baseline_angina_count = cohort_df['pre_visits_unstable_angina'].sum()
    logging.info(f"  - Found {baseline_angina_count} patients with Unstable Angina in baseline period ({baseline_angina_count/len(cohort_df)*100:.2f}%)")
    
    return cohort_df

def count_baseline_healthcare_utilization(cohort_df, core_df):
    """
    Count the number of outpatient, inpatient, and emergency department visits 
    during the baseline period for each patient
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    core_df : pandas.DataFrame
        Core visit dataframe with pid, visit_date, and patient_class columns
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with three new columns added:
        - outpatient_visits: number of outpatient visits during baseline
        - inpatient_visits: number of inpatient visits during baseline
        - ed_visits: number of emergency department visits during baseline
    """
    logging.info("Counting baseline healthcare utilization...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty:
        logging.info("Missing cohort data for healthcare utilization counting")
        return cohort_df
    
    # Create a complete copy of the cohort dataframe to avoid SettingWithCopyWarning
    cohort_df = cohort_df.copy()
    
    if core_df is None or core_df.empty:
        logging.info("Warning: No core visit data available, all utilization counts will be zero")
        # Add default columns with zero values
        cohort_df.loc[:, 'outpatient_visits'] = 0
        cohort_df.loc[:, 'inpatient_visits'] = 0
        cohort_df.loc[:, 'ed_visits'] = 0
        return cohort_df
    
    # Ensure required columns exist in core_df
    required_columns = ['pid', 'visit_date', 'patient_class']
    if not all(col in core_df.columns for col in required_columns):
        logging.info(f"Error: Missing required columns in core_df. Required: {required_columns}, Available: {core_df.columns.tolist()}")
        # Add default columns with zero values
        cohort_df.loc[:, 'outpatient_visits'] = 0
        cohort_df.loc[:, 'inpatient_visits'] = 0
        cohort_df.loc[:, 'ed_visits'] = 0
        return cohort_df
    
    # Ensure we have entry_date in correct format
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Convert visit_date to datetime
    if not pd.api.types.is_datetime64_dtype(core_df['visit_date']):
        core_df['visit_date'] = pd.to_datetime(core_df['visit_date'].astype(str))
    
    # Perform a join to get cohort data alongside core visit data
    joined_df = pd.merge(
        cohort_df[['pid', 'entry_date']], 
        core_df[['pid', 'visit_date', 'patient_class']],
        on='pid',
        how='inner'  # Only keep patients that exist in both dataframes
    )
    
    # Filter for baseline period (1 year before entry date, not including entry date)
    baseline_visits = joined_df[
        (joined_df['visit_date'] >= (joined_df['entry_date'] - pd.Timedelta(days=365))) & 
        (joined_df['visit_date'] < joined_df['entry_date'])
    ]
    
    if baseline_visits.empty:
        logging.info("No baseline visits found for any patients")
        # Add default columns with zero values
        cohort_df.loc[:, 'outpatient_visits'] = 0
        cohort_df.loc[:, 'inpatient_visits'] = 0
        cohort_df.loc[:, 'ed_visits'] = 0
        return cohort_df
    
    # Standardize patient_class values to lowercase
    baseline_visits = baseline_visits.copy()
    baseline_visits.loc[:, 'patient_class'] = baseline_visits['patient_class'].str.lower()
    
    # Initialize visit_type as empty string
    baseline_visits.loc[:, 'visit_type'] = ''
    baseline_visits.loc[baseline_visits['patient_class'].str.contains('inpatient', case=False, na=False), 'visit_type'] = 'Inpatient'
    baseline_visits.loc[baseline_visits['patient_class'].str.contains('outpatient', case=False, na=False), 'visit_type'] = 'Outpatient'
    baseline_visits.loc[baseline_visits['patient_class'].str.contains('emergency', case=False, na=False), 'visit_type'] = 'ED'
    
    # Only keep visit records with a valid classification
    baseline_visits = baseline_visits[baseline_visits['visit_type'] != '']
    
    # Count visits by type for each patient
    visit_counts = baseline_visits.groupby(['pid', 'visit_type']).size().reset_index(name='count')
    
    # Pivot to get counts by visit type
    utilization = pd.pivot_table(
        visit_counts, 
        values='count', 
        index='pid', 
        columns='visit_type', 
        fill_value=0
    ).reset_index()
    
    # Rename columns for clarity
    utilization_columns = {
        'Inpatient': 'inpatient_visits', 
        'Outpatient': 'outpatient_visits', 
        'ED': 'ed_visits'
    }
    
    # Rename existing columns
    for old_col, new_col in utilization_columns.items():
        if old_col in utilization.columns:
            utilization.rename(columns={old_col: new_col}, inplace=True)
    
    # Make sure all columns exist, even if there are no such visits
    for old_col, new_col in utilization_columns.items():
        if new_col not in utilization.columns:
            utilization[new_col] = 0
    
    # Merge utilization data back to cohort dataframe
    result_df = pd.merge(cohort_df, utilization, on='pid', how='left')
    
    # Fill NAs with 0
    for col in ['outpatient_visits', 'inpatient_visits', 'ed_visits']:
        if col in result_df.columns:
            result_df[col] = result_df[col].fillna(0).astype(int)
        else:
            result_df[col] = 0
    
    # Count patients with at least one visit of each type
    outpatient_count = (result_df['outpatient_visits'] > 0).sum()
    inpatient_count = (result_df['inpatient_visits'] > 0).sum()
    ed_count = (result_df['ed_visits'] > 0).sum()
    
    # Count patients with at least one visit of any type
    any_visit_count = ((result_df['outpatient_visits'] > 0) | 
                       (result_df['inpatient_visits'] > 0) | 
                       (result_df['ed_visits'] > 0)).sum()
    
    # Output statistics
    total_patients = len(result_df)
    logging.info(f"Patients with outpatient visits: {outpatient_count} ({outpatient_count/total_patients*100:.2f}%)")
    logging.info(f"Patients with inpatient visits: {inpatient_count} ({inpatient_count/total_patients*100:.2f}%)")
    logging.info(f"Patients with ED visits: {ed_count} ({ed_count/total_patients*100:.2f}%)")
    logging.info(f"Patients with at least one visit of any type: {any_visit_count} ({any_visit_count/total_patients*100:.2f}%)")
    
    # Confirm columns were created successfully
    logging.info(f"Healthcare utilization columns added: outpatient_visits, inpatient_visits, ed_visits")
    
    return result_df

def process_baseline_medical_conditions(cohort_df, dx_df, cluster_file=os.path.join(code_set_folder_dir, 'cluster_master.csv')):
    """
    Process baseline diagnosis data to identify medical conditions based on cluster_master.csv
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    dx_df : pandas.DataFrame
        Diagnosis dataframe with pid, dx_date, and code columns
    cluster_file : str
        Path to the cluster master file with concept_code and cluster columns
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with medical_{cluster} columns added for baseline conditions
    """
    logging.info("Processing baseline medical conditions...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or dx_df is None or dx_df.empty:
        logging.info("Missing input data for baseline medical condition processing")
        return cohort_df
    
    # Ensure we have entry_date in correct format
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Convert dx_date to datetime
    dx_df['dx_date'] = pd.to_datetime(dx_df['dx_date'].astype(str))
    
    # Read cluster master file to get concept codes and clusters
    try:
        # Read cluster master file
        cluster_data = pd.read_csv(cluster_file)
        logging.info(f"Loaded cluster master file with {cluster_data.shape[0]} records")
        
        # Ensure concept_code is treated as string
        cluster_data['concept_code'] = cluster_data['concept_code'].astype(str).str.upper()
        
        # Get unique clusters
        clusters = cluster_data['cluster'].unique()
        logging.info(f"Found {len(clusters)} unique clusters")
        
    except Exception as e:
        logging.info(f"Error reading cluster master file: {str(e)}")
        return cohort_df
    
    # Join cohort with dx data for the baseline period
    logging.info("Joining cohort with diagnosis data...")
    joined_df = pd.merge(
        cohort_df[['pid', 'entry_date']], 
        dx_df[['pid', 'dx_date', 'code']],
        on='pid',
        how='inner'  # Only keep patients with diagnosis records
    )
    
    # Filter for baseline period
    baseline_dx = joined_df[
        (joined_df['dx_date'] >= (joined_df['entry_date'] - pd.Timedelta(days=365))) & 
        (joined_df['dx_date'] <= joined_df['entry_date'])
    ].copy()
    
    if baseline_dx.empty:
        logging.info("No baseline diagnoses found")
        return cohort_df
    
    logging.info(f"Processing {baseline_dx.shape[0]} baseline diagnosis records")
    
    # Ensure code is treated as string and standardize to uppercase
    baseline_dx['code'] = baseline_dx['code'].astype(str).str.upper()
    
    # Create a mapping DataFrame that maps diagnosis codes to clusters
    # This will be used for a merge operation
    code_to_cluster = cluster_data[['concept_code', 'cluster']].copy()
    
    # Add a prefix to the cluster column to create the final column names
    code_to_cluster['medical_column'] = 'medical_' + code_to_cluster['cluster']
    
    # Merge baseline diagnosis with code-to-cluster mapping
    # This directly maps each diagnosis to its corresponding cluster
    dx_with_cluster = pd.merge(
        baseline_dx, 
        code_to_cluster,
        left_on='code',
        right_on='concept_code',
        how='inner'
    )
    
    # If no matches found after merging, return original cohort
    if dx_with_cluster.empty:
        logging.info("No matching diagnoses found in cluster master file")
        return cohort_df
    
    logging.info(f"Found {dx_with_cluster.shape[0]} matching diagnosis records")
    
    # Create a pivot table to get medical_{cluster} flags for each patient
    # 1. Add a constant column for aggregation
    dx_with_cluster['has_condition'] = 1
    
    # 2. Create the pivot table: rows=pid, columns=medical_column, values=has_condition, aggregation=max
    # This gives us a binary flag (0/1) for each patient and condition
    medical_conditions = pd.pivot_table(
        dx_with_cluster,
        index='pid',
        columns='medical_column',
        values='has_condition',
        aggfunc='max',
        fill_value=0
    ).reset_index()
    
    # Count the number of initial medical columns
    initial_medical_cols = len([col for col in medical_conditions.columns if col.startswith('medical_')])
    logging.info(f"Initial number of medical condition columns: {initial_medical_cols}")
    
    # Merge conditions back to cohort
    result_df = pd.merge(cohort_df, medical_conditions, on='pid', how='left')
    
    # Fill NAs with 0 for all medical_ columns
    medical_cols = [col for col in result_df.columns if col.startswith('medical_')]
    for col in medical_cols:
        result_df[col] = result_df[col].fillna(0).astype(int)
    
    # Calculate prevalence for each condition and identify low-prevalence columns
    # Using a more efficient vectorized approach
    prevalence = result_df[medical_cols].mean() * 100
    condition_stats = pd.DataFrame({
        'count': result_df[medical_cols].sum(),
        'prevalence': prevalence
    })
    
    # Instead of printing each condition, just print summary statistics
    logging.info(f"Calculated prevalence for {len(condition_stats)} medical conditions")
    
    # Identify columns with prevalence < 1%
    low_prev_cols = condition_stats[condition_stats['prevalence'] < 1.0].index.tolist()
    logging.info(f"Found {len(low_prev_cols)} conditions with prevalence < 1%")
    
    # Remove columns with prevalence < 1%
    result_df = result_df.drop(columns=low_prev_cols)
    
    # Count the number of remaining medical columns
    final_medical_cols = len([col for col in result_df.columns if col.startswith('medical_')])
    logging.info(f"Final number of medical condition columns after removing low prevalence: {final_medical_cols}")
    logging.info(f"Removed {initial_medical_cols - final_medical_cols} columns with prevalence < 1%")
    
    return result_df

def process_baseline_medications_detail(cohort_df, med_df, med_cluster_file=os.path.join(code_set_folder_dir, 'huilin_med.csv')):
    """
    Process baseline medication data to create flags based on medication clusters.

    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    med_df : pandas.DataFrame
        Medication dataframe with pid, start_date and medication_name columns
    med_cluster_file : str
        Path to the medication cluster file with cluster and name_list columns
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with med_{cluster} columns added for baseline medications
    """
    logging.info("Processing baseline medication details...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or med_df is None or med_df.empty:
        logging.info("Missing input data for medication processing")
        return cohort_df
    
    # Ensure entry_date format is correct
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Read medication cluster file
    try:
        # Read medication cluster file
        med_clusters = pd.read_csv(med_cluster_file)
        logging.info(f"Loaded medication cluster file with {len(med_clusters)} medication categories")
        
        # Process medication name lists, create medication cluster dictionary
        med_dict = {}
        for _, row in med_clusters.iterrows():
            # Split name list and clean up each term
            names = [name.strip().lower() for name in row['name_list'].split(',')]
            med_dict[row['cluster']] = names
            
        logging.info(f"Processed {len(med_dict)} medication clusters")
        
    except Exception as e:
        logging.info(f"Error reading medication cluster file: {str(e)}")
        return cohort_df
    
    # Ensure medication data date column format is correct
    if 'start_date' in med_df.columns:
        med_df = med_df.rename(columns={'start_date': 'med_date'})
    elif 'entry_date' in med_df.columns:
        med_df = med_df.rename(columns={'entry_date': 'med_date'})
    
    # Convert med_date to datetime format
    med_df['med_date'] = pd.to_datetime(med_df['med_date'].astype(str))
    
    # Join cohort with medication data
    logging.info("Joining cohort with medication data...")
    cohort_med_df = pd.merge(
        med_df[['pid', 'med_date', 'medication_name']], 
        cohort_df[['pid', 'entry_date']],
        on='pid',
        how='inner'
    )
    
    # Filter baseline period data
    baseline_med = cohort_med_df[
        (cohort_med_df['med_date'] >= (cohort_med_df['entry_date'] - pd.Timedelta(days=365))) &
        (cohort_med_df['med_date'] <= cohort_med_df['entry_date'])
    ]
    
    if baseline_med.empty:
        logging.info("No baseline medication records found")
        return cohort_df
    
    logging.info(f"Processing {len(baseline_med)} baseline medication records")
    
    # Create flag columns for each medication cluster
    # Initialize result dataframe, ensuring all original pids are preserved
    result_df = cohort_df.copy()
    
    # Create flags for each medication cluster
    for cluster, med_names in med_dict.items():
        column_name = f'med_{cluster}'
        logging.info(f"  - Processing medication cluster: {cluster}")
        
        # Use regex to match medication names
        med_pattern = '|'.join(med_names)
        med_flag = baseline_med['medication_name'].str.contains(med_pattern, case=False, na=False)
        
        # Get list of patients using this type of medication
        med_pids = baseline_med[med_flag]['pid'].unique()
        
        # Create flag column
        result_df[column_name] = result_df['pid'].isin(med_pids).astype(int)
        
        # Print usage statistics for this medication cluster
        count = result_df[column_name].sum()
        prevalence = count / len(result_df) * 100
        logging.info(f"    {column_name}: {count} patients ({prevalence:.2f}%)")
    
    # Count the number of added medication flag columns
    med_cols = [col for col in result_df.columns if col.startswith('med_')]
    logging.info(f"Added {len(med_cols)} medication flag columns")
    
    return result_df

def process_baseline_lab_tests(cohort_df, lab_df):
    """
    Process baseline laboratory test data to get values closest to entry date.

    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    lab_df : pandas.DataFrame
        Laboratory test dataframe with pid, result_date, result_loinc_code and result_num columns
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with lab_* columns added for baseline lab values
    """
    logging.info("Processing baseline laboratory test data...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or lab_df is None or lab_df.empty:
        logging.info("Missing input data for laboratory test processing")
        return cohort_df
    
    # Ensure entry_date format is correct
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Define laboratory test LOINC code mapping
    lab_code_dict = {
        '4548-4': 'lab_HbA1c',          # Hemoglobin A1c
        '2571-8': 'lab_Triglycerides',  # Triglycerides
        '2085-9': 'lab_HDL',            # HDL Cholesterol
        '13457-7': 'lab_LDL',           # LDL Cholesterol
        '2093-3': 'lab_total_cholesterol', # Total Cholesterol
        '48643-1': 'lab_eGFR',          # Estimated GFR
        '1742-6': 'lab_ALT',            # Alanine Aminotransferase
        '1920-8': 'lab_AST'             # Aspartate Aminotransferase
    }
    
    logging.info(f"Will process {len(lab_code_dict)} laboratory test types")
    
    # Ensure laboratory test data date column format is correct
    if 'result_date' in lab_df.columns:
        lab_df['lab_date'] = pd.to_datetime(lab_df['result_date'].astype(str))
    else:
        logging.info("result_date column not found in laboratory test data")
        return cohort_df
    
    # Join cohort with laboratory test data
    logging.info("Joining cohort with laboratory test data...")
    lab_data = pd.merge(
        lab_df[['pid', 'lab_date', 'result_loinc_code', 'result_num']], 
        cohort_df[['pid', 'entry_date']],
        on='pid',
        how='inner'
    )
    
    # Filter baseline period data and relevant LOINC codes
    baseline_lab = lab_data[
        (lab_data['lab_date'] >= (lab_data['entry_date'] - pd.Timedelta(days=365))) &
        (lab_data['lab_date'] <= lab_data['entry_date']) &
        (lab_data['result_loinc_code'].isin(lab_code_dict.keys()))
    ].copy()
    
    if baseline_lab.empty:
        logging.info("No baseline laboratory test records found")
        return cohort_df
    
    logging.info(f"Processing {len(baseline_lab)} baseline laboratory test records")
    
    # Initialize result dataframe, ensuring all original pids are preserved
    result_df = cohort_df.copy()
    
    # Calculate time difference for each record from entry date, used to find closest record
    baseline_lab['time_diff'] = (baseline_lab['entry_date'] - baseline_lab['lab_date']).abs()
    
    # Get closest value to entry date for each test type
    for code, col_name in lab_code_dict.items():
        # Filter records for specific code
        code_records = baseline_lab[baseline_lab['result_loinc_code'] == code]
        
        if not code_records.empty:
            # For each patient, select the record with smallest time difference
            closest_records = code_records.sort_values('time_diff').groupby('pid').first()
            
            # Add test values to result dataframe
            result_df[col_name] = result_df['pid'].map(closest_records['result_num'])
            
            # Print statistics for this test type
            count = result_df[col_name].notna().sum()
            coverage = count / len(result_df) * 100
            logging.info(f"  - {col_name}: {count} patients have records ({coverage:.2f}% coverage)")
    
    # Count the number of added laboratory test columns
    lab_cols = [col for col in result_df.columns if col.startswith('lab_') and col not in ['lab_bmi', 'lab_sbp', 'lab_dbp']]
    logging.info(f"Added {len(lab_cols)} laboratory test value columns")
    
    return result_df

def process_baseline_vital_signs(cohort_df, vital_df):
    """
    Process baseline vital signs data to get values closest to entry date.

    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    vital_df : pandas.DataFrame
        Vital signs dataframe with pid, recorded_time, vital_name and result_vital columns
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with lab_bmi, lab_sbp and lab_dbp columns added for baseline vital signs
    """
    logging.info("Processing baseline vital signs data...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or vital_df is None or vital_df.empty:
        logging.info("Missing input data for vital signs processing")
        return cohort_df
    
    # Ensure entry_date format is correct
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    
    # Ensure vital signs data date column format is correct
    if 'recorded_time' in vital_df.columns:
        vital_df['vital_date'] = pd.to_datetime(vital_df['recorded_time'].astype(str))
    else:
        logging.info("recorded_time column not found in vital signs data")
        return cohort_df
    
    # Join cohort with vital signs data
    logging.info("Joining cohort with vital signs data...")
    vital_data = pd.merge(
        vital_df[['pid', 'vital_date', 'vital_name', 'result_vital']], 
        cohort_df[['pid', 'entry_date']],
        on='pid',
        how='inner'
    )
    
    # Filter baseline period data
    baseline_vital = vital_data[
        (vital_data['vital_date'] >= (vital_data['entry_date'] - pd.Timedelta(days=365))) &
        (vital_data['vital_date'] <= vital_data['entry_date'])
    ].copy()
    
    if baseline_vital.empty:
        logging.info("No baseline vital signs records found")
        return cohort_df
    
    logging.info(f"Processing {len(baseline_vital)} baseline vital signs records")
    
    # Initialize result dataframe, ensuring all original pids are preserved
    result_df = cohort_df.copy()
    
    # Calculate time difference for each record from entry date, used to find closest record
    baseline_vital['time_diff'] = (baseline_vital['entry_date'] - baseline_vital['vital_date']).abs()
    
    # Process BMI
    logging.info("Processing BMI data...")
    bmi_records = baseline_vital[baseline_vital['vital_name'].str.contains('BMI', case=False, na=False)]
    if not bmi_records.empty:
        # For each patient, select the record with smallest time difference
        closest_bmi = bmi_records.sort_values('time_diff').groupby('pid').first()
        
        # Add BMI values to result dataframe
        result_df['lab_bmi'] = result_df['pid'].map(closest_bmi['result_vital'])
        
        # Print statistics for BMI records
        count = result_df['lab_bmi'].notna().sum()
        coverage = count / len(result_df) * 100
        logging.info(f"  - lab_bmi: {count} patients have records ({coverage:.2f}% coverage)")
    
    # Process Blood Pressure
    logging.info("Processing blood pressure data...")
    bp_records = baseline_vital[baseline_vital['vital_name'].str.contains('BP', case=False, na=False)]
    if not bp_records.empty:
        # Try to split blood pressure values (format: systolic/diastolic)
        try:
            # Create temporary systolic and diastolic columns
            bp_records[['sbp', 'dbp']] = bp_records['result_vital'].str.split('/', expand=True).astype(float)
            
            # For each patient, select the record with smallest time difference
            closest_bp = bp_records.sort_values('time_diff').groupby('pid').first()
            
            # Add systolic and diastolic values to result dataframe
            result_df['lab_sbp'] = result_df['pid'].map(closest_bp['sbp'])
            result_df['lab_dbp'] = result_df['pid'].map(closest_bp['dbp'])
            
            # Print statistics for blood pressure records
            sbp_count = result_df['lab_sbp'].notna().sum()
            sbp_coverage = sbp_count / len(result_df) * 100
            logging.info(f"  - lab_sbp: {sbp_count} patients have records ({sbp_coverage:.2f}% coverage)")
            
            dbp_count = result_df['lab_dbp'].notna().sum()
            dbp_coverage = dbp_count / len(result_df) * 100
            logging.info(f"  - lab_dbp: {dbp_count} patients have records ({dbp_coverage:.2f}% coverage)")
        except Exception as e:
            logging.info(f"Error processing blood pressure data: {str(e)}")
            logging.info("Blood pressure format may not be the standard 'systolic/diastolic' format")
    
    # Count the number of added vital signs columns
    vital_cols = ['lab_bmi', 'lab_sbp', 'lab_dbp']
    added_cols = [col for col in vital_cols if col in result_df.columns]
    logging.info(f"Added {len(added_cols)} vital signs columns")
    
    return result_df

@time_execution
def process_nco_outcomes(cohort_df, dx_df, icd_file=os.path.join(code_set_folder_dir, 'glp_nco_icd_0225.csv')):
    """
    Process Non-Cardiovascular Outcomes (NCO) based on diagnosis records
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        Cohort dataframe with pid and entry_date columns
    dx_df : pandas.DataFrame
        Diagnosis dataframe with pid, dx_date, and code columns
    icd_file : str
        Path to the ICD codes file defining NCO conditions
        
    Returns:
    --------
    pandas.DataFrame
        Cohort dataframe with NCO outcomes added:
        - NCO_{condition}: date of first diagnosis during follow-up
        - NCO_binary_{condition}: binary flag (0/1) for diagnosis during follow-up
        - pre_NCO_{condition}: binary flag (0/1) for diagnosis during baseline
    """
    logging.info("Processing Non-Cardiovascular Outcomes (NCO)...")
    
    # Check if input data is available
    if cohort_df is None or cohort_df.empty or dx_df is None or dx_df.empty:
        logging.warning("Missing input data for NCO outcome processing")
        return cohort_df
    
    # Create a complete copy of the cohort dataframe
    cohort_df = cohort_df.copy()
    
    # Ensure pid is string type in both dataframes
    cohort_df['pid'] = cohort_df['pid'].astype(str)
    dx_df['pid'] = dx_df['pid'].astype(str)
    
    # Ensure entry_date is datetime format
    cohort_df['entry_date'] = pd.to_datetime(cohort_df['entry_date'].astype(str))
    dx_df['dx_date'] = pd.to_datetime(dx_df['dx_date'].astype(str))
    
    # Create baseline period
    cohort_df['baseline_start'] = cohort_df['entry_date'] - pd.Timedelta(days=365)
    
    try:
        # Read NCO codes file
        nco_codes = pd.read_csv(icd_file)
        logging.info(f"Found {len(nco_codes)} NCO conditions")
        
        # Create dictionary for each NCO condition with its ICD codes
        nco_dict = {}
        for _, row in nco_codes.iterrows():
            codes = [code.strip() for code in str(row['ICD-10']).split(',')]
            nco_dict[row['name']] = codes
        
    except Exception as e:
        logging.error(f"Error reading NCO ICD codes file: {str(e)}")
        return cohort_df
    
    # Initialize dictionaries to store results
    pre_nco_dict = {}  # For baseline period flags
    nco_date_dict = {}  # For follow-up dates
    nco_binary_dict = {}  # For follow-up binary flags
    
    # Create baseline and followup masks
    merged_dx_data = pd.merge(
        dx_df,
        cohort_df[['pid', 'entry_date', 'baseline_start']], 
        on='pid'
    )
    
    baseline_mask = (
        (merged_dx_data['dx_date'] >= merged_dx_data['baseline_start']) & 
        (merged_dx_data['dx_date'] < merged_dx_data['entry_date'])
    )
    followup_mask = merged_dx_data['dx_date'] >= merged_dx_data['entry_date']
    
    # Split diagnosis data for faster processing
    baseline_dx_filtered = merged_dx_data[baseline_mask]
    followup_dx_filtered = merged_dx_data[followup_mask]
    
    # Process each NCO condition
    for nco_name, icd_codes in nco_dict.items():
        # Create regex pattern to match ICD codes
        pattern = '|'.join([f'^{code}' for code in icd_codes])
        
        # Baseline NCO (0/1 flag)
        if not baseline_dx_filtered.empty:
            baseline_match = baseline_dx_filtered['code'].str.contains(pattern, case=False, regex=True, na=False)
            if baseline_match.any():
                baseline_pids = baseline_dx_filtered[baseline_match]['pid'].unique()
                pre_nco_dict[f'pre_NCO_{nco_name}'] = cohort_df['pid'].isin(baseline_pids).astype(int)
            else:
                pre_nco_dict[f'pre_NCO_{nco_name}'] = pd.Series(0, index=cohort_df.index)
        else:
            pre_nco_dict[f'pre_NCO_{nco_name}'] = pd.Series(0, index=cohort_df.index)
        
        # Follow-up NCO (first diagnosis date)
        if not followup_dx_filtered.empty:
            followup_match = followup_dx_filtered['code'].str.contains(pattern, case=False, regex=True, na=False)
            if followup_match.any():
                first_diagnosis = followup_dx_filtered[followup_match].groupby('pid')['dx_date'].min()
                nco_date_dict[f'NCO_{nco_name}'] = cohort_df['pid'].map(
                    lambda x: first_diagnosis.get(x) if x in first_diagnosis.index else None
                )
                # Binary NCO (0/1 flag for follow-up period)
                followup_pids = followup_dx_filtered[followup_match]['pid'].unique()
                nco_binary_dict[f'NCO_binary_{nco_name}'] = cohort_df['pid'].isin(followup_pids).astype(int)
            else:
                nco_binary_dict[f'NCO_binary_{nco_name}'] = pd.Series(0, index=cohort_df.index)
        else:
            nco_binary_dict[f'NCO_binary_{nco_name}'] = pd.Series(0, index=cohort_df.index)
        
        # Report statistics for this condition
        baseline_count = int(pre_nco_dict[f'pre_NCO_{nco_name}'].sum())
        followup_count = int(nco_binary_dict[f'NCO_binary_{nco_name}'].sum())
        logging.info(f"  - Found {followup_count:,} patients with {nco_name} outcome in follow-up period ({followup_count/len(cohort_df)*100:.2f}%)")
        logging.info(f"  - Found {baseline_count:,} patients with {nco_name} in baseline period ({baseline_count/len(cohort_df)*100:.2f}%)")
    
    # Add all new columns to cohort data at once
    # First create dataframes from the dictionaries
    pre_nco_df = pd.DataFrame(pre_nco_dict)
    nco_binary_df = pd.DataFrame(nco_binary_dict)
    
    # Create dataframe for dates if there are any
    if nco_date_dict:
        nco_date_df = pd.DataFrame(nco_date_dict)
        # Combine all dataframes
        all_new_cols = pd.concat([pre_nco_df, nco_binary_df, nco_date_df], axis=1)
    else:
        # Just combine flags
        all_new_cols = pd.concat([pre_nco_df, nco_binary_df], axis=1)
    
    # Add to cohort data
    result_df = pd.concat([cohort_df, all_new_cols], axis=1)
    
    # Drop temporary column
    result_df = result_df.drop('baseline_start', axis=1)
    
    return result_df

@time_execution
def main():
    """Main function for cohort extraction and processing"""

    # Setup logging
    log_file = setup_logging()
    logging.info("Starting cohort extraction script")
    logging.info(f"Log file: {log_file}")
    
    # Load the dataframes
    logging.info("Loading input dataframes...")
    dx_df = pd.read_parquet(dx_dir)
    logging.info(f"Loaded dx_df with {dx_df.shape[0]:,} records")
    
    core_df = pd.read_parquet(core_dir)
    logging.info(f"Loaded core_df with {core_df.shape[0]:,} records")
    
    med_df = pd.read_parquet(med_dir)
    logging.info(f"Loaded med_df with {med_df.shape[0]:,} records")
 
    # Original cohort building steps
    #===============================================
    logging.info("Starting cohort building process...")
    
    # Process antidiabetic medication cohorts
    logging.info("Processing antidiabetic medication cohorts...")
    antidiabetic_cohort = process_antidiabetic_cohort(med_df)
    
    # Process demographics, filter by birth date and age
    logging.info("Processing demographics...")
    demo_cohort = process_demographics(antidiabetic_cohort)
    
    # Process baseline medications  
    logging.info("Processing baseline medications...")
    medication_cohort = process_baseline_medications(demo_cohort, med_df)
    
    # Process baseline conditions
    logging.info("Processing baseline conditions...")
    t2d_cohort = process_baseline_conditions(medication_cohort, dx_df)
    
    # Process baseline visits
    logging.info("Processing baseline visits...")
    basic_cohort = process_baseline_visits(t2d_cohort, core_df)
    #===============================================   
    
    # Output summary statistics for basic cohort
    logging.info("Final basic cohort summary:")
    logging.info(f"Total patients: {basic_cohort.shape[0]:,}")
    logging.info("Drug class distribution:")
    drug_class_counts = basic_cohort['drug_class'].value_counts()
    for drug_class, count in drug_class_counts.items():
        logging.info(f"  {drug_class}: {count:,}")
    
    try:
        # The following steps are process outcomes and covariates.
        #===============================================
        logging.info("Processing outcomes and covariates...")
        
        # Process mental health outcomes
        logging.info("Processing mental health outcomes...")
        mental_cohort = process_mental_outcomes(basic_cohort, dx_df)
        logging.info(f"Added mental health outcomes to {mental_cohort.shape[0]:,} patients")
        
        # Process gastrointestinal outcomes
        logging.info("Processing gastrointestinal outcomes...")
        gi_cohort = process_gi_outcomes(mental_cohort, dx_df)
        logging.info(f"Added gastrointestinal outcomes to {gi_cohort.shape[0]:,} patients")
        
        # Process cardiac outcomes
        logging.info("Processing cardiac outcomes...")
        cardiac_cohort = process_cardiac_outcomes(gi_cohort, dx_df, core_df)
        logging.info(f"Added cardiac outcomes to {cardiac_cohort.shape[0]:,} patients")
        
        # Count baseline healthcare utilization
        logging.info("Counting baseline healthcare utilization...")
        utilization_cohort = count_baseline_healthcare_utilization(cardiac_cohort, core_df)
        logging.info(f"Added healthcare utilization to {utilization_cohort.shape[0]:,} patients")
        
        # Process baseline medical conditions
        logging.info("Processing baseline medical conditions...")
        medical_cohort = process_baseline_medical_conditions(utilization_cohort, dx_df)
        logging.info(f"Added baseline medical conditions to {medical_cohort.shape[0]:,} patients")
        
        # Process baseline medication details
        logging.info("Processing baseline medication details...")
        med_cohort = process_baseline_medications_detail(medical_cohort, med_df)
        logging.info(f"Added baseline medication details to {med_cohort.shape[0]:,} patients")
        
        # Process baseline lab tests
        # lab_cohort = process_baseline_lab_tests(med_cohort, lab_df)
        # logging.info(f"Added baseline laboratory test results to {lab_cohort.shape[0]:,} patients")
        
        # Process baseline vital signs
        # vital_cohort = process_baseline_vital_signs(lab_cohort, vital_df)
        # logging.info(f"Added baseline vital signs to {vital_cohort.shape[0]:,} patients")

        # Process NCO outcomes
        logging.info("Processing NCO outcomes...")
        final_cohort = process_nco_outcomes(med_cohort, dx_df)
        logging.info(f"Added NCO outcomes to {final_cohort.shape[0]:,} patients")
        #===============================================

        # Save the final cohort with all features to CSV
        output_file = os.path.join(output_dir, 'cohort.csv')
        logging.info(f"Saving cohort with all features to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Final guard: ensure one row per pid before saving
        pre_save_rows, pre_save_unique = final_cohort.shape[0], final_cohort['pid'].nunique()
        if pre_save_rows != pre_save_unique:
            logging.info(f"Final cohort has duplicates before save: rows={pre_save_rows:,}, unique_pid={pre_save_unique:,}. Deduplicating by pid.")
            final_cohort = final_cohort.sort_values(['pid', 'entry_date']).drop_duplicates(subset=['pid'], keep='first')
            logging.info(f"After final deduplication: rows={final_cohort.shape[0]:,}, unique_pid={final_cohort['pid'].nunique():,}")
        final_cohort.to_csv(output_file, index=False)
        logging.info(f"Saved {final_cohort.shape[0]:,} patients to cohort file with all features")
        
        logging.info("Cohort extraction completed successfully!")
        
    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()


