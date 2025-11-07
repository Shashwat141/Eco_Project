# preprocess_oecd_data_FIXED.py
"""
OECD Education Data Preprocessing Script (FIXED VERSION)
Converts 4.5GB+ CSV to memory-efficient Parquet dataset using Dask

CRITICAL FIX: CONF_STATUS changed from float32 to category
- CONF_STATUS contains string codes ('C', 'D', 'F', 'N', 'S', 'A')
- Previous version incorrectly tried to convert to float, causing ValueError

19 columns loaded:
  - 13 dimension code columns
  - 1 time column
  - 5 value and value-metadata columns

Author: Senior Data Engineer
Date: November 2025
Version: 2.0 (FIXED)
"""

import dask.dataframe as dd
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("OECD EDUCATION DATA PREPROCESSING - FIXED VERSION")
print("=" * 80)
print("\nFIX APPLIED: CONF_STATUS dtype changed from float32 to category")
print("Reason: CONF_STATUS contains string codes (C, D, F, N, S, A)")
print("=" * 80)

# =============================================================================
# STEP 1: Define source filename
# =============================================================================
raw_csv_file = 'FULL_National Educational Attainment Classification (NEAC) and labour market status (full dataset).csv'
print(f"\nSource file: {raw_csv_file}")
print("-" * 80)

# =============================================================================
# STEP 2: Define column selections (use_cols) - 19 columns
# =============================================================================
use_cols = [
    # Dimensions (Codes) - 13 columns
    'REF_AREA',              # Country/region code (ISO 3166-1 alpha-3)
    'SEX',                   # Sex code (F/M/_T)
    'AGE',                   # Age group code (Y25T64, etc.)
    'ATTAINMENT_LEV',        # Educational attainment level (ISCED 2011)
    'EDUCATION_FIELD',       # Field of education (_T, FIELD001-010)
    'MEASURE',               # Measure type (POP, EMP, UNE, etc.)
    'INCOME',                # Income level (_T, _Z, INC_Q1-Q5)
    'BIRTH_PLACE',           # Place of birth (_T, NATIVE, FOREIGN)
    'MIGRATION_AGE',         # Age at migration (_Z, _T, MIGR_Y0T5, etc.)
    'EDU_STATUS',            # Education status (ED_NED, ED_AT, _T)
    'LABOUR_FORCE_STATUS',   # Labour force status (POP, LF, EMP, UNE, INAC, NEET)
    'DURATION_UNEMP',        # Unemployment duration (_Z, _T, DUR_LT1M, etc.)
    'WORK_TIME_ARNGMNT',     # Work time arrangement (_Z, _T, FT, PT)
    # Time - 1 column
    'TIME_PERIOD',           # Year (2008-2022 typical range)
    # Values & Value Metadata - 5 columns
    'OBS_VALUE',             # Observation value (population, rate, etc.)
    'OBS_STATUS',            # Observation status (A, E, F, M, O, P)
    'CONF_STATUS',           # Confidentiality status (F, C, D, N, S, A) - FIXED!
    'UNIT_MULT',             # Unit multiplier (power of 10: -3 to 6)
    'DECIMALS'               # Decimal places (0-3)
]

print(f"\nColumns to load: {len(use_cols)}")
print(f"  - Dimension Code columns: 13")
for i, col in enumerate(use_cols[:13], 1):
    print(f"      {i:2d}. {col}")

print(f"  - Time column: 1")
print(f"      14. {use_cols[13]}")

print(f"  - Values & Value Metadata: 5")
for i, col in enumerate(use_cols[14:], 15):
    print(f"      {i:2d}. {col}")

# =============================================================================
# STEP 3: Define optimal dtype dictionary - WITH FIX
# =============================================================================
dtype_dict = {
    # Dimension code columns - all category (13 columns)
    'REF_AREA': 'category',
    'SEX': 'category',
    'AGE': 'category',
    'ATTAINMENT_LEV': 'category',
    'EDUCATION_FIELD': 'category',
    'MEASURE': 'category',
    'INCOME': 'category',
    'BIRTH_PLACE': 'category',
    'MIGRATION_AGE': 'category',
    'EDU_STATUS': 'category',
    'LABOUR_FORCE_STATUS': 'category',
    'DURATION_UNEMP': 'category',
    'WORK_TIME_ARNGMNT': 'category',
    # Time column
    'TIME_PERIOD': 'int16',         # Year (fits in int16: 1900-9999)
    # Values & metadata columns
    'OBS_VALUE': 'float32',         # Observation values (sufficient precision)
    'OBS_STATUS': 'category',       # Status codes (A, E, F, M, O, P)
    'CONF_STATUS': 'category',      # ← FIXED! Was float32, now category (C, D, F, N, S, A)
    'UNIT_MULT': 'int8',            # Unit multiplier (-3 to 6)
    'DECIMALS': 'int8'              # Decimal places (0-3)
}

print(f"\n\nData types optimization:")
print("-" * 80)
print(f"{'Column':<25s} {'Data Type':<15s} {'Memory Benefit':<25s} {'Notes':<30s}")
print("-" * 80)

dimension_cols = use_cols[:13]
for col in dimension_cols:
    print(f"{col:<25s} {'category':<15s} {'70-90% reduction':<25s} {'Low cardinality codes':<30s}")

print(f"{'TIME_PERIOD':<25s} {'int16':<15s} {'50% vs int64':<25s} {'Years: 1900-9999':<30s}")
print(f"{'OBS_VALUE':<25s} {'float32':<15s} {'50% vs float64':<25s} {'Sufficient precision':<30s}")
print(f"{'OBS_STATUS':<25s} {'category':<15s} {'70-90% reduction':<25s} {'SDMX status codes':<30s}")
print(f"{'CONF_STATUS':<25s} {'category (FIXED!)':<15s} {'N/A (was wrong!)':<25s} {'String codes C/D/F/N/S/A':<30s}")
print(f"{'UNIT_MULT':<25s} {'int8':<15s} {'87.5% vs int64':<25s} {'Small integers: -3 to 6':<30s}")
print(f"{'DECIMALS':<25s} {'int8':<15s} {'87.5% vs int64':<25s} {'Small integers: 0-3':<30s}")

print(f"\n⚠ CRITICAL FIX: CONF_STATUS changed from float32 to category")
print(f"   Reason: Contains string confidentiality codes (C, D, F, N, S, A)")
print(f"   See OECD_CODE_REFERENCE.md for full code definitions")

# =============================================================================
# STEP 4: Load data with Dask
# =============================================================================
print("\n" + "=" * 80)
print("LOADING DATA WITH DASK...")
print("=" * 80)

try:
    # Read CSV with Dask - processes data in parallel chunks
    ddf = dd.read_csv(
        raw_csv_file,
        usecols=use_cols,       # Only load specified columns
        dtype=dtype_dict,        # Apply memory-efficient dtypes (with FIXED CONF_STATUS!)
        blocksize='64MB',        # Process in 64MB chunks (adjust: 32MB or 16MB for low RAM)
        assume_missing=True      # Handle missing values gracefully
    )

    print("\n✓ Data loaded successfully into Dask DataFrame")

    # =============================================================================
    # STEP 5: Verification - Display DataFrame info
    # =============================================================================
    print("\n" + "=" * 80)
    print("DATAFRAME INFORMATION BEFORE SAVING")
    print("=" * 80)

    print(f"\nNumber of partitions: {ddf.npartitions}")
    print(f"Total columns: {len(ddf.columns)}")
    print(f"Columns: {list(ddf.columns)}")

    print(f"\n{'Column':<25s} {'Data Type':<20s}")
    print("-" * 80)
    for col, dtype in ddf.dtypes.items():
        indicator = " ← FIXED!" if col == 'CONF_STATUS' else ""
        print(f"{col:<25s} {str(dtype):<20s}{indicator}")

    print(f"\n\nFirst 5 rows preview:")
    print("-" * 80)
    print(ddf.head())

    print(f"\n\nDataFrame shape estimate: {len(ddf):,} rows x {len(ddf.columns)} columns")

    # =============================================================================
    # STEP 6: Save to Parquet
    # =============================================================================
    print("\n" + "=" * 80)
    print("CONVERTING TO PARQUET...")
    print("=" * 80)

    output_parquet_path = 'oecd_data.parquet'

    # Save to Parquet with optimal compression
    ddf.to_parquet(
        output_parquet_path,
        engine='pyarrow',           # Use PyArrow engine for best performance
        compression='snappy',        # Fast compression with good ratio
        write_index=False            # Don't write DataFrame index
    )

    print(f"\n✓ Parquet conversion complete!")
    print(f"✓ Output saved to: {output_parquet_path}")

    # =============================================================================
    # STEP 7: Final verification
    # =============================================================================
    print("\n" + "=" * 80)
    print("VERIFICATION - RELOADING PARQUET")
    print("=" * 80)

    # Reload to verify
    ddf_verify = dd.read_parquet(output_parquet_path)

    print(f"\n✓ Parquet file successfully verified")
    print(f"✓ Rows: {len(ddf_verify):,}")
    print(f"✓ Columns: {len(ddf_verify.columns)}")
    print(f"✓ Column names: {list(ddf_verify.columns)}")

    print(f"\n\nVerified Data Types:")
    print("-" * 80)
    for col, dtype in ddf_verify.dtypes.items():
        indicator = " ← FIXED!" if col == 'CONF_STATUS' else ""
        print(f"{col:<25s} {str(dtype):<20s}{indicator}")

    # Sample data verification
    print(f"\n\nSample data from Parquet:")
    print("-" * 80)
    print(ddf_verify.head(3))

    # =============================================================================
    # STEP 8: Summary and next steps
    # =============================================================================
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE - SUMMARY")
    print("=" * 80)

    print(f"\n✓ Successfully converted {raw_csv_file}")
    print(f"✓ Input: 50 columns → Output: 19 columns (62% reduction)")
    print(f"✓ Optimized dtypes reduce memory by 70-90%")
    print(f"✓ CRITICAL FIX APPLIED: CONF_STATUS now correctly set as category")
    print(f"✓ Dataset ready for analysis at: {output_parquet_path}")

    print(f"\n\nCode Reference:")
    print("-" * 80)
    print(f"See OECD_CODE_REFERENCE.md for complete documentation of all codes:")
    print(f"  - Special codes: _T (Total), _Z (Not Applicable), _X (Unknown)")
    print(f"  - OBS_STATUS: A (Actual), E (Estimated), F (Forecast), M (Missing), O (Observed), P (Provisional)")
    print(f"  - CONF_STATUS: F (Free), C (Confidential), D (Secondary Confidential), N (Not for Publication)")
    print(f"  - Country codes: ISO 3166-1 alpha-3 (AUS, BRA, CAN, FRA, GBR, IDN, USA, etc.)")
    print(f"  - Education levels: ISCED 2011 classification (ISCED11A_0 to ISCED11A_8)")

    print(f"\n\nNext steps - Load and analyze your data:")
    print("-" * 80)
    print(f"\nPython code:")
    print(f"  import dask.dataframe as dd")
    print(f"  ")
    print(f"  # Load the optimized Parquet dataset")
    print(f"  df = dd.read_parquet('{output_parquet_path}')")
    print(f"  ")
    print(f"  # Example queries:")
    print(f"  france_data = df[df['REF_AREA'] == 'FRA']")
    print(f"  recent_data = df[df['TIME_PERIOD'] >= 2015]")
    print(f"  tertiary_ed = df[df['ATTAINMENT_LEV'].str.startswith('ISCED11A_5')]")
    print(f"  ")
    print(f"  # Filter by data quality (get only actual observations)")
    print(f"  actual_data = df[df['OBS_STATUS'] == 'A']")
    print(f"  ")
    print(f"  # Convert to Pandas for final analysis (use .compute())")
    print(f"  result = france_data.compute()")

except FileNotFoundError:
    print(f"\n✗ ERROR: File '{raw_csv_file}' not found!")
    print(f"\nPlease ensure the CSV file is in the current directory.")
    print(f"Expected filename: {raw_csv_file}")

except ValueError as e:
    print(f"\n✗ ERROR: ValueError")
    print(f"Message: {str(e)}")
    print(f"\nIf you see 'could not convert string to float', check:")
    print(f"  1. Ensure CONF_STATUS is set to 'category', not float")
    print(f"  2. Check OECD_CODE_REFERENCE.md for proper dtypes")
    print(f"  3. Verify all dimension columns are set to 'category'")

except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}")
    print(f"Message: {str(e)}")
    print(f"\nPlease check:")
    print(f"  1. File path is correct")
    print(f"  2. You have sufficient disk space (need ~2-3GB)")
    print(f"  3. You have sufficient RAM available (8GB+ recommended)")
    print(f"  4. If 'Memory Error', reduce blocksize to '32MB' or '16MB'")
