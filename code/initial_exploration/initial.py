from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
client_csv_path = PROJECT_ROOT / 'data' / 'results' / 'initial_exploration' / '20_samples_summary.csv'

def run_sample_grouped_sensor_audit(csv_file_path: str):
    print("=" * 75)
    print(f"STARTING ADVANCED DATA AUDIT FOR: {csv_file_path}")
    print("=" * 75)
    
    # Load dataset
    df = pd.read_csv(csv_file_path)
    n_rows, n_cols = df.shape
    print(f"\nDataset Dimensions: {n_rows:,} rows × {n_cols:,} columns")

    # Verify sample_id column exists
    if 'sample_id' not in df.columns:
        raise KeyError("[CRITICAL ERROR] 'sample_id' column was not found in the CSV file!")
        
    unique_samples = sorted(df['sample_id'].dropna().unique())
    print(f"Sample Count: {len(unique_samples)} unique sample_id scenarios found ({unique_samples[0]} to {unique_samples[-1]})\n")

    # =========================================================================
    # 1. EXACT CELL-LEVEL DATA TYPE AUDIT
    # =========================================================================
    print("--- 1. EXACT CELL-LEVEL TYPE AUDIT ---")
    non_numeric_cells = []
    
    for col in df.columns:
        if col == 'sample_id':
            continue
        # Coerce column to numeric to locate exact non-convertible strings/corruptions
        converted = pd.to_numeric(df[col], errors='coerce')
        # Find indices where raw data wasn't null, but numeric conversion produced NaN
        invalid_mask = df[col].notna() & converted.isna()
        if invalid_mask.any():
            invalid_indices = df.index[invalid_mask].tolist()
            for idx in invalid_indices:
                non_numeric_cells.append((idx, col, df.at[idx, col]))

    if non_numeric_cells:
        print(f"[WARNING] Found {len(non_numeric_cells)} non-numeric/string values in the dataset:")
        for row_idx, col_name, val in non_numeric_cells[:15]: # Display first 15
            print(f"  • Row {row_idx}, Column '{col_name}' -> Value: '{val}' (Type: {type(val).__name__})")
        if len(non_numeric_cells) > 15:
            print(f"  ... and {len(non_numeric_cells) - 15} more cells.")
    else:
        print("[PASSED] 100% of sensor cells contain valid numeric or null data. No unexpected strings found.")

    # =========================================================================
    # 2. DETAILED MISSING VALUE AUDIT
    # =========================================================================
    print("\n--- 2. DETAILED MISSING VALUE AUDIT ---")
    
    total_nulls = df.isnull().sum().sum()
    if total_nulls == 0:
        print("[PASSED] Zero missing values found anywhere in the dataset.")
    else:
        print(f"[WARNING] Found {total_nulls:,} missing cells total.")
        
        # Missing per column
        null_cols = df.columns[df.isnull().any()].tolist()
        print(f"\nColumns with missing values ({len(null_cols)}):")
        for c in null_cols[:10]:
            cnt = df[c].isnull().sum()
            print(f"  • Column '{c}': {cnt:,} missing ({cnt/n_rows*100:.2f}%)")
            
        # Missing per row
        null_rows = df.index[df.isnull().any(axis=1)].tolist()
        print(f"\nRows with missing values ({len(null_rows)}):")
        print(f"  • Row indices with missing data: {null_rows[:15]} {'...' if len(null_rows) > 15 else ''}")

    # =========================================================================
    # 3. EXACT DUPLICATE IDENTIFICATION
    # =========================================================================
    print("\n--- 3. DUPLICATE AUDIT ---")
    
    # Row duplicates
    dup_row_indices = df.index[df.duplicated()].tolist()
    if dup_row_indices:
        print(f"[WARNING] Found {len(dup_row_indices)} duplicate rows.")
        print(f"  • Duplicate row indices: {dup_row_indices[:15]}")
    else:
        print("[PASSED] No duplicate rows found.")

    # Column duplicates (Exact matching vectors)
    print("\nScanning for duplicate columns...")
    dup_cols_map = {}
    cols = [c for c in df.columns if c != 'sample_id']
    
    for i in range(len(cols)):
        col_i = cols[i]
        if col_i in dup_cols_map:
            continue
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            if df[col_i].equals(df[col_j]):
                dup_cols_map[col_j] = col_i

    if dup_cols_map:
        print(f"[WARNING] Found {len(dup_cols_map)} DUPLICATE column(s):")
        for dup_col, orig_col in dup_cols_map.items():
            print(f"  • Column '{dup_col}' is an exact duplicate of original column '{orig_col}'")
    else:
        print("[PASSED] No duplicate columns found.")

    # =========================================================================
    # 4. CLIENT-READY DISTRIBUTIONS FOR THE 20 SAMPLE SCENARIOS
    # =========================================================================
    print("\n--- 4. CLIENT-FACING DISTRIBUTION METRICS (20 SCENARIOS) ---")
    
    # Categorize sensor channels
    oes_cols = [c for c in df.columns if c.startswith('OES_')]
    v_cols = [c for c in df.columns if c.startswith('V_t_')]
    i_cols = [c for c in df.columns if c.startswith('I_t_')]
    ir_cols = [c for c in df.columns if c.startswith('IR_pix_')]

    sample_metrics = []

    for s_id in unique_samples:
        s_df = df[df['sample_id'] == s_id]
        
        # Calculate statistical summaries per domain for this scenario
        def calc_domain_stats(column_list):
            if not column_list:
                return (0, 0, 0, 0, 0)
            sub = s_df[column_list].apply(pd.to_numeric, errors='coerce').values
            return (
                round(np.mean(sub), 3),
                round(np.std(sub), 3),
                round(np.min(sub), 3),
                round(np.max(sub), 3),
                round((sub == 0).sum() / sub.size * 100, 1) # % Zeroes
            )

        oes_mean, oes_std, oes_min, oes_max, oes_zero = calc_domain_stats(oes_cols)
        v_mean, v_std, v_min, v_max, _ = calc_domain_stats(v_cols)
        i_mean, i_std, i_min, i_max, _ = calc_domain_stats(i_cols)
        ir_mean, ir_std, ir_min, ir_max, _ = calc_domain_stats(ir_cols)

        sample_metrics.append({
            "Sample ID": s_id,
            "Rows": len(s_df),
            "OES Mean": oes_mean,
            "OES Max": oes_max,
            "OES Zero %": oes_zero,
            "Voltage Mean": v_mean,
            "Voltage Std": v_std,
            "Current Mean": i_mean,
            "Current Std": i_std,
            "IR Temp Mean": ir_mean,
            "IR Temp Max": ir_max,
            "IR Temp Std": ir_std
        })

    client_report_df = pd.DataFrame(sample_metrics)
    
    print("\nExecutive Summary Table Across 20 Sample Scenarios:")
    print(client_report_df.to_string(index=False))

    # Export client-ready per-sample summary
    client_report_df.to_csv(client_csv_path, index=False)
    print(f"\n[INFO] Per-sample distribution metrics exported to: '{client_csv_path}'")

    print("\n" + "=" * 75)
    print("AUDIT COMPLETE")
    print("=" * 75)
    
    return client_report_df


# Calling the function:
if __name__ == "__main__":
    measure_path = PROJECT_ROOT / 'data' / 'measurement' / 'Measurement_Data.csv'
    stats_summary = run_sample_grouped_sensor_audit(measure_path)
