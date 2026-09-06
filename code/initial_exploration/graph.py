from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ir_plots_path = PROJECT_ROOT / 'data' / 'results' / 'initial_exploration' / 'ir_heatmaps.png'
line_plots_path = PROJECT_ROOT / 'data' / 'results' / 'initial_exploration' / 'line_plots.png'

def plot_20_sample_sensor_dashboard(csv_file_path: str):
    """
    Computes 100-row averages per sample_id (0 to 19) and plots:
      1. OES Spectral Intensity curves (1 graph with 20 sample lines)
      2. Voltage Waveform profiles (1 graph with 20 sample lines)
      3. Current Waveform profiles (1 graph with 20 sample lines)
      4. IR Spatial Heatmaps (1 figure with 20 subplots of 32x32 grids)
    """
    print(f"Loading data from: {csv_file_path} ...")
    df = pd.read_csv(csv_file_path)

    if 'sample_id' not in df.columns:
        raise KeyError("Column 'sample_id' not found in dataset!")

    unique_samples = sorted(df['sample_id'].dropna().unique())
    print(f"Detected {len(unique_samples)} unique sample IDs: {unique_samples}")

    # Identify column domain groups
    oes_cols = [c for c in df.columns if c.startswith('OES_')]
    v_cols = [c for c in df.columns if c.startswith('V_t_')]
    i_cols = [c for c in df.columns if c.startswith('I_t_')]
    ir_cols = [c for c in df.columns if c.startswith('IR_pix_')]

    print(f"Found channels: OES={len(oes_cols)}, Voltage={len(v_cols)}, Current={len(i_cols)}, IR={len(ir_cols)}")

    # Pre-calculate 100-row averages grouped by sample_id
    sample_means = {}
    for s_id in unique_samples:
        s_df = df[df['sample_id'] == s_id]
        sample_means[s_id] = {
            'OES': s_df[oes_cols].mean(axis=0).values if oes_cols else None,
            'V': s_df[v_cols].mean(axis=0).values if v_cols else None,
            'I': s_df[i_cols].mean(axis=0).values if i_cols else None,
            'IR': s_df[ir_cols].mean(axis=0).values if ir_cols else None,
        }

    # Generate distinct colors and line styles for readability across 20 samples
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    line_styles = ['-', '--', '-.', ':']
    style_combinations = list(itertools.product(line_styles, colors))
    #print(style_combinations)
    # =========================================================================
    # FIGURE 1: 1D LINE PLOTS (OES, VOLTAGE, CURRENT)
    # =========================================================================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig1, axes = plt.subplots(3, 1, figsize=(16, 18))
    fig1.suptitle("100-Observation Mean Profiles Across 20 Samples", fontsize=16, fontweight='bold')

    # Graph 1: OES Bin Intensities
    if oes_cols:
        x_oes = np.arange(len(oes_cols))
        for idx, s_id in enumerate(unique_samples):
            ls, col = style_combinations[idx % len(style_combinations)]
            axes[0].plot(x_oes, sample_means[s_id]['OES'], label=f"Sample {s_id}", 
                         color=col, linestyle=ls, linewidth=1.5)
        axes[0].set_title("OES Spectral Bin Intensity Average", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("OES Bin Index", fontsize=11)
        axes[0].set_ylabel("Mean Intensity", fontsize=11)
        axes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=2, fontsize=9)

    # Graph 2: Voltage Waveforms
    if v_cols:
        x_v = np.arange(len(v_cols))
        for idx, s_id in enumerate(unique_samples):
            ls, col = style_combinations[idx % len(style_combinations)]
            axes[1].plot(x_v, sample_means[s_id]['V'], label=f"Sample {s_id}", 
                         color=col, linestyle=ls, linewidth=1.5)
        axes[1].set_title("Voltage Waveform (V_t) Average", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Voltage Time-Step Index (t)", fontsize=11)
        axes[1].set_ylabel("Mean Voltage (V)", fontsize=11)
        axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=2, fontsize=9)

    # Graph 3: Current Waveforms
    if i_cols:
        x_i = np.arange(len(i_cols))
        for idx, s_id in enumerate(unique_samples):
            ls, col = style_combinations[idx % len(style_combinations)]
            axes[2].plot(x_i, sample_means[s_id]['I'], label=f"Sample {s_id}", 
                         color=col, linestyle=ls, linewidth=1.5)
        axes[2].set_title("Current Waveform (I_t) Average", fontsize=13, fontweight='bold')
        axes[2].set_xlabel("Current Time-Step Index (t)", fontsize=11)
        axes[2].set_ylabel("Mean Current (A)", fontsize=11)
        axes[2].legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=2, fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig1.savefig(line_plots_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Line plots saved to: '{line_plots_path}'")

    # =========================================================================
    # FIGURE 2: IR 32x32 GRIDDED COLORMAPS (20 SUBPLOTS)
    # =========================================================================
    if ir_cols and len(ir_cols) == 1024:
        fig2, axes_ir = plt.subplots(4, 5, figsize=(18, 14))
        fig2.suptitle("IR Thermal Camera 32×32 Spatial Heatmaps (100-Row Mean per Sample)", 
                      fontsize=16, fontweight='bold')
        
        # Determine global min/max for colorbar consistency across all 20 samples
        all_ir_means = np.array([sample_means[s_id]['IR'] for s_id in unique_samples if sample_means[s_id]['IR'] is not None])
        vmin, vmax = all_ir_means.min(), all_ir_means.max()

        for idx, s_id in enumerate(unique_samples):
            row, col = divmod(idx, 5)
            ax = axes_ir[row, col]
            
            # Reshape 1024 1D array into 32x32 matrix
            ir_grid = sample_means[s_id]['IR'].reshape((32, 32))
            
            im = ax.imshow(ir_grid, cmap='inferno', vmin=vmin, vmax=vmax, aspect='equal')
            ax.set_title(f"Sample {s_id}", fontsize=11, fontweight='bold')
            ax.axis('off')  # Hide axis ticks for visual clarity

        # Add shared colorbar
        fig2.subplots_adjust(right=0.90)
        cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
        fig2.colorbar(im, cax=cbar_ax, label='Mean IR Intensity / Temperature')

        fig2.savefig(ir_plots_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"IR heatmaps saved to: '{ir_plots_path}'")
    else:
        print("[WARNING] Skipping IR plot: requires exactly 1024 IR columns for 32x32 reshaping.")

#Calling the function:
if __name__ == "__main__":
    measure_path = PROJECT_ROOT / 'data' / 'measurement' / 'Measurement_Data.csv'
    plot_20_sample_sensor_dashboard(measure_path)
