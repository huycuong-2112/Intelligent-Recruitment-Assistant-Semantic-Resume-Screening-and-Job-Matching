import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Define paths based on your project structure
project_root = Path(__file__).resolve().parent.parent.parent
METRICS_CSV_PATH = project_root / "Data" / "Results" / "IT" / "EvaluationReports" / "jd_001" / "development_metrics.csv"
OUTPUT_IMAGE_PATH = project_root / "Data" / "Results" / "IT" / "EvaluationReports" / "jd_001" / "ranking_consistency_dual_axis.png"

def plot_dual_axis_chart():
    if not METRICS_CSV_PATH.exists():
        print(f"❌ Error: Could not find {METRICS_CSV_PATH}")
        print("Please run run_metrics.py first.")
        return

    # 2. Load the metrics data
    df = pd.read_csv(METRICS_CSV_PATH)
    
    # 3. Ensure the order of models from baseline to advanced
    method_order = ['baseline_tfidf', 'baseline_cos_sim', 'mdms_gridsearch']
    df['method'] = pd.Categorical(df['method'], categories=method_order, ordered=True)
    df = df.sort_values('method').reset_index(drop=True)
    
    display_names = ['TF-IDF', 'Cosine Sim', 'MDMS Gridsearch']
    
    # 4. Extract metrics
    spearman = df['spearman'].values
    mae = df['mae_0_3'].values
    
    # 5. Set up the figure and first axis (No grid)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 6. Plot Spearman Correlation on primary Y-axis (Left)
    color_spearman = '#c44e52' # Reddish
    line1 = ax1.plot(display_names, spearman, marker='o', markersize=10, 
                     linewidth=3, color=color_spearman, label='Spearman')
    
    ax1.set_ylabel('Spearman Correlation', fontsize=12, fontweight='bold', color=color_spearman)
    ax1.tick_params(axis='y', labelcolor=color_spearman)
    
    # Adjust Y-axis to give empty space at the top left for the legend
    ax1.set_ylim(0.5, 1.1)
    
    # 7. Create a secondary Y-axis sharing the same X-axis
    ax2 = ax1.twinx()
    
    # 8. Plot MAE on secondary Y-axis (Right)
    color_mae = '#8172b3' # Purpleish
    line2 = ax2.plot(display_names, mae, marker='s', markersize=10, 
                     linewidth=3, color=color_mae, label='MAE')
    
    ax2.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold', color=color_mae)
    ax2.tick_params(axis='y', labelcolor=color_mae)
    
    # Adjust secondary Y-axis correspondingly
    ax2.set_ylim(0.6, 1.05)
    
    # 9. Clean up visual clutter
    ax1.grid(False)
    ax2.grid(False)
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.title('Overall Ranking Consistency vs Error', fontsize=14, fontweight='bold')
    
    # 10. Combine legends and place them borderless in the top left
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=False)
    
    
    for i, txt in enumerate(spearman):
        # Increased vertical offset to 15
        ax1.annotate(f"{txt:.3f}", (display_names[i], spearman[i]), 
                     xytext=(0, -20), textcoords='offset points', 
                     ha='center', color=color_spearman, fontweight='bold'
                     )
                     
    for i, txt in enumerate(mae):
        # Increased vertical offset to -20
        ax2.annotate(f"{txt:.3f}", (display_names[i], mae[i]), 
                     xytext=(0, 15), textcoords='offset points', 
                     ha='center', color=color_mae, fontweight='bold'
                     )
    
    plt.tight_layout()
    
    # 12. Save the chart
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=150)
    print(f"✅ Chart saved successfully to:\n{OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    plot_dual_axis_chart()