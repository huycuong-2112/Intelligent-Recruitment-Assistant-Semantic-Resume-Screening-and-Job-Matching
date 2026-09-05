import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Define paths based on your project structure
project_root = Path(__file__).resolve().parent.parent.parent
METRICS_CSV_PATH = project_root / "Data" / "Results" / "IT" / "EvaluationReports" / "jd_001" / "development_metrics.csv"
OUTPUT_IMAGE_PATH = project_root / "Data" / "Results" / "IT" / "EvaluationReports" / "jd_001" / "ranking_performance_bar_chart.png"

def plot_grouped_bar_chart():
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
    
    # Display names for the chart x-axis
    display_names = ['TF-IDF', 'Cosine Sim', 'MDMS Gridsearch']
    
    # 4. Extract metrics
    recall_5 = df['recall_5'].values
    recall_10 = df['recall_10'].values
    ndcg_5 = df['ndcg_5'].values
    ndcg_10 = df['ndcg_10'].values
    
    # 5. Set up the figure and axes (Removed seaborn whitegrid)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(display_names))
    width = 0.2  # The width of the bars
    
    # 6. Plot the bars
    bars1 = ax.bar(x - 1.5*width, recall_5, width, label='Recall@5', color='#8cadd3')
    bars2 = ax.bar(x - 0.5*width, recall_10, width, label='Recall@10', color='#4c72b0')
    bars3 = ax.bar(x + 0.5*width, ndcg_5, width, label='nDCG@5', color='#98d1a6')
    bars4 = ax.bar(x + 1.5*width, ndcg_10, width, label='nDCG@10', color='#55a868')
    
    # 7. Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('Ranking Performance: Recall and nDCG @ K', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=12)
    
    # Expand Y-axis ceiling to give the legend plenty of room so it doesn't overlap
    ax.set_ylim(0, 1.25)
    
    # 8. Clean up visual clutter (No grids, no top/right spines)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Place legend in upper left corner, borderless
    ax.legend(loc='upper left', frameon=False)
    
    # 9. Add value annotations on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    autolabel(bars4)
    
    plt.tight_layout()
    
    # 10. Save the chart
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=150)
    print(f"✅ Chart saved successfully to:\n{OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    plot_grouped_bar_chart()