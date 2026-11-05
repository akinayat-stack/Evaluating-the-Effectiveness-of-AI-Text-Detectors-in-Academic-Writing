"""Visualization script for TASK 3 (Scientific Plots)."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
palette = sns.color_palette("colorblind")

def plot_f1_distribution(df):
    df = df.reset_index(drop=True)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="detector", y="f1", palette="colorblind", capsize=.1, errorbar=None)
    for bar, hatch in zip(ax.patches, ["", "//", "xx"]):
        bar.set_hatch(hatch)
    
    for i, row in df.iterrows():
        plt.errorbar(x=i, y=row["f1"], 
                     yerr=[[row["f1"] - row["f1_ci95_lo"]], [row["f1_ci95_hi"] - row["f1"]]], 
                     fmt='none', c='black', capsize=5)
    
    plt.xlabel("Detector Type", fontsize=12)
    plt.ylabel("F1 Score (dimensionless)", fontsize=12)
    plt.title("F1 Score with 95% Bootstrap CI by Detector", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(FIG_DIR / "fig2_distribution_boxplot.pdf")
    plt.savefig(FIG_DIR / "fig2_distribution_boxplot.png", dpi=300)
    plt.close()

def plot_latency_vs_load(df_lat):
    plt.figure(figsize=(8, 5))
    plt.yscale('log')
    
    detectors = df_lat["detector"].unique()
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    for i, det in enumerate(detectors):
        sub = df_lat[df_lat["detector"] == det].sort_values("batch_size")
        plt.plot(sub["batch_size"], sub["median_ms"], 
                 label=det, marker=markers[i % 4], linestyle=linestyles[i % 4], color=palette[i % len(palette)])
        
        plt.fill_between(sub["batch_size"], sub["ci95_lo_ms"], sub["ci95_hi_ms"], alpha=0.2, color=palette[i % len(palette)])
    
    plt.xscale('log', base=2)
    plt.xticks(df_lat["batch_size"].unique(), df_lat["batch_size"].unique())
    
    plt.xlabel("Batch Size (documents/request)", fontsize=12)
    plt.ylabel("Median Latency (ms)", fontsize=12)
    plt.title("Latency vs Load with 95% CI (warm-up filtered)", fontsize=14)
    plt.legend(title="Detector")
    plt.tight_layout()
    
    plt.savefig(FIG_DIR / "fig1_latency_vs_load.pdf")
    plt.savefig(FIG_DIR / "fig1_latency_vs_load.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    metrics_df = pd.read_csv("data/processed/metrics_summary.csv")
    lat_df = pd.read_csv("data/processed/latency_summary.csv")
    
    metrics_df = metrics_df[metrics_df["detector"] != "random_baseline"]
    lat_df = lat_df[lat_df["detector"] != "random_baseline"]

    plot_f1_distribution(metrics_df)
    plot_latency_vs_load(lat_df)
    
    print(f"Графики успешно сохранены в {FIG_DIR}!")