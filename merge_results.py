import os
import glob
import csv
import statistics
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"
PATTERN = os.path.join(RESULTS_DIR, "metrics_seed_*.txt")
OUT_CSV = os.path.join(RESULTS_DIR, "metrics_mean_std.csv")
OUT_PLOT = os.path.join(RESULTS_DIR, "metrics_bar_err.png")

def read_seed_files(pattern):
    data = {} 
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}. Make sure you ran the loop and saved metrics per seed.")
    for path in files:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["Model"]
                metrics = {k: float(row[k]) for k in ["ARI", "NMI", "Silhouette"]}
                data.setdefault(model, []).append(metrics)
    return data, files

def mean_std(values):
    mu = statistics.mean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mu, sd

def aggregate(data):

    agg = {}
    for model, rows in data.items():
        ari_vals = [r["ARI"] for r in rows]
        nmi_vals = [r["NMI"] for r in rows]
        sil_vals = [r["Silhouette"] for r in rows]
        agg[model] = {
            "ARI": mean_std(ari_vals),
            "NMI": mean_std(nmi_vals),
            "Silhouette": mean_std(sil_vals),
        }
    return agg

def save_csv(agg, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Metric", "Mean", "StdDev"])
        for model, metrics in agg.items():
            for metric, (mu, sd) in metrics.items():
                w.writerow([model, metric, mu, sd])
    print(f"Saved aggregate CSV to {out_csv}")

def plot_bar_with_error(agg, out_png):
    models = list(agg.keys())
    metrics = ["ARI", "NMI", "Silhouette"]
    x = range(len(models))
    width = 0.25

    vals = []
    errs = []
    for metric in metrics:
        vals.append([agg[m][metric][0] for m in models])
        errs.append([agg[m][metric][1] for m in models])

    plt.figure(figsize=(8, 5))

    import numpy as np
    x = np.arange(len(models))
    offsets = [-width, 0, width]
    for i, metric in enumerate(metrics):
        plt.bar(x + offsets[i], vals[i], width, yerr=errs[i], capsize=4, label=metric)
    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Clustering Metrics (mean ± std across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved bar chart with error bars to {out_png}")

def print_table(agg, files):
    print("\nAggregated over files:")
    for p in files:
        print(" -", p)
    print("\n=== Mean ± Std (across seeds) ===")
    for model, metrics in agg.items():
        line = [model]
        for metric in ["ARI", "NMI", "Silhouette"]:
            mu, sd = metrics[metric]
            line.append(f"{metric}: {mu:.4f} ± {sd:.4f}")
        print(" | ".join(line))

def main():
    data, files = read_seed_files(PATTERN)
    agg = aggregate(data)
    print_table(agg, files)
    save_csv(agg, OUT_CSV)
    plot_bar_with_error(agg, OUT_PLOT)

if __name__ == "__main__":
    main()