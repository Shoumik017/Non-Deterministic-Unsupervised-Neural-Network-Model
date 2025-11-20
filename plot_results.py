import os
import csv
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.txt")
PLOT_FILE = os.path.join(RESULTS_DIR, "metrics_bar.png")

def read_metrics(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            
            rows.append({
                "Model": r["Model"],
                "ARI": float(r["ARI"]),
                "NMI": float(r["NMI"]),
                "Silhouette": float(r["Silhouette"]),
            })
    return rows

def make_bar_plot(rows, out_path):
   
    labels = [r["Model"] for r in rows]
    ari = [r["ARI"] for r in rows]
    nmi = [r["NMI"] for r in rows]
    sil = [r["Silhouette"] for r in rows]

    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar([i - width for i in x], ari, width, label="ARI")
    plt.bar(x, nmi, width, label="NMI")
    plt.bar([i + width for i in x], sil, width, label="Silhouette")

    plt.xticks(list(x), labels)
    plt.ylabel("Score")
    plt.title("Clustering Metrics: Baseline vs S-DEC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved bar chart to {out_path}")

def print_table(rows):
    
    print("\n=== Metrics Summary ===")
    print("{:<10}  {:>8}  {:>8}  {:>11}".format("Model", "ARI", "NMI", "Silhouette"))
    for r in rows:
        print("{:<10}  {:>8.4f}  {:>8.4f}  {:>11.4f}".format(r["Model"], r["ARI"], r["NMI"], r["Silhouette"]))

def main():
    if not os.path.exists(METRICS_FILE):
        raise FileNotFoundError(f"Couldn't find {METRICS_FILE}. Run sdec_train.py first.")
    rows = read_metrics(METRICS_FILE)
    print_table(rows)
    make_bar_plot(rows, PLOT_FILE)

if __name__ == "__main__":
    main()