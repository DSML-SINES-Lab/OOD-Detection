import os
import pandas as pd
import matplotlib.pyplot as plt

# Dictionary
thresholds = {
    ("ALEXNET", "CrossPlusKL"):               (0.0588, 0.0854),
    ("ALEXNET", "CrossDividesKL"):            (0.5298, 0.9521),
    ("ALEXNET", "CrossPlusCrossDividesKL"):   (0.4623, 0.9006),
    ("BPS", "CrossPlusKL"):                   (0.0935, 0.1603),
    ("BPS", "CrossDividesKL"):                (0.5025, 0.9117),
    ("BPS", "CrossPlusCrossDividesKL"):       (0.4576, 0.8503),
    ("LENET", "CrossPlusKL"):                 (0.0731, 0.1184),
    ("LENET", "CrossDividesKL"):              (0.9671, 0.9855),
    ("LENET", "CrossPlusCrossDividesKL"):     (0.4550, 0.8903),
    ("SCNN", "CrossPlusKL"):                  (0.0850, 0.1500),
    ("SCNN", "CrossDividesKL"):               (0.2936, 0.5506),
    ("SCNN", "CrossPlusCrossDividesKL"):      (0.2516, 0.5543),
    ("SCNN-LSTM", "CrossPlusKL"):             (0.0947, 0.1638),
    ("SCNN-LSTM", "CrossDividesKL"):          (0.3189, 0.6737),
    ("SCNN-LSTM", "CrossPlusCrossDividesKL"): (0.3077, 0.6187),
}


loss_title_mapping = {
    "CrossPlusKL": "Cross + KLDivergence",
    "CrossDividesKL": "Cross / KLDivergence",
    "CrossPlusCrossDividesKL": "Cross + Cross / KLDivergence"
}

models = ["ALEXNET", "BPS", "LENET", "SCNN", "SCNN-LSTM"]
loss_funcs = ["CrossPlusKL", "CrossDividesKL", "CrossPlusCrossDividesKL"]

actual_days = [1, 2, 3, 6, 7, 8, 13, 14, 15, 16, 17, 18, 20, 21]
# For the x-axis labels, we label them sequentially: Day 1, Day 2, ... Day 14.
x_labels = [f"Day {i}" for i in range(1, len(actual_days) + 1)]

# Path to results that contains maximum probabilities
RESULTS_DIR = "/content/drive/MyDrive/Traffic_Analysis/Results"

graphs_dir = os.path.join(RESULTS_DIR, "Graphs")
os.makedirs(graphs_dir, exist_ok=True)


for model in models:
    for loss in loss_funcs:
        if (model, loss) not in thresholds:
            print(f"No thresholds found for {model} - {loss}")
            continue
        
        percentile_threshold, roc_threshold = thresholds[(model, loss)]
        
        correct_seen_percentile = []
        correct_seen_roc = []
        total_seen = []
        
        correct_unseen_percentile = []
        correct_unseen_roc = []
        total_unseen = []
        
        for d in actual_days:
            seen_filename = f"results_{model}_{loss}_Seen_Day{d}.csv"
            seen_path = os.path.join(RESULTS_DIR, seen_filename)
            if not os.path.exists(seen_path):
                correct_seen_percentile.append(0)
                correct_seen_roc.append(0)
                total_seen.append(0)
                continue
            
            df_seen = pd.read_csv(seen_path)
            if "Max Probability" not in df_seen.columns:
                print(f"Max Probability column not found in {seen_path}")
                correct_seen_percentile.append(0)
                correct_seen_roc.append(0)
                total_seen.append(0)
                continue
            
            max_probs = df_seen["Max Probability"].values
            total = len(max_probs)
            total_seen.append(total)
            c_seen_percentile = sum(max_probs >= percentile_threshold)
            c_seen_roc = sum(max_probs >= roc_threshold)
            
            correct_seen_percentile.append(c_seen_percentile)
            correct_seen_roc.append(c_seen_roc)

        for d in actual_days:
            unseen_filename = f"results_{model}_{loss}_Unseen_Day{d}.csv"
            unseen_path = os.path.join(RESULTS_DIR, unseen_filename)
            if not os.path.exists(unseen_path):
                correct_unseen_percentile.append(0)
                correct_unseen_roc.append(0)
                total_unseen.append(0)
                continue
            
            df_unseen = pd.read_csv(unseen_path)
            if "Max Probability" not in df_unseen.columns:
                print(f"Max Probability column not found in {unseen_path}")
                correct_unseen_percentile.append(0)
                correct_unseen_roc.append(0)
                total_unseen.append(0)
                continue
            
            max_probs = df_unseen["Max Probability"].values
            total = len(max_probs)
            total_unseen.append(total)
            # An unseen sample is correctly identified if its max probability < threshold.
            c_unseen_percentile = sum(max_probs < percentile_threshold)
            c_unseen_roc = sum(max_probs < roc_threshold)
            
            correct_unseen_percentile.append(c_unseen_percentile)
            correct_unseen_roc.append(c_unseen_roc)
        
        
        fig, axs = plt.subplots(2, 2, figsize=(12 , 10))
        
        overall_title = f"{model} \u2014 {loss_title_mapping.get(loss, loss)}"
        fig.suptitle(overall_title, fontsize=16)
        
        
        # Top-left: Seen data using Percentile threshold.
        axs[0, 0].bar(x_labels, correct_seen_percentile, facecolor='white', edgecolor='black', hatch='///')
        axs[0, 0].set_title(f"Seen Data (Percentile Threshold = {percentile_threshold:.4f})", fontsize=14)
        axs[0, 0].set_ylabel("Correctly Identify ID Samples", fontsize=15)
        axs[0, 0].set_xlabel("Days", fontsize=15)
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # Top-right: Seen data using ROC threshold.
        axs[0, 1].bar(x_labels, correct_seen_roc, facecolor='white', edgecolor='black', hatch='\\\\')
        axs[0, 1].set_title(f"Seen Data (ROC Threshold = {roc_threshold:.4f})", fontsize=14)
        axs[0, 1].set_ylabel("Correctly Identify ID Samples", fontsize=15)
        axs[0, 1].set_xlabel("Days", fontsize=15)
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # Bottom-left: Unseen data using Percentile threshold.
        axs[1, 0].bar(x_labels, correct_unseen_percentile, facecolor='white', edgecolor='black', hatch='**')
        axs[1, 0].set_title(f"Unseen Data (Percentile Threshold = {percentile_threshold:.4f})", fontsize=15)
        axs[1, 0].set_ylabel("Correctly Identify OOD Samples", fontsize=15)
        axs[1, 0].set_xlabel("Days", fontsize=15)
        axs[1, 0].tick_params(axis='x', rotation=45)
        
        # Bottom-right: Unseen data using ROC threshold.
        axs[1, 1].bar(x_labels, correct_unseen_roc, facecolor='white', edgecolor='black', hatch='xxx')
        axs[1, 1].set_title(f"Unseen Data (ROC Threshold = {roc_threshold:.4f})", fontsize=14)
        axs[1, 1].set_ylabel("Correctly Identify OOD Samples", fontsize=15)
        axs[1, 1].set_xlabel("Days", fontsize=15)
        axs[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        combined_filename = f"{model}_{loss}_Combined.png"
        save_path = os.path.join(graphs_dir, combined_filename)
        plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close(fig)
