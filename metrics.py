import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# -------------------------------
# 1) Threshold dictionary (already computed for each model–loss)
# -------------------------------
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

# Models, loss functions, days
models = ["ALEXNET", "BPS", "LENET", "SCNN", "SCNN-LSTM"]
loss_funcs = ["CrossPlusKL", "CrossDividesKL", "CrossPlusCrossDividesKL"]
actual_days = [1, 2, 3, 6, 7, 8, 13, 14, 15, 16, 17, 18, 20, 21]

# Path to the CSV results
RESULTS_DIR = "/content/drive/MyDrive/Traffic_Analysis/Results"

# -------------------------------
# 2) Helper function to compute metrics
# -------------------------------
def compute_metrics(y_true, y_pred):
    """
    Given arrays of true labels (0 or 1) and predicted labels (0 or 1),
    compute accuracy, TPR, FPR, precision, recall, and F1 score.
    """
    # Confusion matrix with labels=[1, 0] ensures row=truth, col=prediction
    # cm[0,0] = TP, cm[0,1] = FN, cm[1,0] = FP, cm[1,1] = TN
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    TP = cm[0,0]
    FN = cm[0,1]
    FP = cm[1,0]
    TN = cm[1,1]
    
    total = TP + FN + FP + TN
    accuracy = (TP + TN) / total if total else 0.0
    
    # TPR = recall = TP / (TP + FN)
    TPR = TP / (TP + FN) if (TP + FN) else 0.0
    # FPR = FP / (FP + TN)
    FPR = FP / (FP + TN) if (FP + TN) else 0.0
    
    # Precision = TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TPR  # same as above
    
    if precision + recall == 0:
        F1 = 0.0
    else:
        F1 = 2 * precision * recall / (precision + recall)
    
    return accuracy, TPR, FPR, precision, recall, F1

# -------------------------------
# 3) Main loop: For each model–loss, gather all data and compute metrics
# -------------------------------
for model in models:
    for loss in loss_funcs:
        # Check if thresholds exist
        if (model, loss) not in thresholds:
            print(f"Skipping {model} - {loss}: No thresholds found.")
            continue
        
        percentile_threshold, roc_threshold = thresholds[(model, loss)]
        
        # Arrays to hold all data (across all days) for this model–loss
        # We'll store them in a single array for 'seen' and 'unseen'
        #   y_true: 1 for seen, 0 for unseen
        #   y_prob: max probability
        y_true_all = []
        y_prob_all = []
        
        # Loop over days
        for d in actual_days:
            # (a) Seen data
            seen_filename = f"results_{model}_{loss}_Seen_Day{d}.csv"
            seen_path = os.path.join(RESULTS_DIR, seen_filename)
            if os.path.exists(seen_path):
                df_seen = pd.read_csv(seen_path)
                if "Max Probability" in df_seen.columns:
                    max_probs = df_seen["Max Probability"].values
                    # Append to the arrays
                    y_true_all.extend([1]*len(max_probs))  # 1 for seen
                    y_prob_all.extend(max_probs)
            
            # (b) Unseen data
            unseen_filename = f"results_{model}_{loss}_Unseen_Day{d}.csv"
            unseen_path = os.path.join(RESULTS_DIR, unseen_filename)
            if os.path.exists(unseen_path):
                df_unseen = pd.read_csv(unseen_path)
                if "Max Probability" in df_unseen.columns:
                    max_probs = df_unseen["Max Probability"].values
                    # Append
                    y_true_all.extend([0]*len(max_probs))  # 0 for unseen
                    y_prob_all.extend(max_probs)
        
        y_true_all = np.array(y_true_all)
        y_prob_all = np.array(y_prob_all)
        
        if len(y_true_all) == 0:
            print(f"No data found for {model} - {loss}.")
            continue
        
        # -------------------------------
        # 4) Convert probabilities to predicted labels using each threshold
        # -------------------------------
        # (a) Percentile threshold
        y_pred_percentile = np.where(y_prob_all >= percentile_threshold, 1, 0)
        # (b) ROC threshold
        y_pred_roc = np.where(y_prob_all >= roc_threshold, 1, 0)
        
        # -------------------------------
        # 5) Compute metrics
        # -------------------------------
        acc_p, tpr_p, fpr_p, prec_p, rec_p, f1_p = compute_metrics(y_true_all, y_pred_percentile)
        acc_r, tpr_r, fpr_r, prec_r, rec_r, f1_r = compute_metrics(y_true_all, y_pred_roc)
        
        print(f"\nModel: {model}, Loss: {loss}")
        print(f"  -> Data samples: {len(y_true_all)} (Seen + Unseen combined)")
        
        print("  Percentile Threshold Metrics:")
        print(f"    Accuracy:  {acc_p:.4f}")
        print(f"    TPR:       {tpr_p:.4f}")
        print(f"    FPR:       {fpr_p:.4f}")
        print(f"    Precision: {prec_p:.4f}")
        print(f"    Recall:    {rec_p:.4f}")
        print(f"    F1 Score:  {f1_p:.4f}")
        
        print("  ROC Threshold Metrics:")
        print(f"    Accuracy:  {acc_r:.4f}")
        print(f"    TPR:       {tpr_r:.4f}")
        print(f"    FPR:       {fpr_r:.4f}")
        print(f"    Precision: {prec_r:.4f}")
        print(f"    Recall:    {rec_r:.4f}")
        print(f"    F1 Score:  {f1_r:.4f}")
