import os
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import roc_curve

# -------------------------------
# Define the three custom loss functions
# -------------------------------
def custom_loss_function_CrossPlusKL(y_true, y_pred):
    cce = CategoricalCrossentropy()
    kl = KLDivergence()
    cross = cce(y_true, y_pred)
    test = tf.fill(tf.shape(y_true), 1/30)
    kl_d = kl(test, y_pred)
    loss = cross + kl_d
    return loss

def custom_loss_function_CrossDividesKL(y_true, y_pred):
    cce = CategoricalCrossentropy()
    kl = KLDivergence()
    cross = cce(y_true, y_pred)
    test = tf.fill(tf.shape(y_true), 1/30)
    kl_d = kl(test, y_pred)
    loss = cross / kl_d
    return loss

def custom_loss_function_CrossPlusCrossDividesKL(y_true, y_pred):
    cce = CategoricalCrossentropy()
    kl = KLDivergence()
    cross = cce(y_true, y_pred)
    test = tf.fill(tf.shape(y_true), 1/30)
    kl_d = kl(test, y_pred)
    loss = cross + cross / kl_d
    return loss

# Map loss function names to the corresponding functions
loss_function_mapping = {
    "CrossPlusKL": custom_loss_function_CrossPlusKL,
    "CrossDividesKL": custom_loss_function_CrossDividesKL,
    "CrossPlusCrossDividesKL": custom_loss_function_CrossPlusCrossDividesKL
}

# -------------------------------
# Configuration: Base directories and file patterns
# -------------------------------
MODELS_BASE_DIR = "/content/drive/MyDrive/Traffic_Analysis/Models/With_loss"
SEEN_DATA_DIR = "/content/drive/MyDrive/Traffic_Analysis/Seen_Data"
UNSEEN_DATA_DIR = "/content/drive/MyDrive/Traffic_Analysis/Unseen_Data"
RESULTS_DIR = "/content/drive/MyDrive/Traffic_Analysis/Results"

# Make sure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of loss functions and model names (folder names)
loss_functions = ["CrossPlusKL", "CrossDividesKL", "CrossPlusCrossDividesKL"]
model_names = ["ALEXNET", "BPS", "LENET", "SCNN", "SCNN-LSTM"]

# Percentile value for threshold (for seen data)
percentile_value = 5

# Dictionary to store thresholds: keys = (model, loss), values = (percentile_threshold, roc_threshold)
threshold_results = {}

# -------------------------------
# Helper functions to parse 'day' from filenames/folders
# -------------------------------
def extract_day_from_seen_filename(filename):
    """
    Extracts the day number from a seen data filename of the form:
      5000bucket_tcpBPSfile_3nov23.csv
    Returns the string '3' for day 3, or None if no match.
    """
    base = os.path.basename(filename).lower()
    match = re.search(r'5000bucket_tcpbpsfile_(\d+)nov23\.csv', base)
    if match:
        return match.group(1)
    return None

def extract_day_from_unseen_folder(foldername):
    """
    Extracts the day number from an unseen folder name of the form '3NOV'.
    Returns '3' if foldername is '3NOV' or '3nov', else None.
    """
    match = re.search(r'(\d+)', foldername.lower())
    if match:
        return match.group(1)
    return None

# -------------------------------
# Loop over each model folder and loss function combination
# -------------------------------
for model_name in model_names:
    for loss_func in loss_functions:
        print(f"\nProcessing Model: {model_name}, Loss: {loss_func}")
        # Expected filenames:
        #   Model file: "{ModelName}_{LossFunc}_FINAL.h5"
        #   Vocabulary file: "vocab_{ModelName}_{LossFunc}.pkl"
        model_filename = f"{model_name}_{loss_func}_FINAL.h5"
        vocab_filename = f"vocab_{model_name}_{loss_func}.pkl"
        model_path = os.path.join(MODELS_BASE_DIR, model_name, model_filename)
        vocab_path = os.path.join(MODELS_BASE_DIR, model_name, vocab_filename)
        
        if not os.path.exists(model_path):
            print(f"  Model file not found: {model_path}")
            continue
        if not os.path.exists(vocab_path):
            print(f"  Vocabulary file not found: {vocab_path}")
            continue
        
        # Load the model using its corresponding custom loss function.
        custom_loss = loss_function_mapping[loss_func]
        model = load_model(model_path, custom_objects={'custom_loss_function': custom_loss})
        with open(vocab_path, 'rb') as f:
            vocabulary, vocabulary_inv, labels = pickle.load(f)
        vocab_size = len(vocabulary_inv)
        print(f"  Loaded model and vocabulary (vocab size: {vocab_size})")
        
        # Prepare to store aggregated max probs for threshold calculation
        all_seen_max_probs = []
        all_unseen_max_probs = []
        
        # -------------------------------
        # 1) Process SEEN data (day by day), save results, collect max probs
        # -------------------------------
        seen_pattern = os.path.join(SEEN_DATA_DIR, "5000bucket_tcpBPSfile_*nov23.csv")
        seen_files = glob.glob(seen_pattern)
        if not seen_files:
            print(f"  No seen data files found with pattern: {seen_pattern}")
            continue
        
        for seen_file in seen_files:
            day_str = extract_day_from_seen_filename(seen_file)
            if not day_str:
                print(f"    Could not parse day from seen file: {seen_file}")
                continue
            # e.g. day_str = '3' -> Day3
            day_label = f"Day{day_str}"
            
            # Read the CSV and predict
            df_seen = pd.read_csv(seen_file)
            X_seen = df_seen.iloc[:, :-1].values.astype(np.int32)
            X_seen[X_seen >= vocab_size] = vocabulary.get("<UNK>", 0)
            probs_seen = model.predict(X_seen)
            seen_max = np.max(probs_seen, axis=1)
            
            # Collect for threshold
            all_seen_max_probs.extend(seen_max.tolist())
            
            # Save results for graph code
            # We'll store just "Sample ID" and "Max Probability"
            out_df = pd.DataFrame({
                "Sample ID": range(1, len(seen_max)+1),
                "Max Probability": seen_max
            })
            
            # Construct output filename, e.g. results_MODEL_LOSS_Seen_Day3.csv
            out_filename = f"results_{model_name}_{loss_func}_Seen_{day_label}.csv"
            out_path = os.path.join(RESULTS_DIR, out_filename)
            out_df.to_csv(out_path, index=False)
            print(f"    Saved seen results for {day_label} -> {out_path}")
        
        # -------------------------------
        # 2) Process UNSEEN data (day folders), save results, collect max probs
        # -------------------------------
        # Unseen data are in subfolders of UNSEEN_DATA_DIR
        unseen_day_folders = [f for f in os.listdir(UNSEEN_DATA_DIR)
                              if os.path.isdir(os.path.join(UNSEEN_DATA_DIR, f))]
        
        for day_folder in unseen_day_folders:
            day_str = extract_day_from_unseen_folder(day_folder)
            if not day_str:
                print(f"    Could not parse day from unseen folder: {day_folder}")
                continue
            day_label = f"Day{day_str}"
            
            day_lower = day_folder.lower()
            unseen_pattern = os.path.join(UNSEEN_DATA_DIR, day_folder,
                                          f"5000bucket_tcpBPSfile_{day_lower}23_link*.csv")
            files_unseen = glob.glob(unseen_pattern)
            if not files_unseen:
                print(f"    No unseen CSV files found in {day_folder} with pattern {unseen_pattern}")
                continue
            
            # We'll collect max probs for *this day* to save in one CSV
            day_unseen_max = []
            
            for uf in files_unseen:
                df_unseen = pd.read_csv(uf)
                X_unseen = df_unseen.iloc[:, :-1].values.astype(np.int32)
                X_unseen[X_unseen >= vocab_size] = vocabulary.get("<UNK>", 0)
                probs_unseen = model.predict(X_unseen)
                max_unseen = np.max(probs_unseen, axis=1)
                day_unseen_max.extend(max_unseen.tolist())
            
            # Add to global aggregator
            all_unseen_max_probs.extend(day_unseen_max)
            
            # Save day-level results for unseen
            out_df = pd.DataFrame({
                "Sample ID": range(1, len(day_unseen_max)+1),
                "Max Probability": day_unseen_max
            })
            out_filename = f"results_{model_name}_{loss_func}_Unseen_{day_label}.csv"
            out_path = os.path.join(RESULTS_DIR, out_filename)
            out_df.to_csv(out_path, index=False)
            print(f"    Saved unseen results for {day_label} -> {out_path}")
        
        # Convert to numpy arrays for threshold calculation
        all_seen_max_probs = np.array(all_seen_max_probs)
        all_unseen_max_probs = np.array(all_unseen_max_probs)
        print(f"  Processed seen data: {len(all_seen_max_probs)} samples")
        print(f"  Processed unseen data: {len(all_unseen_max_probs)} samples")
        
        if len(all_seen_max_probs) == 0 or len(all_unseen_max_probs) == 0:
            print("  Skipping threshold calculation (no seen/unseen samples).")
            continue
        
        # -------------------------------
        # 3) Compute thresholds
        # -------------------------------
        # Method 1: Percentile-based threshold (using seen data)
        percentile_threshold = np.percentile(all_seen_max_probs, percentile_value)
        
        # Method 2: ROC analysis threshold (combine seen and unseen predictions)
        y_true = np.concatenate([
            np.ones(len(all_seen_max_probs)),
            np.zeros(len(all_unseen_max_probs))
        ])
        y_scores = np.concatenate([all_seen_max_probs, all_unseen_max_probs])
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        diff = np.abs(tpr - (1 - fpr))
        optimal_idx = np.argmin(diff)
        roc_threshold = roc_thresholds[optimal_idx]
        
        # Save thresholds for this model-loss combination
        threshold_results[(model_name, loss_func)] = (percentile_threshold, roc_threshold)
        
        print(f"  {model_name} with {loss_func}:")
        print(f"    Percentile ({percentile_value}th) Threshold: {percentile_threshold:.4f}")
        print(f"    ROC-based Threshold: {roc_threshold:.4f}")

# -------------------------------
# 4) Print Summary of Thresholds
# -------------------------------
print("\nFinal Threshold Summary (Model, Loss -> (Percentile, ROC)):")
for key, (perc_thresh, roc_thresh) in threshold_results.items():
    print(f"  Model: {key[0]}, Loss: {key[1]} -> Percentile: {perc_thresh:.4f}, ROC: {roc_thresh:.4f}")
