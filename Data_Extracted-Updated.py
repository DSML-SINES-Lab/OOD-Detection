import pandas as pd
import os
import re

# Define input and output directories
input_dir = "/content/drive/MyDrive/Traffic_Analysis/BPSnew/"
output_base_dir = "/content/drive/MyDrive/Traffic_Analysis/Unseen_Data/"

# Define the classes to separate
classes_to_separate = ["link1"] + [f"link{i}" for i in range(111, 123)]

# Loop over all CSV files in the input directory
for file in os.listdir(input_dir):
    if file.endswith(".csv") and file.startswith("5000bucket_tcpBPSfile_"):
        # Use regex to extract the day and month from the file name.
        # Expected pattern: 5000bucket_tcpBPSfile_<day><month>23.csv (e.g. 5000bucket_tcpBPSfile_1nov23.csv)
        match = re.match(r"5000bucket_tcpBPSfile_(\d+)([a-zA-Z]+)23\.csv", file)
        if match:
            day = match.group(1)         # e.g. "1" or "21"
            month = match.group(2)         # e.g. "nov"
            folder_name = f"{day}{month.capitalize()}"  # e.g. "1Nov" or "21Nov"
        else:
            folder_name = "UnknownDate"
        
        # Create the output directory for this date if it doesn't exist
        output_dir = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        input_file = os.path.join(input_dir, file)
        print(f"Processing {input_file} into folder {output_dir}")
        df = pd.read_csv(input_file, delimiter=",", header=None)
        class_column = df.columns[-1]
        remaining_df = df.copy()
        
        # Process each specified class
        for class_name in classes_to_separate:
            class_df = df[df[class_column] == class_name]
            if not class_df.empty:
                # Create an output file name appending the class name before the .csv extension
                output_file = os.path.join(output_dir, file.replace(".csv", f"_{class_name}.csv"))
                class_df.to_csv(output_file, index=False, header=False, encoding="ascii", sep=",")
                print(f"Saved: {output_file}")
                remaining_df = remaining_df[remaining_df[class_column] != class_name]
        
        # Save the remaining data using the original file name in the same date folder
        cleaned_file = os.path.join(output_dir, file)
        remaining_df.to_csv(cleaned_file, index=False, header=False, encoding="ascii", sep=",")
        print(f"Saved cleaned file: {cleaned_file}\n")

print("Separation completed!")
