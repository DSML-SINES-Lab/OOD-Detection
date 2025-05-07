import pandas as pd
import os

input_file = "/content/drive/MyDrive/Traffic_Analysis/DataSet/5000bucket_tcpBPSfile_combinetill1nov23.csv" 
file_name = "5000bucket_tcpBPSfile_combinetill1nov23"
df = pd.read_csv(input_file, delimiter=",", header=None)


class_column = df.columns[-1]

# Define the classes to separate
classes_to_separate = ["link1"]+[f"link{i}" for i in range(111, 123)]


output_dir = "/content/drive/MyDrive/Traffic_Analysis/Unseen_Data/"
dataset_dir = "/content/drive/MyDrive/Traffic_Analysis/DataSet/"
os.makedirs(output_dir, exist_ok=True)

remaining_df = df.copy()

for class_name in classes_to_separate:
    class_df = df[df[class_column] == class_name]
    if not class_df.empty:
        output_file = os.path.join(output_dir, file_name + f"_{class_name}.csv")
        class_df.to_csv(output_file, index=False, header=False, encoding="ascii", sep=",")
        print(f"Saved: {output_file}")
        
        remaining_df = remaining_df[remaining_df[class_column] != class_name]
        
        
cleaned_file = os.path.join(dataset_dir, "5000bucket_tcpBPSfile_combinetill1nov23_updated.csv")
remaining_df.to_csv(cleaned_file, index=False, header=False, encoding="ascii", sep=",")

print("Separation completed!")
