import pandas as pd
import random
import os


def sample_data(df, n, seed=1013):
    """Samples n unique TIFF files from the master table."""
    random.seed(seed)
    df_to_sample = df[~df["tifffile"].str.contains("Simc")]
    
    return df_to_sample.sample(n)

def determine_odlocation(csvfile, base_dir):
    """Determines the OD location for a given CSV file."""
    if not csvfile.endswith(".csv"):
        csvfile += ".csv"
    return os.path.join(base_dir, "CSVs", csvfile)

def process_and_save(df, base_dir, output_dir, output_file="subsample.csv"):
    """Adds ODLocation column and saves the sampled data to a specified directory."""
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    df["ODLocation"] = df["csvfile"].apply(lambda x: determine_odlocation(x, base_dir))
    output_path = os.path.join(output_dir, output_file)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    base_dir = "/Users/adelelauzon/Desktop/MSc/STA5243"
    output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData"

    df = pd.read_excel("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/MasterTable_AI_FlowCAM.xlsx", sheet_name="MasterTable")

    subsample = sample_data(df, n=50, seed=1013)
    subsample = process_and_save(subsample, base_dir=base_dir, output_dir=output_dir, output_file="subsample.csv")












