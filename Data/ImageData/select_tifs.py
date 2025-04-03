import pandas as pd
import os
import shutil

def copy_tif_files(source_dir, output_dir, data_cleaned_path):
    # Read the list of TIFF files from data_cleaned.csv
    data_cleaned = pd.read_csv(data_cleaned_path)
    tifs = data_cleaned["Image.File"]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the list of TIFF files and copy each file
    for tif in tifs:
        source_path = os.path.join(source_dir, tif)
        destination_path = os.path.join(output_dir, tif)

        # Check if the file already exists in the destination folder
        if not os.path.exists(destination_path):  # Only copy if the file isn't already in the destination folder
            # Check if the file exists in the source directory before copying
            if os.path.exists(source_path):
                try:
                    shutil.copy(source_path, destination_path)  # Use copy instead of move
                except Exception as e:
                    print(f"Error copying {tif}: {e}")
            else:
                print(f"File not found: {source_path}")

    print("Finished copying TIFF files.")

# Only run the following code if this script is executed directly (not imported)
if __name__ == "__main__":
    source_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/HURON_OverlapTiffsWithPP"
    output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/tifs"
    data_cleaned_path = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/cleaned_merged.csv"

    copy_tif_files(source_dir, output_dir, data_cleaned_path)
