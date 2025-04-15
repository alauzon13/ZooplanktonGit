"""Functions to merge two sources of plain-text data into a single dataframe"""

import pandas as pd 
import os

def merge_data(base_dir, output_dir, geometric, environmental):
    """
    Merges the geometric and environmental data into one dataset. 
    
    Args: 
        base_dir (str): The base directory where the input CSV files are located.
        output_dir (str): The directory where output files should be saved.
        geometric (DataFrame): Data containing ODLocation references.
        environmental (DataFrame): Environmental data to merge.
    
    Returns:
        DataFrame: The final merged and filtered dataset.
    """
    
    stacked_csvs = pd.DataFrame()

    for csv_location in geometric["ODLocation"]:
        try:
            # Construct the full file path
            full_file_path = os.path.join(base_dir, csv_location)
            # Load the CSV file
            csv_to_add = pd.read_csv(full_file_path)
            # Add new column with the name of the CSV
            name_of_file = os.path.basename(full_file_path)
            csv_to_add["csvfile"] = name_of_file
            stacked_csvs = pd.concat([stacked_csvs, csv_to_add], ignore_index=True)
        except FileNotFoundError:
            print(f"File not found: {full_file_path}")
            continue

    # Indicate that the script has finished searching through the list
    print("Finished searching through the list of CSV files.")

    stacked_csvs.to_csv(os.path.join(output_dir, "stacked_csvs.csv"), index=False)

    stacked_csvs = pd.read_csv(os.path.join(output_dir, "stacked_csvs.csv"))

    # Merge the stacked CSVs with the master DataFrame on a common column (e.g., 'csvfile')
    merged_data = pd.merge(stacked_csvs, environmental, left_on="Image.File", right_on="tifffile", how="left")

    # Save the merged data to a new CSV file
    merged_data.to_csv(os.path.join(output_dir, "merged_data.csv"), index=False)

    # Filter the merged data based on a list of values
    values_to_filter = ["Calanoid_1", "Cylopoid_1", "Bosmina_1", "Herpacticoida", "Chironomid", 
                        "Chydoridae", "Daphnia"]
    merged_data_filtered = merged_data[merged_data["Class"].isin(values_to_filter)]

    # Filter to only include unique rows 
    unique_merged_data_filtered = merged_data_filtered.drop_duplicates()

    # Identify + Remove Constant features
    constant_columns = unique_merged_data_filtered.columns[unique_merged_data_filtered.nunique() == 1]
    data_cleaned = unique_merged_data_filtered.drop(columns=constant_columns)

    # Save the filtered data to a new CSV file
    unique_merged_data_filtered.to_csv(os.path.join(output_dir, "filtered_data_new.csv"), index=False)

    # Save
    data_cleaned_path = os.path.join(output_dir, "cleaned_merged.csv")

    data_cleaned.to_csv(data_cleaned_path, index=False)

    return data_cleaned



if __name__ == "__main__":

    # Define input and output directories
    base_dir = "/Users/adelelauzon/Desktop/MSc/STA5243"
    output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData"

    # Load input datasets
    geometric = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/subsample.csv")  
    environmental = pd.read_excel("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/MasterTable_AI_FlowCAM.xlsx", sheet_name="MasterTable")

    # Run merge function
    merged_data = merge_data(base_dir, output_dir, geometric, environmental)






# subsample = pd.read_csv("subsample.csv")
# master = pd.read_excel("MasterTable_AI_FlowCAM.xlsx", sheet_name="MasterTable")

# # Set the base directory where the CSV files are located
# base_dir = "/Users/adelelauzon/Desktop/MSc/STA5243"
# # Extract necessary CSVs based on subsample
# stacked_csvs = pd.DataFrame()

# for csv_location in subsample["ODLocation"]:
#     try:
#         # Construct the full file path
#         full_file_path = os.path.join(base_dir, csv_location)
#         # Load the CSV file
#         csv_to_add = pd.read_csv(full_file_path)
#         # Add new column with the name of the CSV
#         name_of_file = os.path.basename(full_file_path)
#         csv_to_add["csvfile"] = name_of_file
#         stacked_csvs = pd.concat([stacked_csvs, csv_to_add], ignore_index=True)
#     except FileNotFoundError:
#         print(f"File not found: {full_file_path}")
#         continue

# # Indicate that the script has finished searching through the list
# print("Finished searching through the list of CSV files.")

# #stacked_csvs.to_csv("stacked_csvs.csv", index=False)

# stacked_csvs = pd.read_csv("stacked_csvs.csv")

# # Merge the stacked CSVs with the master DataFrame on a common column (e.g., 'csvfile')
# merged_data = pd.merge(stacked_csvs, master, left_on="Image.File", right_on="tifffile", how="left")

# # Save the merged data to a new CSV file
# merged_data.to_csv("merged_data.csv", index=False)

# # Filter the merged data based on a list of values
# values_to_filter = ["Calanoid_1", "Cylopoid_1", "Bosmina_1", "Herpacticoida", "Chironomid", 
#                     "Chydoridae", "Daphnia"]
# merged_data_filtered = merged_data[merged_data["Class"].isin(values_to_filter)]

# # Filter to only include unique rows 
# unique_merged_data_filtered = merged_data_filtered.drop_duplicates()

# # Save the filtered data to a new CSV file
# unique_merged_data_filtered.to_csv("filtered_data_new.csv", index=False)
