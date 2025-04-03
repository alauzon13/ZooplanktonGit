import sys
import os
import pandas as pd

# Import relevant functions from other modules
sys.path.append(os.path.abspath(".."))
from Data.TextData.Subsample import sample_data, process_and_save
from Data.TextData.MergeData import merge_data



base_dir = "/Users/adelelauzon/Desktop/MSc/STA5243"
output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/MiniExample"


master = pd.read_excel("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/MasterTable_AI_FlowCAM.xlsx", sheet_name="MasterTable")
subsample = sample_data(master, n=20, seed=1013)
subsample = process_and_save(subsample, base_dir=base_dir, output_dir=output_dir, output_file="subsample.csv")

merged_data = merge_data(base_dir, output_dir, subsample, master)

