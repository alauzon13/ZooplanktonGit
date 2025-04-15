"""Functions to split TIFs into individual vignettes"""

import pandas as pd
import os
import cv2

def extract_vignettes(data_path, source_dir, output_dir, extracted_particles_path):
    # Read the list of particles and their specifications
    data_cleaned = pd.read_csv(data_path)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a list to store the file names and classes
    extracted_particles = []
    
    # Create empty ParticleID column if not already present
    if 'ParticleID' not in data_cleaned.columns:
        data_cleaned['ParticleID'] = ""
    
    # Iterate through the list of particles and extract vignettes
    for index, row in data_cleaned.iterrows():
        file_name = row["Image.File"]
        file_path = os.path.join(source_dir, file_name)
        
        # Generate the vignette file name and path
        vignette_name = f"{os.path.splitext(file_name)[0]}_vign{index:06d}.png"
        vignette_path = os.path.join(output_dir, vignette_name)
        
        if os.path.exists(vignette_path):
            print(f"Vignette already exists: {vignette_name}")
            continue
        
        if os.path.exists(file_path):
            image = cv2.imread(file_path)
            
            # Extract the vignette based on the specified coordinates and dimensions
            x, y, h, w = int(row["Image.X"]), int(row["Image.Y"]), int(row["Image.Height"]), int(row["Image.Width"])
            vignette = image[y:y+h, x:x+w]
            
            # Create a unique particle ID
            particle_id = f"{os.path.splitext(file_name)[0]}_particle_{index}"
            
            # Save the vignette
            cv2.imwrite(vignette_path, vignette)
            
            # Store the particle ID, file name, and class in the list
            extracted_particles.append({"ParticleID": particle_id, "Vignette": vignette_name, "Class": row["Class"]})
            
            # Add ParticleID to the original data row
            data_cleaned.at[index, 'ParticleID'] = particle_id
            
            print(f"Extracted vignette: {vignette_name}")
        else:
            print(f"File not found: {file_path}")
    
    # Save extracted particles DataFrame
    pd.DataFrame(extracted_particles).to_csv(extracted_particles_path, index=False)
    
    # Save updated data_cleaned DataFrame
    data_cleaned.to_csv(data_path, index=False)
    
    print("Finished extracting vignettes and storing classes.")

if __name__ == "__main__":
    extract_vignettes("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/cleaned_merged.csv", 
                      "/Users/adelelauzon/Desktop/MSc/STA5243/tifs", 
                      "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes", 
                      "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/extracted_particles.csv")
