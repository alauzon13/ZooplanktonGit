"""Splitting image and text data into train/test/validation. Here we also oversample and generate augmented images."""

import os
import numpy as np
import time
import pandas as pd
import json
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


def process_text_data(text_data, seed=42):
    """Process and standardize text data."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    text_target = text_data["Class"]
    text_tifffile = text_data["tifffile"]
    text_particleID = text_data["ParticleID"]
    
    # Remove unimportant columns
    columns_to_drop = ['Biovolume..Sphere.', 'Biovolume..Cylinder.', 'Biovolume..P..Spheroid.', 'EFFSPEED', 'Class.Particle.ID', 'Rep', 'Date', 'Key', 'Image.File', 'Original.Reference.ID', 'Source.Image', 'Time', 'Timestamp', 'csvfile_x', 'csvfile_y', 'Year', 'Month', 'Day', 'Class', 'tifffile', 'ParticleID']
    columns_to_drop = [col for col in columns_to_drop if col in text_data.columns]
    text_features_cleaned = text_data.drop(columns=columns_to_drop)    
    
    # One-hot encoding for features
    categorical_cols = text_features_cleaned.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(text_features_cleaned[categorical_cols])
    one_hot_features_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    # One-hot encoding for class
    one_hot_class_df = pd.get_dummies(text_target, prefix="class")

    # Save decoder dictionary (mapping one-hot columns back to original labels)
    decoder_dict = {col: col.replace("class_", "") for col in one_hot_class_df.columns}
    with open("decoder_dict.json", "w") as f:
        json.dump(decoder_dict, f)

        
    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_columns = list(set(text_features_cleaned.columns) - set(categorical_cols))
    numeric_standardized = pd.DataFrame(scaler.fit_transform(text_features_cleaned[numerical_columns]), columns=numerical_columns)
    
    # Concatenate data
    text_all_cleaned = pd.concat([numeric_standardized.reset_index(drop=True), one_hot_features_df.reset_index(drop=True), one_hot_class_df.reset_index(drop=True)], axis=1)
    text_all_cleaned['tifffile'] = text_tifffile
    text_all_cleaned["ParticleID"] = text_particleID

    # Remove any rows with empty data
    text_all_cleaned = text_all_cleaned.dropna()
    
    return text_all_cleaned

def oversample_classes(df, classes_to_oversample, target_count, seed=42):
    """Oversample specific classes to the target count."""
    oversampled_dfs = []
    for class_name in classes_to_oversample:
        class_df = df[df["Class"] == class_name]
        oversampled_class_df = class_df.sample(target_count, replace=True, random_state=seed)
        oversampled_dfs.append(oversampled_class_df)
    return pd.concat(oversampled_dfs)

def split_data(particles, text_all_cleaned, classes_to_oversample=('Chydoridae', 'Daphnia'), target_count=100):
    """Split the particle and text data into train, validation, and test sets."""
    
    # Oversample the specified classes
    particles_oversampled = oversample_classes(particles, classes_to_oversample, target_count)

    particles_combined = pd.concat([particles, particles_oversampled])
    
    train_val, test = train_test_split(particles_combined, test_size=0.2, stratify=particles_combined['Class'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['Class'], random_state=42)
    
    # Use ParticleID to filter text data
    train_particle_ids = train['ParticleID']
    val_particle_ids = val['ParticleID']
    test_particle_ids = test['ParticleID']
    
    train_text = text_all_cleaned[text_all_cleaned['ParticleID'].isin(train_particle_ids)]
    val_text = text_all_cleaned[text_all_cleaned['ParticleID'].isin(val_particle_ids)]
    test_text = text_all_cleaned[text_all_cleaned['ParticleID'].isin(test_particle_ids)]
    
    return train, val, test, train_text, val_text, test_text

def generate_and_save_augmented_images(df, folder_path, target_size=(128, 128), augmentations=2, class_label=None):
    """Generate and save augmented images."""
    augmented_image_paths = []

    # Instantiate the ImageDataGenerator inside the function
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    for index, row in df.iterrows():
        original_file_name = row["Vignette"]
        original_image_path = os.path.join(folder_path, original_file_name)

        if not os.path.exists(original_image_path):
            print(f"Warning: Image not found - {original_image_path}")
            continue

        original_image = load_img(original_image_path, target_size=target_size)
        original_image_array = img_to_array(original_image)
        original_image_array = np.expand_dims(original_image_array, axis=0)

        before_files = set(os.listdir(folder_path))
        i = 0
        for _ in datagen.flow(original_image_array, batch_size=1, save_to_dir=folder_path, 
                              save_prefix=f"{original_file_name.replace('.png', '')}_aug", save_format='png'):
            i += 1
            if i >= augmentations:
                break

        time.sleep(1)
        after_files = set(os.listdir(folder_path))
        new_files = list(after_files - before_files)

        for new_file in new_files:
            augmented_image_paths.append((new_file, class_label))

        print(f"Generated {len(new_files)} augmented images for {original_file_name}")

    return augmented_image_paths

if __name__ == "__main__":
    particle_path = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/extracted_particles.csv"  
    text_data_path = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/cleaned_merged.csv"  
    particles = pd.read_csv(particle_path)
    text_data = pd.read_csv(text_data_path)
    text_all_cleaned = process_text_data(text_data, seed=42)
    
    # Specify the classes you want to oversample
    classes_to_oversample = ('Chydoridae', 'Daphnia')  
    train_img, val_img, test_img, train_text, val_text, test_text = split_data(particles, text_all_cleaned, classes_to_oversample, target_count=100)

    vignettes_folder = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes"  
    
    # Generate augmented images for each oversampled class
    augmented_images = []
    for class_label in classes_to_oversample:
        train_class = train_img[train_img['Class'] == class_label]
        aug_images = generate_and_save_augmented_images(train_class, vignettes_folder, class_label=class_label)
        augmented_images.extend(aug_images)
    
    augmented_df = pd.DataFrame(augmented_images, columns=["Vignette", "Class"])
    train_img = pd.concat([train_img, augmented_df])
    
    train_img.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_train.csv", index=False)
    val_img.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_val.csv", index=False)
    test_img.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_test.csv", index=False)
    train_text.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_train.csv", index=False)
    val_text.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_al.csv", index=False)
    test_text.to_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_test.csv", index=False)
    print("Data augmentation and saving completed.")





