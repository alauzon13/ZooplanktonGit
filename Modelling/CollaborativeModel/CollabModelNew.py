"""Script for creating Collaborative Model"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Input, concatenate, Dense
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt

# === Helper Functions ===
def match_particle_counts(image_df, text_df):
    """
    Ensures image and text datasets have the same particle counts by downsampling
    to the minimum count per ParticleID.
    
    Parameters:
        image_df (pd.DataFrame): DataFrame containing image data.
        text_df (pd.DataFrame): DataFrame containing text data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Downsampled image and text datasets.
    """

    # # Find the count of each ParticleID in both datasets
    # image_counts = image_df['ParticleID'].value_counts()
    # text_counts = text_df['ParticleID'].value_counts()

    # # Get the minimum count for each ParticleID across both datasets
    # min_counts = pd.concat([image_counts, text_counts], axis=1).min(axis=1)

    # def downsample(df, min_counts):
    #     return (
    #         df.groupby('ParticleID')
    #         .apply(lambda x: x.sample(n=int(min_counts[x.name]), random_state=42), include_groups=False)
    #         .reset_index(drop=True)
    #     )

    # # Apply downsampling to match counts in both datasets
    # image_df_final = downsample(image_df, min_counts)
    # text_df_final = downsample(text_df, min_counts)

    # return image_df_final, text_df_final

     # Identify common ParticleIDs
    common_particle_ids = set(image_df['ParticleID']) & set(text_df['ParticleID'])
    
    # Compute the minimum count per ParticleID
    image_counts = image_df['ParticleID'].value_counts()
    text_counts = text_df['ParticleID'].value_counts()
    
    min_counts = {pid: min(image_counts[pid], text_counts[pid]) for pid in common_particle_ids}
    
    # Downsample function
    def downsample(df, min_counts):
        sampled_data = []
        for pid, count in min_counts.items():
            sampled_data.append(df[df['ParticleID'] == pid].sample(n=count, random_state=42))
        return pd.concat(sampled_data).reset_index(drop=True)
    
    # Apply downsampling
    image_df_final = downsample(image_df, min_counts)
    text_df_final = downsample(text_df, min_counts)
    
    return image_df_final, text_df_final



def add_filepath(df, to_modify, filepath):
    df[to_modify] = df[to_modify].apply(lambda x: f"{filepath}/{x}").astype(str)
    return(df)

def map_str_to_int(train_df):
    # Map string labels to integer indices
    label_to_index = {label: index for index, label in enumerate(train_df['Class'].unique())}
    return(label_to_index)

def process_image_df(label_to_index, df, to_modify, filepath):
    """Add filepath and convert string classes to integers"""
    df = add_filepath(df, to_modify, filepath)
    df['Class'] = df['Class'].map(label_to_index)
    return(df)

def create_tf_dataset(df):
    """
    Creates tensorflow datasset. 
    """
    return tf.data.Dataset.from_tensor_slices((df["Vignette"].values, df["Class"].values))

@tf.autograph.experimental.do_not_convert
def load_and_preprocess_image(image_path, label):
    image_size = (300, 300)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Adjust for PNG if needed
    image = tf.image.resize(image, image_size)
    return image, label

# One hot encoding 
@tf.autograph.experimental.do_not_convert
def input_preprocess(image, label, num_classes):
    label = tf.one_hot(label, num_classes)
    return image, label

def extract_image_data(df):
    ImagesX = []
    LabelsY = []

    for images, labels in df:
        ImagesX.append(images.numpy())
        LabelsY.append(labels.numpy()) 
    
    ImagesX = np.concatenate(ImagesX, axis=0)
    LabelsY = np.concatenate(LabelsY, axis=0)
    
    return ImagesX, LabelsY

def remove_cols(df):
    """
    Removes unecessary cols. 
    """
    new_df = df.drop(columns=["tifffile", "ParticleID"])
    return(new_df)

def get_attr_labels(df):
    """
    Gets attributes and labels from df
    """
    class_columns = [col for col in df.columns if col.startswith('class_')]
    # Extract feature columns (excluding class columns)
    feature_columns = [col for col in df.columns if not col.startswith('class_')]

    AttrX = df[feature_columns].to_numpy()
    Y = df[class_columns].to_numpy()

    return((AttrX, Y))

def create_collab_model(mlp_path, cnn_path, num_classes=6):
    # Load the original models
    mlp_model = tf.keras.models.load_model(mlp_path, custom_objects=None, compile=True, safe_mode=True)
    cnn_model = tf.keras.models.load_model(cnn_path, custom_objects=None, compile=True, safe_mode=True)

    # Remove the last layer (softmax) from both models
    mlp_model = keras.Model(inputs=mlp_model.inputs, outputs=mlp_model.layers[-2].output)
    cnn_model = keras.Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)

    # Define new inputs with the correct shapes
    mlp_input_shape = mlp_model.input_shape[1:]  # Exclude the batch size dimension
    cnn_input_shape = cnn_model.input_shape[1:]  # Exclude the batch size dimension

    mlp_input = Input(shape=mlp_input_shape)
    cnn_input = Input(shape=cnn_input_shape)

    # Get the outputs from the modified models
    mlp_output = mlp_model(mlp_input)
    cnn_output = cnn_model(cnn_input)

    # Concatenate the outputs
    combinedInput = concatenate([mlp_output, cnn_output])

    # Add a fully connected layer with 512 neurons
    fc_layer = Dense(512, activation='relu')(combinedInput)   

    # Add a softmax layer to form the final output
    output_layer = Dense(num_classes, activation='softmax')(fc_layer)  # Adjust the number of classes as needed

    # Create the collaborative model
    collaborative_model = Model(inputs=[mlp_input, cnn_input], outputs=output_layer)

    # Compile the model
    collaborative_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return(collaborative_model)




def collab_plot_hist(hist, fig_output_dir):
    """
    Plots the training and validation accuracy and saves the plot as an image.
    """
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(fig_output_dir)  # Save the plot as an image
    plt.close()  # Close the plot to avoid displaying it

def train_and_plot_collab(model, output_dir, trainAttrX, trainImagesX, trainY, valAttrX, valImagesX, valY, num_epochs):
    """
    Train, plot, and save the final collaborative model. 
    """
    # Train the model
    collaborative_model_history = model.fit(
        [trainAttrX, trainImagesX], 
        trainY, 
        validation_data=([valAttrX, valImagesX], valY), 
        epochs=num_epochs, 
        batch_size=64
    )

    fig_output_dir = os.path.join(output_dir, "collab_accuracy.png")
    # Plot the training history
    collab_plot_hist(collaborative_model_history, fig_output_dir)

    # Save the trained model
    model_output_dir = os.path.join(output_dir, "final_collab_model.keras")
    model.save(model_output_dir)


# === Main Data Preparation Function ===
def prepare_data(image_paths, text_paths, vignette_path, num_classes=6):
    """
    Outputs trainAttrX, trainImagesX, trainY, valAttrX, valImagesX, valY, testAttrX, testImagesX, testY.
    """

    # Load datasets
    image_train, image_val, image_test = [pd.read_csv(p) for p in image_paths]
    text_train, text_val, text_test = [pd.read_csv(p) for p in text_paths]

    # Match particle counts
    image_train, text_train = match_particle_counts(image_train, text_train)
    image_val, text_val = match_particle_counts(image_val, text_val)
    image_test, text_test = match_particle_counts(image_test, text_test)

    # Remove cols 
    text_train, text_val, text_test = map(remove_cols, [text_train, text_val, text_test])


    # Map labels to integers
    label_to_index = map_str_to_int(image_train)
    num_classes = len(label_to_index)

    image_train_final, image_test_final, image_val_final = [
        process_image_df(label_to_index, df, to_modify="Vignette", 
                        filepath=vignette_path) 
                        for df in (image_train, image_test, image_val)]
    
    # Convert image DataFrame to TensorFlow Dataset
    train_ds, val_ds, test_ds = map(create_tf_dataset, [image_train_final, image_val_final, image_test_final])
   
    # train_ds = train_ds.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # val_ds = val_ds.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # test_ds = test_ds.map(load_and_preprocess_image).map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = train_ds.map(lambda img_path, lbl: load_and_preprocess_image(img_path, lbl),
                        num_parallel_calls=tf.data.AUTOTUNE).map(lambda img, lbl: input_preprocess(img, lbl, num_classes),
                                                                  num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(lambda img_path, lbl: load_and_preprocess_image(img_path, lbl),
                        num_parallel_calls=tf.data.AUTOTUNE).map(lambda img, lbl: input_preprocess(img, lbl, num_classes),
                                                                num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(lambda img_path, lbl: load_and_preprocess_image(img_path, lbl),
                        num_parallel_calls=tf.data.AUTOTUNE).map(lambda img, lbl: input_preprocess(img, lbl, num_classes),
                                                                num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)

    trainImagesX, trainLabelsY = extract_image_data(train_ds)
    valImagesX, valLabelsY = extract_image_data(val_ds)
    testImagesX, testLabelsY = extract_image_data(test_ds)

    (trainAttrX, trainY), (testAttrX, testY), (valAttrX, valY) = map(get_attr_labels, [text_train, text_test, text_val])

    return (num_classes, trainAttrX, trainImagesX, trainY, valAttrX, valImagesX, valY, testAttrX, testImagesX, testY)


# === Main Model Training Function ===
def train_collaborative_model(mlp_path, cnn_path, output_dir, trainAttrX, trainImagesX, trainY, valAttrX, valImagesX, valY, num_epochs=25, num_classes=6):
    """
    Loads the MLP and CNN models, creates a collaborative model, trains it, and saves results.
    """
    # Create collaborative model
    collaborative_model = create_collab_model(mlp_path, cnn_path, num_classes)

    # Train and save model
    train_and_plot_collab(collaborative_model, output_dir, trainAttrX, trainImagesX, trainY, valAttrX, valImagesX, valY, num_epochs)







if __name__=="__main__":
    # 1. Process Image Data
    image_size = (300, 300)
    batch_size = 64

    # Paths
    image_paths = [
        "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_train.csv",
        "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_val.csv",
        "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_test.csv"
    ]

    text_paths = [
        "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_train.csv",
        "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_val.csv",
        "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/text_test.csv"
    ]

    vignette_path = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes/"
    output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CollaborativeModel"
    mlp_path = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/final_mlp_model.keras"
    cnn_path = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/final_model.keras"
    metrics_output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CollaborativeModel/performance_metrics.txt"



    # Prepare data
    num_classes, trainAttrX, trainImagesX, trainY, valAttrX, valImagesX, valY, testAttrX, testImagesX, testY = prepare_data(image_paths, text_paths, vignette_path, num_classes=6)



    # Train and save model
    train_collaborative_model(mlp_path, cnn_path, output_dir, trainAttrX, trainImagesX, trainY, valAttrX, valImagesX, valY)


