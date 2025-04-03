import numpy as np
import pandas as pd
import tensorflow as tf  # For tf.data
import matplotlib.pyplot as plt
import keras
from keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3, preprocess_input, decode_predictions
from sklearn.metrics import precision_score, recall_score, f1_score


def preprocess_data(train_df, val_df, test_df, base_path):
    # Define image size and batch size
    image_size = (300, 300)
    batch_size = 64
    num_classes = len(train_df['Class'].unique())
    
    # Add correct filepath in front of image paths and ensure they are strings
    train_df["Vignette"] = train_df["Vignette"].apply(lambda x: f"{base_path}{x}").astype(str)
    val_df["Vignette"] = val_df["Vignette"].apply(lambda x: f"{base_path}{x}").astype(str)
    test_df["Vignette"] = test_df["Vignette"].apply(lambda x: f"{base_path}{x}").astype(str)

    # Map string labels to integer indices
    label_to_index = {label: index for index, label in enumerate(train_df['Class'].unique())}
    train_df['Class'] = train_df['Class'].map(label_to_index)
    val_df['Class'] = val_df['Class'].map(label_to_index)
    test_df['Class'] = test_df['Class'].map(label_to_index)
    
    return train_df, val_df, test_df, num_classes

# Resize images 
@tf.autograph.experimental.do_not_convert
def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, (300, 300))
    return image, label

# One-hot encoding 
@tf.autograph.experimental.do_not_convert
def input_preprocess(image, label, num_classes):
    label = tf.one_hot(label, num_classes)
    return image, label

def create_tf_datasets(train_df, val_df, test_df, num_classes, batch_size=64):
    train_ds = tf.data.Dataset.from_tensor_slices((train_df["Vignette"].values, train_df["Class"].values))
    train_ds = train_ds.map(load_and_preprocess_image).map(lambda img, lbl: input_preprocess(img, lbl, num_classes), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_df["Vignette"].values, val_df["Class"].values))
    val_ds = val_ds.map(load_and_preprocess_image).map(lambda img, lbl: input_preprocess(img, lbl, num_classes), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_df["Vignette"].values, test_df["Class"].values))
    test_ds = test_ds.map(load_and_preprocess_image).map(lambda img, lbl: input_preprocess(img, lbl, num_classes), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def build_model(num_classes):
    inputs = layers.Input(shape=(300, 300, 3))
    model = EfficientNetV2B3(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = keras.Model(inputs, outputs, name="EfficientNet")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def plot_hist(hist, output_dir):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(output_dir)
    plt.close()

def cnn_evaluate_model(model, test_ds, test_df, metrics_output):
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f'Test accuracy: {test_accuracy:.2f}')

    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_df['Class'].values

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    with open(metrics_output, 'w') as f:
        f.write(f'Test accuracy: {test_accuracy:.2f}\n')
        f.write(f'Precision: {precision:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'F1 Score: {f1:.2f}\n')

if __name__ == "__main__":
    train_df = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_train.csv")
    val_df = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_val.csv")
    test_df = pd.read_csv("/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/image_test.csv")
    base_path = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/ImageData/vignettes/"
    fig_output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/model_accuracy.png"
    metrics_output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/performance_metrics.txt"
    model_output_dir = '/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/CNN/final_model.keras'

    train_df, val_df, test_df, num_classes = preprocess_data(train_df, val_df, test_df, base_path)
    train_ds, val_ds, test_ds = create_tf_datasets(train_df, val_df, test_df, num_classes)
    model = build_model(num_classes)
    hist = model.fit(train_ds, epochs=25, validation_data=val_ds)
    plot_hist(hist, fig_output_dir)
    model.save(model_output_dir)
    cnn_evaluate_model(model, test_ds, test_df, metrics_output_dir)
