"""
This script builds and trains a Multi-Layer Perceptron (MLP) for text classification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import tensorflow as tf 
from tensorflow import keras
from keras import layers
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad



# --- Data Loading and Preprocessing ---
def load_and_preprocess(filename, base_path):
    """Loads CSV and removes unnecessary columns."""
    df = pd.read_csv(os.path.join(base_path, filename))
    return df.drop(columns=["tifffile", "ParticleID"])

def plot_history(history, fig_output_dir):
    """Plots training accuracy and saves as an image."""
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(fig_output_dir)
    plt.close()

def build_mlp(input_shape, num_classes, hidden_layers=3, neurons_per_layer=512, dropout_rate=0.2):
    """Builds a configurable MLP model."""
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    for _ in range(hidden_layers):
        model.add(layers.Dense(neurons_per_layer, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))  # Add dropout here

    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def perform_grid_search(X_train, y_train, X_val, y_val, input_shape, num_classes, fig_output_dir, num_epochs=25):
    """Performs grid search over hidden layers and neuron configurations."""
    best_accuracy, best_model, best_params = 0, None, None

    
    for hidden_layers, neurons_per_layer in product([1, 2, 3, 4, 5], [256, 512, 1024, 2048]):
        print(f"Training: {hidden_layers} layers, {neurons_per_layer} neurons")
        model = build_mlp(input_shape, num_classes, hidden_layers, neurons_per_layer)
        history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=0)
        
        val_acc = max(history.history["val_accuracy"])
        if val_acc > best_accuracy:
            best_accuracy, best_model, best_params = val_acc, model, (hidden_layers, neurons_per_layer)
    
    print(f"Best Model: {best_params} with Accuracy: {best_accuracy:.2f}")
    return best_model

def mlp_evaluate_model(X_train, y_train, X_test, y_test, X_val, y_val, input_shape, num_classes, model_output_dir, metrics_output_dir, fig_output_dir, num_epochs):
    """Evaluates the best model on the test set and saves performance metrics."""
    
    best_model = perform_grid_search(X_train, y_train, X_val, y_val, input_shape, num_classes, fig_output_dir, num_epochs=25)
    
    best_model.save(model_output_dir)

    
    history = best_model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=0)
        
    plot_history(history, fig_output_dir)

    test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.2f}')

    y_pred_probs = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Save performance metrics
    with open(metrics_output_dir, 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")

if __name__ == "__main__":
  
    base_path = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Data/TextData/"
    model_output_dir = '/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/final_mlp_model.keras'
    fig_output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/model_accuracy.png"
    metrics_output_dir = "/Users/adelelauzon/Desktop/MSc/STA5243/2453Github/Modelling/MLP/performance_metrics.txt"

    # Load datasets
    text_train, text_val, text_test = [load_and_preprocess(file, base_path) for file in 
                                       ["text_train.csv", "text_val.csv", "text_test.csv"]]


    # Feature and label extraction
    feature_columns = [col for col in text_train.columns if not col.startswith('class_')]
    class_columns = [col for col in text_train.columns if col.startswith('class_')]

    X_train, y_train = text_train[feature_columns].to_numpy(), text_train[class_columns].to_numpy()
    X_val, y_val = text_val[feature_columns].to_numpy(), text_val[class_columns].to_numpy()
    X_test, y_test = text_test[feature_columns].to_numpy(), text_test[class_columns].to_numpy()

    input_shape, num_classes = X_train.shape[1], len(class_columns)

  
    # Evaluate model
    mlp_evaluate_model(X_train, y_train, X_test, y_test, X_val, y_val, input_shape, num_classes, model_output_dir, metrics_output_dir, fig_output_dir, num_epochs=25)
