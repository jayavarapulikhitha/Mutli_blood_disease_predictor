# ml_dl_project.py
import os
import zipfile
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from tqdm import tqdm

def main():
    # --- Step 1: Data Acquisition and Preparation ---
    print("Step 1: Acquiring and preparing the blood cell dataset...")

    # New, more stable URL for a blood cell classification dataset with a simple folder structure.
    dataset_url = "https://github.com/paiml/yolo-blood-cell-detection/raw/main/BCCD.zip"
    data_dir = "blood_cells_dataset"
    archive_file = "BCCD.zip"
    
    # Download and extract the dataset if it doesn't exist
    if not os.path.exists(data_dir):
        print(f"Downloading dataset from {dataset_url}...")
        
        response = requests.get(dataset_url)
        
        if response.status_code == 200:
            with open(archive_file, "wb") as file:
                file.write(response.content)
            
            print("Dataset downloaded successfully.")
            print("Extracting files...")
            
            try:
                with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                    # This dataset has a single top-level folder 'BCCD'
                    zip_ref.extractall(data_dir)
                os.remove(archive_file)
                print("Dataset extracted successfully.")
            except zipfile.BadZipFile as e:
                print(f"Error: The downloaded file is not a valid zip file. Error: {e}")
                return
        else:
            print(f"Error: Failed to download dataset. Status code: {response.status_code}")
            return
    
    # Corrected paths for the new dataset structure.
    # The new dataset has a very simple folder structure: 'BCCD/train' and 'BCCD/test'
    train_data_path = os.path.join(data_dir, "BCCD", "train")
    test_data_path = os.path.join(data_dir, "BCCD", "test")

    # Get list of image paths and labels from the data directory
    def load_image_paths_and_labels(base_path):
        image_paths = []
        labels = []
        if not os.path.exists(base_path):
            print(f"Error: Directory not found at {base_path}. Please check the extracted folder structure.")
            return [], []
        
        for label in os.listdir(base_path):
            label_path = os.path.join(base_path, label)
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    image_paths.append(os.path.join(label_path, filename))
                    labels.append(label)
        return image_paths, labels

    train_image_paths, train_labels = load_image_paths_and_labels(train_data_path)
    test_image_paths, test_labels = load_image_paths_and_labels(test_data_path)

    if not train_image_paths or not test_image_paths:
        print("Error: No images found in the train or test directories. Exiting.")
        return

    # Create a DataFrame for easier handling
    train_df = pd.DataFrame({'path': train_image_paths, 'label': train_labels})
    test_df = pd.DataFrame({'path': test_image_paths, 'label': test_labels})

    # Label encoding for classification
    label_map = {label: i for i, label in enumerate(np.unique(train_labels))}
    train_df['label_encoded'] = train_df['label'].map(label_map)
    test_df['label_encoded'] = test_df['label'].map(label_map)

    # --- Step 2: Deep Learning for Feature Extraction ---
    print("Step 2: Using a pre-trained VGG16 model for feature extraction...")

    # Load the VGG16 model with pre-trained ImageNet weights.
    feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the feature extractor so they are not retrained.
    for layer in feature_extractor.layers:
        layer.trainable = False

    def extract_features(dataframe):
        features = []
        for path in tqdm(dataframe['path'], desc="Extracting features"):
            try:
                img = load_img(path, target_size=(224, 224))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                feature = feature_extractor.predict(img, verbose=0)
                features.append(feature.flatten())
            except Exception as e:
                print(f"Warning: Could not process image at {path}. Error: {e}")
        return np.array(features)

    # Extract features for both training and testing sets
    X_train_features = extract_features(train_df)
    y_train_labels = train_df['label_encoded']
    X_test_features = extract_features(test_df)
    y_test_labels = test_df['label_encoded']

    print("Features extracted for training and testing sets.")

    # --- Step 3: Traditional Machine Learning for Classification ---
    print("Step 3: Training a RandomForestClassifier on the extracted features...")

    # Initialize and train the RandomForestClassifier model
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_features, y_train_labels)

    print("Classifier training complete.")

    # --- Step 4: Evaluation and Saving ---
    print("Step 4: Evaluating the combined model...")

    # Make predictions on the test set
    y_pred = classifier.predict(X_test_features)

    # Print evaluation metrics
    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred, target_names=label_map.keys()))

    # Save both the feature extractor and the classifier for later use
    with open('vgg16_feature_extractor.pkl', 'wb') as f:
        pickle.dump(feature_extractor, f)
    with open('random_forest_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    print("Combined models saved successfully.")

if _name_ == "_main_":
    main()