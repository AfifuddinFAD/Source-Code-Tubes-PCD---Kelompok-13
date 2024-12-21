import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import matplotlib.pyplot as plt

# Load and preprocess dataset
def load_dataset(image_folder):
    images = []
    labels = []
    label_mapping = {}

    for label, subfolder in enumerate(os.listdir(image_folder)):
        label_mapping[label] = subfolder
        subfolder_path = os.path.join(image_folder, subfolder)
        for image_file in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (128, 128))  # Resize to uniform size
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels), label_mapping

# Extract HOG features
def extract_hog_features(images):
    features = []
    for image in images:
        hog_feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')
        features.append(hog_feature)
    return np.array(features)

# Visualize search results
def visualize_results(query_image, result_images):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, len(result_images) + 1, 1)
    plt.imshow(query_image, cmap='gray')
    plt.title("Query")
    for i, result in enumerate(result_images):
        plt.subplot(1, len(result_images) + 1, i + 2)
        plt.imshow(result, cmap='gray')
        plt.title(f"Result {i+1}")
    plt.show()

# Main workflow
if __name__ == "__main__":
    dataset_folder = "C:\Users\USER\Downloads\Source Code Tubes PCD - Kelompok 13\DeepFashion-MultiModal-main\DeepFashion-MultiModal-main\assets"

    print("Loading dataset...")
    try:
        images, labels, label_mapping = load_dataset(dataset_folder)
    except FileNotFoundError:
        print("Dataset folder not found. Please check the path.")
        exit()

    print("Extracting HOG features...")
    features = extract_hog_features(images)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print("Training K-NN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    print("Evaluating classifier...")
    accuracy = knn.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    query_index = 0
    query_image = images[query_index]
    query_feature = features[query_index].reshape(1, -1)

    print("Searching for similar images...")
    distances, indices = knn.kneighbors(query_feature, n_neighbors=5)
    result_images = [images[i] for i in indices[0]]

    print("Visualizing results...")
    visualize_results(query_image, result_images)
