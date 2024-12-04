#final check

import os
import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Paths to dataset (update paths accordingly)
cat_folder = '/content/drive/MyDrive/Dataset/cats'
dog_folder = '/content/drive/MyDrive/Dataset/dogs'

# Preprocessing function
def preprocess_image(image_path, img_size=(128, 128)):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    # Resize image
    img = cv2.resize(img, img_size)
    return img

# Feature extraction using HOG (optimized for fewer features)
def extract_hog_features(img):
    features, _ = hog(img, orientations=9, pixels_per_cell=(16, 16),  # Increased cell size to reduce features
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualize=True, transform_sqrt=True)
    return features

# Load dataset
def load_dataset(cat_folder, dog_folder):
    data = []
    labels = []
    valid_image_extensions = ['.jpg', '.jpeg', '.png']  # List of valid image extensions

    # Process cat images
    for img_name in os.listdir(cat_folder):
        if any(img_name.endswith(ext) for ext in valid_image_extensions):
            img_path = os.path.join(cat_folder, img_name)
            img = preprocess_image(img_path)
            if img is not None:
                features = extract_hog_features(img)
                data.append(features)
                labels.append(0)  # Label for cats

    # Process dog images
    for img_name in os.listdir(dog_folder):
        if any(img_name.endswith(ext) for ext in valid_image_extensions):
            img_path = os.path.join(dog_folder, img_name)
            img = preprocess_image(img_path)
            if img is not None:
                features = extract_hog_features(img)
                data.append(features)
                labels.append(1)  # Label for dogs

    return np.array(data), np.array(labels)

# Main execution
print("Loading dataset...")
X, y = load_dataset(cat_folder, dog_folder)

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train using SGDClassifier for large datasets
print("Training SGDClassifier...")
sgd_model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)  # Faster for large datasets
sgd_model.fit(X_train, y_train)

# Function to classify a new image
def classify_image(image_path, model):
    img = preprocess_image(image_path)
    if img is None:
        return "Error: Could not load image for classification."
    features = extract_hog_features(img)
    features = scaler.transform([features])  # Apply same scaling as training
    prediction = model.predict(features)
    return "\n\nThis is a cat." if prediction[0] == 0 else "\n\nThis is a dog."

# Evaluate the model
print("Evaluating model...")
y_pred = sgd_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Test with an uploaded image
uploaded_image_path = '/content/sample_data/dog.4009.jpg'  # Change this to the path of your uploaded image
result = classify_image(uploaded_image_path, sgd_model)
print(result)

uploaded_image_path1 = '/content/sample_data/cat.4001.jpg'  # Change this to the path of your uploaded image
result1 = classify_image(uploaded_image_path1, sgd_model)
print(result1)


uploaded_image_path2 = '/content/sample_data/dog.9.jpg'  # Change this to the path of your uploaded image
result2 = classify_image(uploaded_image_path2, sgd_model)
print(result2)

uploaded_image_path3 = '/content/sample_data/cat.27.jpg'  # Change this to the path of your uploaded image
result3 = classify_image(uploaded_image_path3, sgd_model)
print(result3)
