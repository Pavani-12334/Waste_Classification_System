# -------------------------------
# Waste Classification System
# -------------------------------

import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# -------------------------------
# Step 0: Suppress TensorFlow info messages
# -------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hides info messages

# -------------------------------
# Step 1: Reorganize dataset into train/test folders
# -------------------------------

# Correct path to the inner dataset folder (Windows)
original_path = r"C:\Users\Pavani\OneDrive\Desktop\Waste_Classification_Sys\dataset\Garbage classification\Garbage classification"
base_path = r"C:\Users\Pavani\OneDrive\Desktop\Waste_Classification_Sys\dataset"

split_ratio = 0.8  # 80% train, 20% test

# Map folders to classes
classes = {
    "Recyclable": ["cardboard", "glass", "metal", "paper", "plastic"],
    "Non-Recyclable": ["trash"]
}

# Verify folders exist
print("Folders inside dataset:", os.listdir(original_path))

# Create train/test folders
for split in ["train", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(base_path, split, cls), exist_ok=True)

# Move images into train/test folders
for cls, folders in classes.items():
    for folder in folders:
        folder_path = os.path.join(original_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: folder not found -> {folder_path}")
            continue
        images = os.listdir(folder_path)
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        for img in train_images:
            shutil.copy(os.path.join(folder_path, img), os.path.join(base_path, "train", cls))
        for img in test_images:
            shutil.copy(os.path.join(folder_path, img), os.path.join(base_path, "test", cls))

print("Dataset reorganized successfully!")

# -------------------------------
# Step 2: Data preprocessing and augmentation
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_path, "train"),
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_path, "test"),
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

# -------------------------------
# Step 3: Build CNN model
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# Step 4: Train the model
# -------------------------------
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=test_generator,
    validation_steps=test_generator.samples // 32,
    epochs=10  # adjust based on dataset size
)

# -------------------------------
# Step 5: Evaluate and save model
# -------------------------------
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy*100:.2f}%")
model.save(os.path.join(base_path, "waste_classifier_model.h5"))
print("Model saved as waste_classifier_model.h5")

# -------------------------------
# Step 6: Predict on a single image
# -------------------------------
def predict_waste(img_path):
    img = load_img(img_path, target_size=(128,128))
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_labels = list(train_generator.class_indices.keys())
    print("Predicted Class:", class_labels[class_index])

# Example usage:
# predict_waste(r"C:\Users\Pavani\OneDrive\Desktop\Waste_Classification_Sys\dataset\test\Recyclable\cardboard1.jpg")
