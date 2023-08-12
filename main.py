import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import image_processing 

THRESHOLD = 0.6
TARGET_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5

# Load images and preprocess them
def preprocess_image(image_path):
    img = load_img(image_path, target_size=TARGET_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet.preprocess_input(img)
    return img

# Create pairs of images and their labels
def create_pairs(dataset_dir):
    pairs = []
    labels = []

    # Collect image paths and labels
    image_paths = []
    labels_dict = {}  # To map label (writer ID) to image paths

    for writer_label, writer_folder in enumerate(os.listdir(dataset_dir)):
        writer_folder_path = os.path.join(dataset_dir, writer_folder)
        if not os.path.isdir(writer_folder_path):
            continue

        labels_dict[writer_label] = []

        for img_name in os.listdir(writer_folder_path):
            img_path = os.path.join(writer_folder_path, img_name)
            image_paths.append(img_path)
            labels_dict[writer_label].append(img_path)

    # Create same-writer pairs
    for label, paths in labels_dict.items():
        for i in range(len(paths) - 1):
            for j in range(i + 1, len(paths)):
                pairs.append((paths[i], paths[j]))
                labels.append(1)

    # Create different-writer pairs
    for i in range(len(image_paths) - 1):
        for j in range(i + 1, len(image_paths)):
            pairs.append((image_paths[i], image_paths[j]))
            labels.append(0)

    return np.array(pairs), np.array(labels)

# Siamese network architecture
def create_siamese_network(input_shape):
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    input_1 = layers.Input(input_shape)
    input_2 = layers.Input(input_shape)

    features_1 = base_model(input_1)
    features_2 = base_model(input_2)

    L1_distance = tf.abs(features_1 - features_2)
    output = layers.Dense(1, activation="sigmoid")(L1_distance)

    siamese_network = keras.Model(inputs=[input_1, input_2], outputs=output)
    return siamese_network

# Load and preprocess data
pairs, labels = create_pairs("dataset/train")

# Replace preprocessing step with processed image
processed_image = image_processing.final_image
processed_images = [processed_image for _ in range(len(pairs))]

# Shuffle and split data
random_indices = np.random.permutation(len(pairs))
pairs = pairs[random_indices]
labels = labels[random_indices]
num_train = int(0.8 * len(pairs))

train_pairs, train_labels = pairs[:num_train], labels[:num_train]
test_pairs, test_labels = pairs[num_train:], labels[num_train:]

# Create and compile the siamese network
siamese_network = create_siamese_network(TARGET_SIZE + (3,))
siamese_network.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the siamese network
siamese_network.fit(
    [processed_images[pair[0]] for pair in train_pairs],
    [processed_images[pair[1]] for pair in train_pairs],
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(
        [processed_images[pair[0]] for pair in test_pairs],
        [processed_images[pair[1]] for pair in test_pairs],
        test_labels,
    ),
)

# Save the model
siamese_network.save("siamese_model.h5")
