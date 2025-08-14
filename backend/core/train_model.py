# File: backend/core/train_model.py
# This script trains the U-Net model and saves it to an HDF5 file.

import rasterio
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def unet_model(input_shape=(256, 256, 6)):
    """Defines the U-Net model architecture."""
    inputs = tf.keras.Input(input_shape)

    # Encoder (Downsampling)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)

    # Decoder (Upsampling)
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = Conv2D(64, 2, activation='relu', padding='same')(up4)
    merge4 = concatenate([conv2, up4], axis=3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(merge4)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(32, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv1, up5], axis=3)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(conv5)

    # Output layer with a single neuron for the segmentation mask
    output = Conv2D(1, 1, activation='sigmoid')(conv5)

    return Model(inputs=inputs, outputs=output)

def main():
    """Main function to load data, train the model, and save it."""
    # Define your file paths (assuming you have sample data in the data folder)
    input_path = '../../data/deforestation_input_image.tif'
    labels_path = '../../data/deforestation_labels.tif'

    # Load the full input image and labels
    try:
        with rasterio.open(input_path) as src:
            full_input_image = src.read()
        with rasterio.open(labels_path) as src:
            full_labels = src.read(1)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file. Please ensure {input_path} and {labels_path} exist.")
        return

    # Normalize the input image data
    full_input_image = full_input_image.astype('float32') / 3000.0

    # Pad the images to be a multiple of 256 for tiling
    tile_size = 256
    padded_height = (full_input_image.shape[1] // tile_size + 1) * tile_size
    padded_width = (full_input_image.shape[2] // tile_size + 1) * tile_size
    input_padded = np.pad(full_input_image, ((0, 0), (0, padded_height - full_input_image.shape[1]), (0, padded_width - full_input_image.shape[2])), mode='constant')
    labels_padded = np.pad(full_labels, ((0, padded_height - full_labels.shape[0]), (0, padded_width - full_labels.shape[1])), mode='constant')

    # Transpose the input image to (height, width, channels)
    input_padded = np.transpose(input_padded, (1, 2, 0))

    # Create tiles from the padded images
    input_tiles = []
    label_tiles = []
    for y in range(0, padded_height, tile_size):
        for x in range(0, padded_width, tile_size):
            input_tile = input_padded[y:y + tile_size, x:x + tile_size, :]
            label_tile = labels_padded[y:y + tile_size, x:x + tile_size]
            input_tiles.append(input_tile)
            label_tiles.append(np.expand_dims(label_tile, axis=-1))

    X = np.array(input_tiles)
    y = np.array(label_tiles)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and compile the U-Net model
    # Note: The input shape should be (256, 256, 6) for a before/after image pair
    model = unet_model(input_shape=(256, 256, 6))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Starting model training...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
    print("Training finished.")

    # Save the trained model to the models folder
    model_save_path = '../../models/deforestation_model.h5'
    model.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")

if __name__ == '__main__':
    main()
