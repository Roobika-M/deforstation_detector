# File: backend/core/model_detection.py
import rasterio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def run_full_detection_process(latest_image_path='../data/sample_2023.tif'):
    """
    This function encapsulates the entire detection process.
    It loads the model, makes a prediction on the latest image,
    and visualizes the results, saving the output images to a temp folder.
    
    For this demo, we're using a sample image instead of the GEE output.
    You will need to integrate the GEE exporter logic here to get a truly live image.
    """
    # ----------------------------------------
    # 1. Load Model and Images
    # ----------------------------------------
    model_path = '../models/deforestation_model.h5'
    before_image_path = '../data/sample_2018.tif'

    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        with rasterio.open(latest_image_path) as src:
            new_image = src.read()
        with rasterio.open(before_image_path) as src:
            before_image = src.read()
        print("Images loaded successfully.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {e}. Please check your file paths.")

    # ----------------------------------------
    # 2. Pre-process and Make a Prediction
    # ----------------------------------------
    def normalize_and_pad(image, tile_size=256):
        image = image.astype('float32') / 3000.0
        padded_height = (image.shape[1] // tile_size + 1) * tile_size
        padded_width = (image.shape[2] // tile_size + 1) * tile_size
        padded_image = np.pad(image, ((0, 0), (0, padded_height - image.shape[1]), (0, padded_width - image.shape[2])), mode='constant')
        padded_image = np.transpose(padded_image, (1, 2, 0))
        return padded_image, padded_height, padded_width

    tile_size = 256
    processed_image, padded_height, padded_width = normalize_and_pad(new_image, tile_size)

    input_tiles = []
    for y in range(0, padded_height, tile_size):
        for x in range(0, padded_width, tile_size):
            input_tile = processed_image[y:y + tile_size, x:x + tile_size, :]
            input_tiles.append(input_tile)

    input_for_prediction = np.array(input_tiles)
    predictions = model.predict(input_for_prediction)
    print("\nPrediction complete.")

    full_prediction_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
    tile_index = 0
    for y in range(0, padded_height, tile_size):
        for x in range(0, padded_width, tile_size):
            prediction_tile = (predictions[tile_index, :, :, 0] > 0.5).astype(np.uint8)
            full_prediction_mask[y:y + tile_size, x:x + tile_size] = prediction_tile
            tile_index += 1

    original_height = new_image.shape[1]
    original_width = new_image.shape[2]
    final_mask = full_prediction_mask[:original_height, :original_width]

    # ----------------------------------------
    # 3. Visualize and Save Results
    # ----------------------------------------
    def visualize_image(image):
        rgb_image = np.transpose(image[:3, :, :], (1, 2, 0))
        vis_min = 100
        vis_max = 3000
        rgb_image_clipped = np.clip(rgb_image, vis_min, vis_max)
        normalized_image = (rgb_image_clipped - vis_min) / (vis_max - vis_min)
        return normalized_image

    # Create temporary files for the images
    temp_dir = 'temp_images'
    os.makedirs(temp_dir, exist_ok=True)
    before_output_path = os.path.join(temp_dir, 'before.png')
    after_output_path = os.path.join(temp_dir, 'after.png')
    detection_output_path = os.path.join(temp_dir, 'detection.png')

    # Save the 'before' image
    plt.imsave(before_output_path, visualize_image(before_image))

    # Save the 'after' image
    plt.imsave(after_output_path, visualize_image(new_image))

    # Save the 'detection' image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(visualize_image(new_image))
    ax.imshow(final_mask, cmap='Reds', alpha=0.5)
    ax.axis('off')
    plt.savefig(detection_output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Close the plot to free up memory

    return before_output_path, after_output_path, detection_output_path
