# File: backend/api.py
from flask import Flask, jsonify
from flask_cors import CORS
import base64
import os
import shutil

# Import the core logic from your other file
from core.model_detection import run_full_detection_process

app = Flask(__name__)
CORS(app)  # This allows your React app to make requests to this server

@app.route('/api/run-detection', methods=['GET'])
def detect_deforestation():
    """
    API endpoint to trigger the deforestation detection process.
    It calls the main function, handles the results, and sends them to the frontend.
    """
    try:
        # Run the core detection process and get the paths to the output images
        before_image_path, after_image_path, detection_image_path = run_full_detection_process()

        # Function to read an image file and encode it as a base64 string
        def encode_image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Encode all three images
        before_image_b64 = encode_image_to_base64(before_image_path)
        after_image_b64 = encode_image_to_base64(after_image_path)
        detection_image_b64 = encode_image_to_base64(detection_image_path)

        # Clean up temporary files
        os.remove(before_image_path)
        os.remove(after_image_path)
        os.remove(detection_image_path)

        return jsonify({
            'beforeImage': f'data:image/png;base64,{before_image_b64}',
            'afterImage': f'data:image/png;base64,{after_image_b64}',
            'detectionImage': f'data:image/png;base64,{detection_image_b64}',
            'message': 'Detection complete.'
        }), 200

    except Exception as e:
        # In case of an error, send a clear message to the frontend
        print(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # You can change the port if needed, e.g., port=5001
    app.run(debug=True, port=5000)
