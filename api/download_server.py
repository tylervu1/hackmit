from flask import Flask, request, jsonify
import os
from app import process_uploaded_file

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Create the '../uploads' directory if it doesn't exist
        upload_dir = os.path.join('..', 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # Save the file to your server or process it
        save_path = os.path.join(upload_dir, file.filename)
        file.save(save_path)

        process_uploaded_file(file.filename)
        return jsonify({"message": "File uploaded successfully"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5300)