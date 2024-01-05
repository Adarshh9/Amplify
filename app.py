from flask import Flask, render_template, request, send_file, jsonify
from methods import get_custom_augmented_images, create_tar_gz
import os
import threading
import time

app = Flask(__name__)

# Flag to check if image processing is complete
processing_complete = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def upload():
    global processing_complete
    processing_complete = False  # Reset the flag
    input_dir = 'images/'
    output_dir = 'augmented_images/'
    os.makedirs(output_dir, exist_ok=True)

    uploaded_files = request.files.getlist("image")

    for uploaded_file in uploaded_files:
        img_path = os.path.join(input_dir, uploaded_file.filename)
        uploaded_file.save(img_path)

    selected_options = request.form.getlist('selected_options[]')

    print(selected_options)

    # Perform image augmentation
    thread = threading.Thread(target=process_images, args=(input_dir, output_dir, selected_options))
    thread.start()

    return render_template('loading.html')


def process_images(input_dir, output_dir, selected_options):
    global processing_complete
    get_custom_augmented_images(input_dir, output_dir, user_choice=selected_options)
    zip_filename = 'augmented_images.tar.gz'
    create_tar_gz(folder_path=output_dir, output_filename=zip_filename)
    processing_complete = True


@app.route('/check_processing_complete')
def check_processing_complete():
    global processing_complete
    return jsonify({"processing_complete": processing_complete})


@app.route('/reset', methods=['POST'])
def reset():
    zip_path = 'augmented_images.tar.gz'
    output_dir = 'augmented_images/'
    input_dir = 'images/'

    # Delete the zip file if it exists
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # Delete the output directory and its contents
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        os.rmdir(output_dir)

    # Delete all files in the input directory
    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return jsonify({"status": "success"})


@app.route('/download')
def download():
    zip_path = 'augmented_images.tar.gz'
    return send_file(zip_path, as_attachment=True)


@app.route('/result_page')
def result_page():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True, port=5004)
