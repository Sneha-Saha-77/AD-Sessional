import os
import time
import uuid
import tempfile
import numpy as np
import tensorflow as tf
import json
import random
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'your_secret_key'  # Required for session management and flashing messages

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['HISTORY_FILE'] = 'prediction_history.json'
app.config['USER_DATA_FILE'] = 'users.json'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pre-trained model
try:
    model = tf.keras.models.load_model('plant_disease_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Disease class mapping
CLASS_INDICES = {
    0: 'Bacterial infection in Bell Pepper',
    1: 'Healthy Bell Pepper',
    2: 'Early blight in Potato',
    3: 'Late blight in Potato',
    4: 'Healthy Potato',
    5: 'Bacterial spot in Tomato',
    6: 'Early blight in Tomato',
    7: 'Late blight in Tomato',
    8: 'Mold in leaf in Tomato',
    9: 'Septoria in leaf in Tomato',
    10: 'Spider mites in Tomato',
    11: 'Target spots in Tomato',
    12: 'Yellowing in Tomato',
    13: 'Mosaic in Tomato',
    14: 'Healthy Tomato'
}

def load_history():
    if os.path.exists(app.config['HISTORY_FILE']):
        with open(app.config['HISTORY_FILE'], 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(app.config['HISTORY_FILE'], 'w') as f:
        json.dump(history, f)

def load_users():
    if os.path.exists(app.config['USER_DATA_FILE']):
        with open(app.config['USER_DATA_FILE'], 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(app.config['USER_DATA_FILE'], 'w') as f:
        json.dump(users, f)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_remove_file(filepath, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            os.remove(filepath)
            return True
        except PermissionError:
            time.sleep(0.2)
        except Exception as e:
            print(f"Error removing file {filepath}: {e}")
            break
    return False

def prepare_image(file_path):
    try:
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [256, 256])
        img = img / 255.0
        img = tf.expand_dims(img, 0)
        return img
    except Exception as e:
        print(f"Image preparation error: {e}")
        return None

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/login.html', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()

        if username in users and users[username] == password:
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/register.html', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()

        if username in users:
            flash('Username already exists', 'danger')
        else:
            users[username] = password
            save_users(users)
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/random_image', methods=['GET'])
def random_image():
    folder = request.args.get('folder')
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400

    folder_path = os.path.join('PlantVillage', folder)
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder not found'}), 404

    try:
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]
        if not image_files:
            return jsonify({'error': 'No images found in folder'}), 404

        random_image = random.choice(image_files)
        image_path = os.path.join(folder_path, random_image)
        return jsonify({'image_path': image_path})
    except Exception as e:
        print(f"Error fetching random image: {e}")
        return jsonify({'error': 'Failed to fetch random image'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Plant disease model could not be initialized.'
        }), 500

    if 'file' not in request.files:
        return jsonify({
            'error': 'No file',
            'message': 'No file was uploaded.'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'error': 'No selected file',
            'message': 'Please select a file to upload.'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Only image files (png, jpg, jpeg, gif) are allowed.'
        }), 400

    filepath = None
    try:
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        processed_image = prepare_image(filepath)
        if processed_image is None:
            safe_remove_file(filepath)
            return jsonify({
                'error': 'Image processing failed',
                'message': 'Could not process the uploaded image.'
            }), 400

        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_INDICES.get(predicted_class_index, 'Unknown Disease')
        confidence = float(predictions[0][predicted_class_index])

        prediction_history = load_history()
        prediction_history.append({
            'disease': predicted_class,
            'confidence': confidence * 100,
            'image': unique_filename
        })
        save_history(prediction_history)

        safe_remove_file(filepath)

        return jsonify({
            'disease': predicted_class,
            'confidence': confidence * 100,
            'history': prediction_history
        })

    except Exception as e:
        if filepath and os.path.exists(filepath):
            safe_remove_file(filepath)
        print(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)