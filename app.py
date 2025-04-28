from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash, jsonify, session, send_file
import face_recognition
import pickle
import os
from werkzeug.utils import secure_filename
import shutil
from pymongo import MongoClient
from datetime import datetime
import base64
from bson.binary import Binary
import gridfs
from bson import ObjectId
import json
from functools import wraps
from models import User
from PIL import Image
import numpy as np
import glob
import zipfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MATCHED_FOLDER'] = 'static/matched'
app.config['TRAINING_FOLDER'] = 'static/training'
app.secret_key = 'your-secret-key'

# MongoDB connection (local)
client = MongoClient('mongodb://localhost:27017/')
db = client.face_matching_db
fs = gridfs.GridFS(db)

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MATCHED_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAINING_FOLDER'], exist_ok=True)

def train_model():
    """Train the model on images in the training folder and store encodings in MongoDB"""
    try:
        # Clear existing encodings
        db.images.delete_many({})
        
        # Get list of images in training folder
        image_paths = glob.glob(os.path.join(app.config['TRAINING_FOLDER'], '*.jpg')) + \
                     glob.glob(os.path.join(app.config['TRAINING_FOLDER'], '*.jpeg')) + \
                     glob.glob(os.path.join(app.config['TRAINING_FOLDER'], '*.png'))
        
        print(f"Found {len(image_paths)} images to train on")
        
        # Process each image
        for (i, image_path) in enumerate(image_paths):
            print(f"Processing image {i + 1}/{len(image_paths)}: {image_path}")
            
            try:
                # Load image using PIL
                image = Image.open(image_path)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize if too large
                if image.width > 800:
                    scale = 800 / image.width
                    new_height = int(image.height * scale)
                    image = image.resize((800, new_height), Image.Resampling.LANCZOS)
                
                # Convert to numpy array for face_recognition
                image_array = np.array(image)
                
                # Detect faces
                encodings = face_recognition.face_encodings(image_array)
                
                if not encodings:
                    print(f"No face detected in {image_path}")
                    continue
                
                # Save image to GridFS
                filename = os.path.basename(image_path)
                with open(image_path, 'rb') as f:
                    file_id = fs.put(f.read(), filename=filename)
                
                # Store only the first face found
                image_doc = {
                    'filename': filename,
                    'image_id': file_id,
                    'encoding': Binary(pickle.dumps(encodings[0])),  # Store only first face
                    'upload_date': datetime.utcnow(),
                    'is_training_image': True
                }
                db.images.insert_one(image_doc)
                
                print(f"Successfully processed {filename}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print("Training completed successfully")
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        return False

def cleanup_temp_files():
    """Clean up temporary files from upload and matched folders"""
    try:
        for folder in [app.config['UPLOAD_FOLDER'], app.config['MATCHED_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
    except Exception as e:
        print(f"Error in cleanup: {e}")

def check_disk_space():
    """Check if there's enough disk space (at least 100MB free)"""
    import shutil
    total, used, free = shutil.disk_usage("/")
    return free > 100 * 1024 * 1024

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or not user.get('is_admin'):
            flash('Admin access required', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.get_user_by_username(db, username)
        if user and User.verify_password(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            session['is_admin'] = user.get('is_admin', False)

            if user.get('is_admin'):
                return redirect(url_for('admin_panel'))
            else:
                return redirect(url_for('student_panel'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        is_admin = request.form.get('is_admin') == 'on'

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))

        user_data = {
            'username': username,
            'email': email,
            'password': password,
            'is_admin': is_admin
        }

        user = User.create_user(db, user_data)
        if user:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username or email already exists', 'danger')

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/student', methods=['GET', 'POST'])
@login_required
def student_panel():
    if request.method == 'POST':
        try:
            if not check_disk_space():
                cleanup_temp_files()
                if not check_disk_space():
                    flash('Not enough disk space.', 'error')
                    return render_template('student.html', matched_images=[])

            cleanup_temp_files()
            os.makedirs(app.config['MATCHED_FOLDER'], exist_ok=True)

            uploaded_file = request.files.get('image')
            if not uploaded_file or uploaded_file.filename == '':
                flash('No file uploaded', 'error')
                return render_template('student.html', matched_images=[])

            if uploaded_file.content_length > 5 * 1024 * 1024:
                flash('File size too large.', 'error')
                return render_template('student.html', matched_images=[])

            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save uploaded file
            with open(file_path, 'wb') as f:
                while True:
                    chunk = uploaded_file.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)

            # Load and process the uploaded image
            try:
                image = face_recognition.load_image_file(file_path)
                
                # Resize if too large
                if image.shape[0] > 800 or image.shape[1] > 800:
                    pil_image = Image.fromarray(image)
                    scale = min(800 / image.shape[0], 800 / image.shape[1])
                    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    image = np.array(pil_image)

                # Get face encodings
                uploaded_encodings = face_recognition.face_encodings(image)
                
                if not uploaded_encodings:
                    flash('No face detected in uploaded image', 'error')
                    return render_template('student.html', matched_images=[])

                uploaded_encoding = uploaded_encodings[0]
                matched_images = []

                # Get all stored images from MongoDB
                stored_images = list(db.images.find({'is_training_image': True}))
                print(f"Found {len(stored_images)} stored encodings to compare against")

                for stored_image in stored_images:
                    try:
                        if 'encoding' not in stored_image:
                            print(f"No encoding found for {stored_image['filename']}")
                            continue

                        try:
                            stored_encoding = pickle.loads(stored_image['encoding'])
                        except Exception as e:
                            print(f"Failed to load encoding for {stored_image['filename']}: {e}")
                            continue

                        # Calculate face distance
                        face_distance = face_recognition.face_distance([stored_encoding], uploaded_encoding)[0]
                        print(f"Distance to {stored_image['filename']}: {face_distance}")

                        # If face distance is less than 0.6, consider it a match
                        if face_distance < 0.6:
                            try:
                                # Get image from GridFS
                                image_data = fs.get(stored_image['image_id'])
                                
                                # Save to matched folder
                                dest_path = os.path.join(app.config['MATCHED_FOLDER'], stored_image['filename'])
                                with open(dest_path, 'wb') as f:
                                    f.write(image_data.read())
                                
                                # Calculate confidence percentage
                                confidence = 100 - (face_distance * 100)
                                matched_images.append({
                                    'filename': stored_image['filename'],
                                    'confidence': f"{confidence:.1f}%",
                                    'image_id': str(stored_image['_id'])  # Add image_id for download
                                })
                                print(f"Match found: {stored_image['filename']} with confidence {confidence:.1f}%")
                            except Exception as e:
                                print(f"Error processing matched image {stored_image['filename']}: {e}")
                    except Exception as e:
                        print(f"Error comparing with {stored_image['filename']}: {e}")

                # Clean up uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)

                if matched_images:
                    # Sort matches by confidence
                    matched_images.sort(key=lambda x: float(x['confidence'].strip('%')), reverse=True)
                    flash(f'Found {len(matched_images)} matching images', 'success')
                else:
                    flash('No matching images found', 'info')

                return render_template('student.html', matched_images=matched_images)

            except Exception as e:
                print(f"Error processing image: {e}")
                flash(f'Error processing image: {str(e)}', 'error')
                return render_template('student.html', matched_images=[])

        except Exception as e:
            print(f"General error: {e}")
            flash(f'Error: {str(e)}', 'error')
            return render_template('student.html', matched_images=[])

    return render_template('student.html', matched_images=[])

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/image/<image_id>')
def get_image(image_id):
    try:
        image = db.images.find_one({'_id': ObjectId(image_id)})
        if image and 'image_id' in image:
            grid_file = fs.get(image['image_id'])
            return send_file(
                grid_file,
                mimetype='image/jpeg',
                as_attachment=False
            )
        return 'Image not found', 404
    except Exception as e:
        return str(e), 500

@app.route('/download/<image_id>')
def download_image(image_id):
    try:
        image = db.images.find_one({'_id': ObjectId(image_id)})
        if image and 'image_id' in image:
            grid_file = fs.get(image['image_id'])
            return send_file(
                grid_file,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=image['filename']
            )
        return 'Image not found', 404
    except Exception as e:
        return str(e), 500

@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin_panel():
    if request.method == 'POST':
        try:
            files = request.files.getlist('files[]')
            if not files or files[0].filename == '':
                return jsonify({'success': False, 'message': 'No files selected'})
            
            uploaded_files = []
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['TRAINING_FOLDER'], filename)
                    file.save(file_path)
                    uploaded_files.append(filename)
            
            if uploaded_files:
                # Train model with new images
                if train_model():
                    return jsonify({
                        'success': True,
                        'message': f'Successfully uploaded and trained on {len(uploaded_files)} files'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Error during training'
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No valid files were uploaded'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error uploading files: {str(e)}'
            })
    
    # GET request handling
    training_images = list(db.images.find({'is_training_image': True}).sort('upload_date', -1))
    return render_template('admin.html', uploaded_images=training_images)

@app.route('/delete_image/<image_id>', methods=['POST'])
@admin_required
def delete_image(image_id):
    try:
        # Convert string ID to ObjectId
        image_id = ObjectId(image_id)
        
        # Get image document
        image = db.images.find_one({'_id': image_id})
        if not image:
            return jsonify({'success': False, 'message': 'Image not found'}), 404
        
        try:
            # Delete from GridFS
            fs.delete(image['image_id'])
            
            # Delete from MongoDB
            db.images.delete_one({'_id': image_id})
            
            # Delete from training folder if exists
            training_path = os.path.join(app.config['TRAINING_FOLDER'], image['filename'])
            if os.path.exists(training_path):
                os.remove(training_path)
            
            return jsonify({'success': True, 'message': 'Image deleted successfully'})
        except Exception as e:
            print(f"Error deleting image: {e}")
            return jsonify({'success': False, 'message': f'Error deleting image: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/delete_images', methods=['POST'])
@admin_required
def delete_images():
    try:
        data = request.get_json()
        image_ids = [ObjectId(id) for id in data.get('image_ids', [])]
        
        if not image_ids:
            return jsonify({'success': False, 'message': 'No images selected'}), 400
        
        # Get all images to be deleted
        images = list(db.images.find({'_id': {'$in': image_ids}}))
        
        # Delete from GridFS and training folder
        for image in images:
            try:
                # Delete from GridFS
                fs.delete(image['image_id'])
                
                # Delete from training folder if exists
                training_path = os.path.join(app.config['TRAINING_FOLDER'], image['filename'])
                if os.path.exists(training_path):
                    os.remove(training_path)
            except Exception as e:
                print(f"Error deleting image {image['filename']}: {e}")
                continue
        
        # Delete from MongoDB
        db.images.delete_many({'_id': {'$in': image_ids}})
        
        return jsonify({'success': True, 'message': f'{len(image_ids)} images deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/matched/<filename>')
def download_file(filename):
    return send_from_directory(app.config['MATCHED_FOLDER'], filename, as_attachment=True)

@app.route('/download_all', methods=['POST'])
def download_all():
    try:
        image_ids = request.form.get('image_ids')
        if not image_ids:
            return jsonify({'success': False, 'message': 'No images selected'}), 400
            
        image_ids = json.loads(image_ids)
        if not image_ids:
            return jsonify({'success': False, 'message': 'No images selected'}), 400
            
        # Create a temporary directory
        temp_dir = os.path.join(app.config['MATCHED_FOLDER'], 'temp_download')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download all images
        for image_id in image_ids:
            try:
                image = db.images.find_one({'_id': ObjectId(image_id)})
                if image and 'image_id' in image:
                    grid_file = fs.get(image['image_id'])
                    file_path = os.path.join(temp_dir, image['filename'])
                    with open(file_path, 'wb') as f:
                        f.write(grid_file.read())
            except Exception as e:
                print(f"Error downloading image {image_id}: {e}")
                continue
        
        # Create a zip file
        zip_path = os.path.join(app.config['MATCHED_FOLDER'], 'matched_images.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        # Send the zip file
        return send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name='matched_images.zip'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
