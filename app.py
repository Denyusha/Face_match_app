from flask import Flask, render_template, request, send_from_directory
import face_recognition
import pickle
import os
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MATCHED_FOLDER'] = 'static/matched'

# Update this path to your actual dataset root
DATASET_PATH = "C:/Users/Denyusha/Desktop/face-clustering/face-clustering-master/"

# Load face encodings
with open(os.path.join(DATASET_PATH, "encodings.pickle"), "rb") as f:
    data = pickle.load(f)

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MATCHED_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    matched_images = []

    if request.method == 'POST':
        shutil.rmtree(app.config['MATCHED_FOLDER'], ignore_errors=True)
        os.makedirs(app.config['MATCHED_FOLDER'], exist_ok=True)

        uploaded_file = request.files.get('image')
        if not uploaded_file or uploaded_file.filename == '':
            print("⚠️ No file uploaded.")
            return render_template("index.html", matched_images=matched_images)

        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        # Load and encode the uploaded image
        image = face_recognition.load_image_file(file_path)
        uploaded_encodings = face_recognition.face_encodings(image)

        if not uploaded_encodings:
            print("❌ No face detected in uploaded image.")
            return render_template("index.html", matched_images=matched_images)

        uploaded_encoding = uploaded_encodings[0]

        for entry in data:
            known_encoding = entry.get('encoding')
            known_rel_path = entry.get('imagePath')

            if known_encoding is None or known_rel_path is None:
                print("❌ Skipping entry: missing encoding or imagePath.")
                continue

            full_path = os.path.join(DATASET_PATH, known_rel_path)
            if not os.path.isfile(full_path):
                print(f"❌ File does not exist: {full_path}")
                continue

            matches = face_recognition.compare_faces([known_encoding], uploaded_encoding, tolerance=0.6)
            print(f"🔍 Comparing to: {full_path} → Match: {matches[0]}")

            if matches[0]:
                try:
                    dest_path = os.path.join(app.config['MATCHED_FOLDER'], os.path.basename(known_rel_path))
                    shutil.copy(full_path, dest_path)
                    matched_images.append(os.path.basename(known_rel_path))
                    print(f"✅ Match found: {full_path}")
                except Exception as e:
                    print(f"⚠️ Error copying {full_path}: {e}")

    return render_template("index.html", matched_images=matched_images)

@app.route('/matched/<filename>')
def download_file(filename):
    return send_from_directory(app.config['MATCHED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
