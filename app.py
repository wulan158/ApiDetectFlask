import os
import io
import numpy as np
import tensorflow as tf
import sqlite3
import json
from datetime import datetime
from flask import Flask, request, jsonify, g, send_from_directory
from PIL import Image
import uuid

# --- Konfigurasi Awal ---
# Matikan oneDNN jika tidak diperlukan untuk menghindari potensi masalah performa atau kompatibilitas
# Ini opsional, bisa dihapus jika tidak ada masalah.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

app = Flask(__name__)

# --- Konfigurasi Path File ---
# Penting: Pastikan path ini benar relatif terhadap lokasi app.py
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'model_comp.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'assets', 'labels.txt')
DESCRIPTIONS_PATH = os.path.join(BASE_DIR, 'assets', 'descriptions.json')

# --- Konfigurasi Model Input ---
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Konfigurasi Database ---
DATABASE = os.path.join(BASE_DIR, 'database.db')

# --- Konfigurasi Folder Upload ---
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'upload_images')

# --- Variabel Global untuk Model, Label, dan Deskripsi ---
interpreter = None
labels = []
avocado_descriptions = {}

# --- Fungsi Utility ---
def create_upload_folder():
    """Memastikan folder upload_images ada."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"Folder '{UPLOAD_FOLDER}' dibuat.")

# --- Fungsi Database ---
def get_db():
    """Membuka koneksi database baru jika belum ada untuk request saat ini."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            DATABASE,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row # Mengembalikan baris sebagai objek mirip dict
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    """Menutup koneksi database jika ada."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Menginisialisasi skema database dari schema.sql."""
    with app.app_context():
        db = get_db()
        schema_path = os.path.join(BASE_DIR, 'schema.sql')
        if not os.path.exists(schema_path):
            print(f"Error: schema.sql tidak ditemukan di {schema_path}")
            return

        with app.open_resource('schema.sql', mode='r') as f:
            db.executescript(f.read())
        print("Database diinisialisasi.")

# --- Fungsi untuk memuat model, label, dan deskripsi ---
def load_resources():
    """Memuat model TFLite, daftar label, dan deskripsi alpukat."""
    global interpreter, labels, avocado_descriptions

    success = True

    # Load Model
    if interpreter is None:
        try:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            print(f"Model TFLite berhasil dimuat dari: {MODEL_PATH}")
        except Exception as e:
            print(f"Gagal memuat model TFLite dari {MODEL_PATH}: {e}")
            interpreter = None
            success = False

    # Load Labels
    if not labels:
        try:
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]
            print(f"Label berhasil dimuat dari: {LABELS_PATH} ({len(labels)} labels)")
        except Exception as e:
            print(f"Gagal memuat label dari {LABELS_PATH}: {e}")
            labels = []
            success = False

    # Load Descriptions
    if not avocado_descriptions:
        try:
            with open(DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
                avocado_descriptions = json.load(f)
            print(f"Deskripsi alpukat berhasil dimuat dari: {DESCRIPTIONS_PATH}")
        except Exception as e:
            print(f"Gagal memuat deskripsi alpukat dari {DESCRIPTIONS_PATH}: {e}")
            avocado_descriptions = {}
            success = False
    
    return success

# --- Pastikan sumber daya dimuat dan folder dibuat saat aplikasi dimulai ---
with app.app_context():
    create_upload_folder() # Pastikan folder upload ada
    if load_resources():
        print("Semua sumber daya (model, label, deskripsi) berhasil dimuat.")
    else:
        print("Peringatan: Gagal memuat beberapa sumber daya. Prediksi mungkin tidak berfungsi dengan benar.")
    print("Aplikasi Flask siap.")

# --- Endpoint Prediksi ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk melakukan prediksi klasifikasi daun alpukat.
    Menerima gambar dalam bentuk file. Gambar akan disimpan di folder UPLOAD_FOLDER.
    """
    # Pastikan model, label, dan deskripsi sudah dimuat
    if interpreter is None or not labels or not avocado_descriptions:
        if not load_resources(): # Coba muat ulang jika belum ada
            return jsonify({'error': 'Model, label, atau deskripsi gagal dimuat. Coba restart server.'}), 500

    if 'file' not in request.files:
        print("DEBUG: No 'file' part in request.")
        return jsonify({'error': 'Tidak ada bagian file dalam request'}), 400

    file = request.files['file']
    if file.filename == '':
        print("DEBUG: No selected file.")
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if file:
        try:
            # 1. Simpan gambar asli ke folder upload dengan nama unik
            original_filename = file.filename
            extension = os.path.splitext(original_filename)[1].lower() # Pastikan ekstensi huruf kecil
            unique_filename = str(uuid.uuid4()) + extension # UUID + original extension
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)

            img_bytes = file.read() # Baca byte gambar sekali
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
            print(f"DEBUG: File saved to {filepath}.")

            # 2. Proses gambar untuk inferensi model
            image = Image.open(io.BytesIO(img_bytes)) # Buka gambar dari bytes yang sudah dibaca
            print(f"DEBUG: Image opened. Original mode: {image.mode}")

            # Pastikan gambar dalam mode RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            print(f"DEBUG: Image converted to RGB: {image.mode}")

            # Resize gambar ke ukuran input model
            image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS) # Gunakan Resampling.LANCZOS untuk kualitas lebih baik
            print(f"DEBUG: Image resized to {IMG_WIDTH}x{IMG_HEIGHT}.")

            # Konversi ke NumPy array
            input_data = np.asarray(image_resized, dtype=np.float32)
            
            # --- PENTING: Preprocessing MobileNetV3 ---
            # Ini akan menormalisasi piksel ke rentang [-1, 1] dan penyesuaian lain
            input_data = tf.keras.applications.mobilenet_v3.preprocess_input(input_data)
            print("DEBUG: Image pre-processed using MobileNetV3.preprocess_input.")

            # Tambahkan dimensi batch (batch_size=1)
            input_data = np.expand_dims(input_data, axis=0)
            print(f"DEBUG: Final input data shape for model: {input_data.shape}")

            # 3. Lakukan Inferensi Model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Verifikasi bentuk input yang diharapkan model
            expected_input_shape = input_details[0]['shape']
            if list(input_data.shape) != list(expected_input_shape):
                raise ValueError(f"Input shape mismatch! Model expects {expected_input_shape}, but got {input_data.shape}.")

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            print("DEBUG: Model inference invoked.")

            output_tensor = interpreter.get_tensor(output_details[0]['index'])
            probabilities = output_tensor[0] # Ambil hasil probabilitas pertama dari batch
            
            # Debugging probabilitas dan label
            print(f"DEBUG: Raw probabilities from model: {probabilities}")
            print(f"DEBUG: Length of probabilities array: {len(probabilities)}")
            print(f"DEBUG: Number of labels loaded: {len(labels)}")

            # Pastikan jumlah probabilitas cocok dengan jumlah label
            if len(probabilities) != len(labels):
                raise ValueError(f"Mismatch between number of model outputs ({len(probabilities)}) and loaded labels ({len(labels)}).")

            predicted_index = np.argmax(probabilities)
            predicted_label = labels[predicted_index]
            confidence = float(probabilities[predicted_index]) # Pastikan confidence adalah float

            print(f"DEBUG: Predicted Index: {predicted_index}")
            print(f"DEBUG: Predicted Label: {predicted_label}")
            print(f"DEBUG: Confidence: {confidence:.4f}")

            # 4. Ambil Deskripsi
            description = avocado_descriptions.get(predicted_label, {
                "nama_ilmiah": "Tidak diketahui",
                "asal": "Tidak diketahui",
                "karakteristik": "Deskripsi tidak tersedia untuk jenis alpukat ini.",
                "musim_panen": "Tidak diketahui",
                "keunggulan": "Tidak diketahui"
            })
            print("DEBUG: Description retrieved.")

            # 5. Simpan ke Database
            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO predictions (timestamp, predicted_label, confidence, image_filename) VALUES (?, ?, ?, ?)",
                (datetime.now(), predicted_label, confidence, unique_filename)
            )
            db.commit()
            print("DEBUG: Prediction history saved to database.")

            # 6. Kirim Respons
            # Buat dictionary probabilities agar lebih mudah dikonsumsi client
            prob_dict = {label: float(prob) for label, prob in zip(labels, probabilities)}

            response = {
                'prediction': predicted_label,
                'confidence': confidence,
                'description': description,
                'probabilities': prob_dict,
                'saved_image': unique_filename # Nama file yang disimpan di server
            }
            return jsonify(response), 200

        except Exception as e:
            print(f"ERROR in predict endpoint: {e}")
            import traceback
            traceback.print_exc() # Cetak stack trace untuk debug lebih lanjut
            return jsonify({'error': f'Gagal memproses gambar: {e}'}), 500

# --- Endpoint untuk Melayani Gambar yang Disimpan ---
@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    """
    Endpoint untuk melayani file gambar yang disimpan di folder UPLOAD_FOLDER.
    """
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Gambar tidak ditemukan.'}), 404
    except Exception as e:
        print(f"ERROR serving image {filename}: {e}")
        return jsonify({'error': 'Gagal melayani gambar.'}), 500

# --- Endpoint History (GET) ---
@app.route('/history', methods=['GET'])
def get_history():
    """
    Endpoint untuk mendapatkan riwayat prediksi.
    """
    try:
        db = get_db()
        cursor = db.cursor()
        predictions = cursor.execute(
            "SELECT id, timestamp, predicted_label, confidence, image_filename FROM predictions ORDER BY timestamp DESC"
        ).fetchall()

        history_list = []
        for pred in predictions:
            history_list.append({
                'id': pred['id'],
                'timestamp': pred['timestamp'], # Ini sudah dalam format string dari DB
                'predicted_label': pred['predicted_label'],
                'confidence': pred['confidence'],
                'image_filename': pred['image_filename']
            })
        return jsonify({'history': history_list}), 200
    except Exception as e:
        print(f"ERROR in get_history endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Gagal mengambil riwayat: {e}'}), 500

# --- Endpoint History (GET BY ID) ---
@app.route('/history/<int:id>', methods=['GET'])
def get_history_by_id(id):
    """
    Endpoint untuk mendapatkan satu entri riwayat prediksi spesifik berdasarkan ID,
    termasuk deskripsi detail untuk jenis alpukat yang diprediksi.
    """
    # Pastikan deskripsi dimuat
    if not avocado_descriptions:
        if not load_resources(): # Coba muat ulang jika belum ada
            return jsonify({'error': 'Deskripsi gagal dimuat. Coba restart server.'}), 500

    try:
        db = get_db()
        cursor = db.cursor()
        
        prediction = cursor.execute(
            "SELECT id, timestamp, predicted_label, confidence, image_filename FROM predictions WHERE id = ?",
            (id,)
        ).fetchone()

        if prediction is None:
            return jsonify({'message': f'Riwayat dengan ID {id} tidak ditemukan.'}), 404

        # Ambil deskripsi berdasarkan predicted_label
        description = avocado_descriptions.get(prediction['predicted_label'], {
            "nama_ilmiah": "Tidak diketahui",
            "asal": "Tidak diketahui",
            "karakteristik": "Deskripsi tidak tersedia.",
            "musim_panen": "Tidak diketahui",
            "keunggulan": "Tidak diketahui"
        })

        response_data = {
            'id': prediction['id'],
            'timestamp': prediction['timestamp'],
            'predicted_label': prediction['predicted_label'],
            'confidence': prediction['confidence'],
            'image_filename': prediction['image_filename'],
            'description': description # Tambahkan detail deskripsi di sini
        }
        return jsonify(response_data), 200
    except Exception as e:
        print(f"ERROR in get_history_by_id endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Gagal mengambil riwayat dengan ID {id}: {e}'}), 500

		
# --- Endpoint Hapus History (DELETE ALL) ---
@app.route('/history/delete_all', methods=['POST'])
def delete_all_history():
    """
    Endpoint untuk menghapus seluruh riwayat prediksi.
    Juga menghapus file gambar terkait di folder upload_images.
    """
    try:
        db = get_db()
        cursor = db.cursor()

        # Get all image filenames before deleting records
        image_filenames = cursor.execute("SELECT image_filename FROM predictions").fetchall()

        cursor.execute("DELETE FROM predictions")
        db.commit()

        # Delete corresponding image files
        deleted_files_count = 0
        for row in image_filenames:
            filename = row['image_filename']
            if filename: # Pastikan filename tidak None atau kosong
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    deleted_files_count += 1
                    print(f"DEBUG: Deleted file: {filepath}")
                else:
                    print(f"DEBUG: File not found for deletion: {filepath}") # Log jika file tidak ada

        return jsonify({'message': f'Seluruh riwayat prediksi dan {deleted_files_count} file gambar berhasil dihapus.'}), 200
    except Exception as e:
        print(f"ERROR in delete_all_history endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Gagal menghapus seluruh riwayat: {e}'}), 500

# --- Endpoint Hapus History by ID (DELETE) ---
@app.route('/history/delete/<int:id>', methods=['DELETE'])
def delete_history_by_id(id):
    """
    Endpoint untuk menghapus riwayat prediksi berdasarkan ID.
    Juga menghapus file gambar terkait di folder upload_images.
    """
    try:
        db = get_db()
        cursor = db.cursor()

        # Get filename before deleting record
        cursor.execute("SELECT image_filename FROM predictions WHERE id = ?", (id,))
        record = cursor.fetchone()
        
        if record is None:
            return jsonify({'message': f'Riwayat dengan ID {id} tidak ditemukan.'}), 404
            
        filename_to_delete = record['image_filename']

        cursor.execute("DELETE FROM predictions WHERE id = ?", (id,))
        db.commit()

        # Delete corresponding image file
        if filename_to_delete:
            filepath = os.path.join(UPLOAD_FOLDER, filename_to_delete)
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"DEBUG: Deleted file: {filepath}")
            else:
                print(f"DEBUG: File not found for deletion: {filepath}") # Log jika file tidak ada

        return jsonify({'message': f'Riwayat prediksi dengan ID {id} berhasil dihapus.'}), 200
    except Exception as e:
        print(f"ERROR in delete_history_by_id endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Gagal menghapus riwayat dengan ID {id}: {e}'}), 500

# --- Endpoint Get All Descriptions ---
@app.route('/descriptions', methods=['GET'])
def get_all_descriptions():
    """
    Endpoint untuk mendapatkan semua deskripsi jenis alpukat yang tersedia.
    """
    if not avocado_descriptions:
        if not load_resources(): # Coba muat ulang jika belum ada
            return jsonify({'error': 'Deskripsi gagal dimuat. Coba restart server.'}), 500
        
    try:
        # Mengembalikan objek avocado_descriptions secara langsung
        # karena sudah berupa dictionary yang siap jadi JSON
        return jsonify(avocado_descriptions), 200
    except Exception as e:
        print(f"ERROR in get_all_descriptions endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Gagal mengambil semua deskripsi: {e}'}), 500
		
# --- Endpoint Get Single Description by Name ---
@app.route('/descriptions/<string:nama_jenis>', methods=['GET'])
def get_single_description(nama_jenis):
    """
    Endpoint untuk mendapatkan deskripsi detail satu jenis alpukat spesifik.
    Menerima nama jenis alpukat sebagai parameter di URL.
    """
    if not avocado_descriptions:
        if not load_resources(): # Coba muat ulang jika belum ada
            return jsonify({'error': 'Deskripsi gagal dimuat. Coba restart server.'}), 500
        
    # Format nama_jenis dari URL untuk memastikan konsistensi (misal: "Alpukat_Aligator")
    # Anda mungkin perlu menyesuaikan ini jika format nama di URL berbeda dari kunci di JSON
    formatted_nama_jenis = nama_jenis.replace(' ', '_').replace('-', '_') # Contoh: "Alpukat Aligator" -> "Alpukat_Aligator"

    description = avocado_descriptions.get(formatted_nama_jenis)

    if description:
        # Kembalikan objek deskripsi untuk jenis yang ditemukan
        return jsonify(description), 200
    else:
        # Jika nama jenis tidak ditemukan
        return jsonify({'message': f'Deskripsi untuk jenis "{nama_jenis}" tidak ditemukan.'}), 404

# --- Command Line untuk Inisialisasi Database ---
@app.cli.command('init-db')
def init_db_command():
    """Clear existing data and create new tables."""
    init_db()
    print('Initialized the database.')

if __name__ == '__main__':
    # Pastikan inisialisasi database hanya dilakukan sekali atau secara manual
    # init_db() # Jangan panggil ini di sini jika Anda tidak ingin database direset setiap kali server dimulai
    app.run(debug=True, host='0.0.0.0', port=5000)