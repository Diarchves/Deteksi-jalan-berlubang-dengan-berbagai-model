import argparse
import numpy as np
from pathlib import Path
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:
    raise ImportError("Sistem mendeteksi modul 'tensorflow' tidak tersedia. Install dengan: pip install tensorflow") from exc

# ==========================================
# 1. PARAMETER KONFIGURASI
# ==========================================
IMAGE_SIZE = 128  # Wajib sama dengan resolusi saat training

# Jalur dinamis relatif terhadap skrip
SCRIPT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = SCRIPT_DIR / "models" / "cnn_model.keras"
ENCODER_PATH = SCRIPT_DIR / "models" / "label_encoder.pkl"
DEFAULT_TEST_IMAGE = SCRIPT_DIR / "results" / "pothole-17.png" # Ganti dengan gambar default yang Anda punya

# ==========================================
# 2. FUNGSI PREDIKSI
# ==========================================
def predict_image(image_path):
    # Verifikasi keberadaan file
    if not Path(image_path).exists():
        print(f"Galat: Gambar tidak ditemukan pada {image_path}")
        return
    
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        print("Galat: Artefak model (cnn_model.keras atau label_encoder.pkl) tidak ditemukan di folder models.")
        return

    # Memuat Model dan Encoder
    try:
        model = keras.models.load_model(str(MODEL_PATH))
        encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        print(f"Galat saat memuat artefak: {e}")
        return

    # Pemrosesan Awal Gambar (Persis seperti fungsi map di training)
    try:
        image_bytes = tf.io.read_file(str(image_path))
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = tf.cast(image, tf.float32) / 255.0  # Normalisasi 0-1
        
        # Tambahkan dimensi batch (karena model mengharapkan input bentuk [batch_size, width, height, channels])
        image_batch = tf.expand_dims(image, axis=0) 
    except Exception as e:
        print(f"Galat saat memproses gambar: {e}")
        return

    # Eksekusi Prediksi
    predictions = model.predict(image_batch, verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    
    # Dekode Label
    predicted_label = encoder.inverse_transform([predicted_index])[0]

    # Menampilkan Hasil
    print("\n" + "="*30)
    print("HASIL PREDIKSI CNN")
    print("="*30)
    print(f"Gambar   : {Path(image_path).name}")
    print(f"Kelas    : {predicted_label}")
    print(f"Keyakinan: {confidence * 100:.2f}%")
    print("="*30)

# ==========================================
# 3. ALUR PROGRAM UTAMA
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Pengujian Model CNN untuk Deteksi Jalan")
    parser.add_argument(
        "image", 
        nargs="?", 
        default=str(DEFAULT_TEST_IMAGE), 
        help="Jalur (path) menuju gambar yang ingin diprediksi"
    )
    args = parser.parse_args()

    predict_image(args.image)

if __name__ == "__main__":
    main()