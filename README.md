Berikut adalah **README** dalam bahasa Indonesia untuk aplikasi Flask Anda, yang mencakup langkah-langkah instalasi, cara menggunakan Ngrok, dokumentasi API, dan cara menggunakannya.

---

# Aplikasi Flask - Klasifikasi Daun Alpukat

Aplikasi berbasis Flask ini digunakan untuk mengklasifikasikan daun alpukat menggunakan model yang sudah dilatih (TensorFlow Lite). Aplikasi ini menyediakan beberapa endpoint API untuk melakukan prediksi, mengelola riwayat prediksi, dan lainnya.

## Daftar Isi

1. [Instalasi](#instalasi)
2. [Setup Ngrok](#setup-ngrok)
3. [Dokumentasi API](#dokumentasi-api)
4. [Penggunaan](#penggunaan)
5. [Lisensi](#lisensi)

---

## Instalasi

### Prasyarat

Pastikan Anda memiliki Python 3.x yang terinstal. Anda bisa memeriksa versi Python yang terinstal dengan menjalankan:

```bash
python --version
```

### Langkah 1: Clone Repository

Pertama, clone repository ini ke lokal Anda dan masuk ke folder aplikasi:

```bash
cd nama_folder_aplikasi
```

### Langkah 2: Membuat Virtual Environment

Disarankan untuk membuat environment virtual guna mengelola dependensi:

```bash
python app.py
```

### Langkah 3: Menginstal Dependensi

Instal paket-paket Python yang dibutuhkan dari `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Langkah 4: Menyiapkan Model dan Database

Pastikan model (`model_comp.tflite`), label (`labels.txt`), dan deskripsi (`descriptions.json`) diletakkan di dalam direktori `assets/`.

Anda juga perlu menyiapkan database SQLite. Untuk melakukannya, jalankan perintah berikut untuk menginisialisasi database:

```bash
flask init-db
```

### Langkah 5: Menjalankan Aplikasi Flask

Sekarang, Anda dapat menjalankan aplikasi Flask secara lokal:

```bash
flask run
```

Aplikasi Flask Anda akan berjalan di `http://127.0.0.1:5000`.

---

## Setup Ngrok

Jika Anda ingin mengekspos server Flask lokal Anda ke internet (misalnya, untuk pengujian dengan layanan eksternal atau aplikasi mobile), Anda dapat menggunakan [Ngrok](https://ngrok.com/).

### Langkah 1: Download dan Instal Ngrok

1. Download Ngrok dari [sini](https://ngrok.com/download).
2. Ekstrak file yang diunduh.
3. Pada Windows, jalankan `ngrok.exe` di command prompt, dan pada Mac/Linux, jalankan `ngrok` di terminal.

### Langkah 2: Mengekspos Server Lokal

Mulai aplikasi Flask Anda, lalu buka terminal baru dan jalankan:

```bash
ngrok http 5000
```

Ngrok akan memberikan Anda URL publik (misalnya, `https://1234abcd.ngrok.io`). Anda bisa menggunakan URL ini untuk mengakses aplikasi Flask Anda dari mana saja.

### Langkah 3: Konfigurasi Domain Statis (Opsional)

Jika Anda ingin memiliki **URL statis** yang tetap sama setiap kali Anda menjalankan Ngrok, Anda bisa mendaftar dan menggunakan fitur **Ngrok Authtoken** untuk mendapatkan subdomain statis.

1. **Mendaftar Akun Ngrok**:
   Kunjungi [ngrok.com](https://ngrok.com/signup) dan daftar untuk akun gratis.

2. **Mendapatkan Authtoken**:
   Setelah mendaftar, Anda akan mendapatkan **Authtoken** yang dapat digunakan untuk menghubungkan Ngrok dengan akun Anda.

3. **Mengonfigurasi Authtoken**:
   Jalankan perintah berikut untuk menyetel **Authtoken** di Ngrok:

   ```bash
   ngrok authtoken <Your-Authtoken>
   ```

4. **Menjalankan Ngrok dengan Subdomain Statis**:
   Sekarang Anda dapat meminta Ngrok untuk memberikan **subdomain statis**. Jalankan perintah berikut untuk mengakses aplikasi Flask Anda menggunakan subdomain yang tetap (misalnya `myapp`):

   ```bash
   ngrok http -subdomain=myapp 5000
   ```

   Ngrok akan memberikan URL seperti ini:

   ```
   Forwarding                    https://myapp.ngrok.io -> http://localhost:5000
   ```

Dengan **subdomain statis**, URL Anda tidak akan berubah setiap kali Anda menjalankan Ngrok, sehingga lebih mudah untuk melakukan pengujian berkelanjutan atau integrasi dengan aplikasi eksternal.
---

## Dokumentasi API

### 1. **Endpoint Prediksi** (`/predict`)

**Metode**: `POST`
**Deskripsi**: Menerima gambar daun alpukat, memprosesnya, dan mengembalikan hasil klasifikasi.

**Request**:

* Content-Type: `multipart/form-data`
* Parameter file dengan nama `file` (gambar dalam format JPEG/PNG)

Contoh request body:

```bash
curl -X POST -F "file=@path_to_your_image.jpg" http://127.0.0.1:5000/predict
```

**Response**:

```json
{
  "prediction": "Avocado Type A",
  "confidence": 0.92,
  "description": {
    "nama_ilmiah": "Persea americana",
    "asal": "Meksiko",
    "karakteristik": "Daun berbentuk oval",
    "musim_panen": "Musim semi",
    "keunggulan": "Tahan terhadap penyakit"
  },
  "probabilities": {
    "Avocado Type A": 0.92,
    "Avocado Type B": 0.08
  },
  "saved_image": "unique_image_filename.jpg"
}
```

### 2. **Get Image** (`/images/<filename>`)

**Metode**: `GET`
**Deskripsi**: Mengambil gambar yang disimpan berdasarkan nama file.

**Request**:

```bash
curl http://127.0.0.1:5000/images/unique_image_filename.jpg
```

**Response**: File gambar.

### 3. **Get Prediction History** (`/history`)

**Metode**: `GET`
**Deskripsi**: Mengambil seluruh riwayat prediksi.

**Response**:

```json
{
  "history": [
    {
      "id": 1,
      "timestamp": "2023-07-21T12:34:56",
      "predicted_label": "Avocado Type A",
      "confidence": 0.92,
      "image_filename": "unique_image_filename.jpg"
    },
    ...
  ]
}
```

### 4. **Get Prediction History by ID** (`/history/<int:id>`)

**Metode**: `GET`
**Deskripsi**: Mengambil satu entri riwayat prediksi berdasarkan ID.

**Response**:

```json
{
  "id": 1,
  "timestamp": "2023-07-21T12:34:56",
  "predicted_label": "Avocado Type A",
  "confidence": 0.92,
  "image_filename": "unique_image_filename.jpg",
  "description": {
    "nama_ilmiah": "Persea americana",
    "asal": "Meksiko",
    "karakteristik": "Daun berbentuk oval",
    "musim_panen": "Musim semi",
    "keunggulan": "Tahan terhadap penyakit"
  }
}
```

### 5. **Delete All History** (`/history/delete_all`)

**Metode**: `POST`
**Deskripsi**: Menghapus seluruh riwayat prediksi dan gambar terkait.

**Response**:

```json
{
  "message": "Seluruh riwayat prediksi dan 5 file gambar berhasil dihapus."
}
```

### 6. **Delete History by ID** (`/history/delete/<int:id>`)

**Metode**: `DELETE`
**Deskripsi**: Menghapus satu entri riwayat prediksi berdasarkan ID.

**Response**:

```json
{
  "message": "Riwayat prediksi dengan ID 1 berhasil dihapus."
}
```

