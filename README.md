# Tugas Besar 2 IF3270 Pembelajaran Mesin - Convolutional Neural Network dan Recurrent Neural Network

## Deskripsi
Repository ini berisi implementasi model Feedforward Neural Network (FFNN) untuk Tugas Besar 1 IF3270 Pembelajaran Mesin. Proyek ini mencakup eksperimen dan analisis berbagai teknik, seperti regularisasi (L1 dan L2) serta RMS Norm, untuk meningkatkan performa model. Selain kode sumber Python, repository ini juga menyediakan file Jupyter Notebook (.ipynb) untuk eksplorasi interaktif serta folder `doc` yang berisi laporan lengkap.

Repository ini berisi implementasi model Convolutional Neural Network (CNN) dan Recurrent Neural Network (RNN) untuk Tugas Besar 2 IF3270 Pembelajaran Mesinp. Fokus tugas ini adalah mengimplementasikan fungsi forward propagation secara from scratch untuk model CNN, RNN, dan Long-Short Term Memory (LSTM). Selain itu, terdapat juga pengujian-pengujian pada masing-masing model
dari segi jumlah layer, ukuran layer, bidirectional atau unidirectional, dan lain-lain

## Cara Setup dan Menjalankan Program
1. **Clone Repository:**
   ```bash
   git clone https://github.com/wegeh/Tubes-3-IF3270.git
   cd Tubes-2-IF3270
   ```

2. **Setup Environment (virtual environment pada Python):**
   ```bash
    python -m venv env
    source env/bin/activate   # Untuk Linux/MacOS
    env\Scripts\activate   # Untuk Windows
   ```

3. **Move to directory you want to see (choose 1):**
    ```bash
    cd src/cnn 
    cd src/rnn
    cd src/lstm  
    ```

4. **Install Dependencies: Pastikan Anda telah menginstal pip. Kemudian, jalankan:**
   ```bash
    pip install -r requirements.txt
   ```

5. **Menjalankan Program: Untuk menjalankan program, jalankan file Python yang diinginkan:**
   ```bash
    python 1<nama_file_python.py>
   ```
    Anda juga dapat membuka dan menjalankan file Jupyter Notebook (.ipynb) yang tersedia untuk eksplorasi lebih lanjut.

## Struktur Repository
- **folder src**: Berisi kode sumber Python untuk implementasi FFNN dan eksperimen.
- **folder doc**: Berisi dokumen laporan lengkap.
- **requirements.txt**: Daftar dependencies yang diperlukan.
- **README.md**: Penjelasan mengenai tugas dan dokumentasi.

## Pembagian Tugas (Kelompok 37)
- **Filbert (13522021)**
  - **Kontribusi**: Laporan, CNN, RNN, LSTM.
- **Benardo (13522055)**
  - **Kontribusi**: Laporan, CNN, RNN, LSTM.
- **William Glory Henderson (13522113)**
  - **Kontribusi**: Laporan, CNN, RNN, LSTM.