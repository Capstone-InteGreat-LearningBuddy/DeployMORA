# 1. Gunakan Python 3.9 (Versi paling stabil untuk Scikit-Learn & Pandas)
FROM python:3.9

# 2. Set folder kerja di dalam container (Virtual Computer)
WORKDIR /code

# 3. Copy file requirements.txt terlebih dahulu
# (Tujuannya agar Docker bisa 'cache' proses install library, biar cepat kalau deploy ulang)
COPY ./requirements.txt /code/requirements.txt

# 4. Install library yang ada di requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy seluruh sisa file proyek (folder app, model_artifacts, main.py, dll) ke dalam container
COPY . /code

# 6. Atur izin (Permissions)
# Hugging Face menjalankan aplikasi sebagai user 'non-root' (user ID 1000).
# Kita harus memberi izin akses ke folder cache agar aplikasi tidak error saat menulis file sementara.
RUN mkdir -p /code/cache
RUN chmod -R 777 /code

# Set Environment Variable untuk Cache (biar library ML gak bingung nyimpan cache dimana)
ENV XDG_CACHE_HOME=/code/cache

# 7. Perintah Menyalakan Server
# PENTING: Hugging Face WAJIB menggunakan port 7860. Jangan diganti ke 8000.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]