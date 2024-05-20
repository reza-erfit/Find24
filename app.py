import streamlit as st
import pandas as pd
import numpy as np
import pickle
import qrcode
from PIL import Image
from datetime import datetime
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import gzip
import io

# Load model yang telah dilatih
with gzip.open('model.pkl.gz', 'rb') as f:
  model = pickle.load(f)
  
  
st.image('src\Marketing.png', caption='✨ Marketing Campaign - iris tentan ✨', use_column_width=True)


def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# Membuat QR code
buffer = generate_qr_code("https://iristentan-findit2024.streamlit.app/")

# Setup sidebar dengan styling yang lebih menarik
with st.sidebar:
    # QR Code di bagian atas
    st.image(buffer, caption='Scan QR Code for Our App', use_column_width=True)
    
    # Informasi kompetisi
    st.markdown("<h2 style='text-align: center; color: #FFA500;'>DATA ANALYTICS COMPETITION</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FFA500;'>FIND IT! 2024</h3>", unsafe_allow_html=True)
    
    # Nama tim
    st.markdown("<h3 style='text-align: center; color: #0066CC;'>Tim IRIS Tentan</h3>", unsafe_allow_html=True)
    
    # Logo Unair
    st.image("src\Logo UNAIR.png", caption='Universitas Airlangga', use_column_width=True)
    
    # Disusun oleh
    st.markdown("<h4 style='text-align: center;'>Disusun Oleh:</h4>", unsafe_allow_html=True)
    st.markdown("""
        <ul style='list-style-position: inside; text-align: left;'>
            <li>Netri Alia Rahmi</li>
            <li>Elzandi Irfan Zikra</li>
            <li>Muhammad Reza Erfit</li>
        </ul>
    """, unsafe_allow_html=True)


st.title('Form Input Data Pengguna')

with st.form("data_form"):
    st.subheader("Identitas Pengguna")
    tahun_kelahiran = st.number_input('Tahun Kelahiran', min_value=1900, max_value=2015, step=1)
    tanggal_menjadi_anggota = st.date_input('Tanggal Menjadi Anggota', min_value=datetime(1900, 1, 1))

    st.subheader("Data Kategorikal")
    pendidikan = st.selectbox('Pendidikan', ['SMP', 'SMA', 'Diploma', 'Sarjana', 'Magister', 'Doktor'])
    status_pernikahan = st.selectbox('Status Pernikahan', ['Sendiri', 'Rencana Menikah', 'Menikah', 'Cerai', 'Cerai Mati'])
    keluhan = st.selectbox(
        "Keluhan",
        options=[0, 1],
        format_func=lambda x: 'Tidak Ada Keluhan' if x == 0 else 'Ada Keluhan'
    )

    st.subheader("Data Numerik")
    pendapatan = st.number_input('Pendapatan', value=50000, format='%d')
    jumlah_anak_balita = st.slider('Jumlah Anak Balita', min_value=0, max_value=10, value=0)
    jumlah_anak_remaja = st.slider('Jumlah Anak Remaja', min_value=0, max_value=10, value=0)
    terakhir_belanja = st.number_input('Terakhir Belanja (hari)', value=0, format='%d')
    belanja_buah = st.number_input('Belanja Buah', value=0, format='%d')
    belanja_daging = st.number_input('Belanja Daging', value=0, format='%d')
    belanja_ikan = st.number_input('Belanja Ikan', value=0, format='%d')
    belanja_kue = st.number_input('Belanja Kue', value=0, format='%d')
    pembelian_diskon = st.number_input('Pembelian Diskon', value=0, format='%d')
    pembelian_web = st.number_input('Pembelian Web', value=0, format='%d')
    pembelian_toko = st.number_input('Pembelian Toko', value=0, format='%d')

    submitted = st.form_submit_button("Submit")
    
    
def interpret_prediction(pred, df):
    # age = 2015 - df['tahun_kelahiran'].iloc[0]
    income = df['pendapatan'].iloc[0]
    average_income = 114483200  # rata-rata pendapatan yang diberikan
    education_level = df['pendidikan'].iloc[0]
    has_teenage_children = df['jumlah_anak_remaja'].iloc[0] > 0
    has_toddlers = df['jumlah_anak_balita'].iloc[0] > 0

    interpretations = {
        0: "Pelanggan ini diperkirakan tidak mau menerima promosi sama sekali, menunjukkan kurangnya minat atau relevansi promosi.",
        1: f"Promosi pertama berhasil mencapai tingkat penerimaan sebesar 13.5%, efektif untuk pelanggan dengan pendidikan tinggi seperti {education_level}." +
           (f" Ini sangat cocok untuk keluarga dengan anak remaja." if has_teenage_children else ""),
        2: f"Promosi kedua, dengan penurunan penerimaan menjadi 10.2%, menunjukkan kurangnya daya tarik bagi pelanggan dengan pendapatan tinggi. Promosi ini berhasil menjangkau pelanggan dengan pendapatan bervariasi, termasuk mereka yang pendapatannya jauh di bawah rata-rata ({average_income:,}).",
        3: f"Promosi ketiga meningkatkan penerimaan menjadi 12.4%, menarik bagi berbagai usia, termasuk keluarga dengan anak remaja." +
           (f" Strategi ini cocok karena melibatkan keluarga dengan anak remaja, sesuai dengan profil customer ini." if has_teenage_children else ""),
        4: f"Promosi keempat mengembalikan tingkat penerimaan ke 13.5%, berhasil menarik kembali keluarga dengan anak remaja dan pelanggan dengan pendapatan tinggi, mirip dengan rata-rata pendapatan ({average_income:,})." +
           (f" Cocok untuk customer ini yang memiliki anak remaja dan pendapatan yang sebanding dengan atau lebih tinggi dari rata-rata." if has_teenage_children and income >= average_income else ""),
        5: f"Promosi kelima mencapai tingkat penerimaan tertinggi sebesar 14.6%, dengan penawaran yang menarik bagi pelanggan berpendapatan tinggi dan peningkatan pembelian di semua kategori produk." +
           (f" Sangat cocok untuk customer ini dengan pendapatan tinggi ({income:,}), memaksimalkan keuntungan dari investasi customer ini." if income >= average_income else ""),
        6: "Promosi keenam mengalami penurunan penerimaan menjadi 10.1%, mungkin karena kelelahan promosi atau kurangnya relevansi dengan kebutuhan pelanggan yang telah lama tidak berbelanja."
    }

    return interpretations.get(pred, "Terjadi kesalahan dalam prediksi, mohon periksa kembali data input atau hubungi administrator.")


    return interpretations.get(pred, "Terjadi kesalahan dalam prediksi, mohon periksa kembali data input atau hubungi administrator.")



if submitted:
    data = {
        'tahun_kelahiran': [tahun_kelahiran],
        'pendidikan': [pendidikan],
        'status_pernikahan': [status_pernikahan],
        'pendapatan': [pendapatan],
        'jumlah_anak_balita': [jumlah_anak_balita],
        'jumlah_anak_remaja': [jumlah_anak_remaja],
        'terakhir_belanja': [terakhir_belanja],
        'belanja_buah': [belanja_buah],
        'belanja_daging': [belanja_daging],
        'belanja_ikan': [belanja_ikan],
        'belanja_kue': [belanja_kue],
        'pembelian_diskon': [pembelian_diskon],
        'pembelian_web': [pembelian_web],
        'pembelian_toko': [pembelian_toko],
        'keluhan': [keluhan],
        'tanggal_menjadi_anggota': [tanggal_menjadi_anggota.strftime('%Y-%m-%d')]
    }
    df = pd.DataFrame(data)
    
    df['umur'] = 2015 - df['tahun_kelahiran']
    df = df.drop("tahun_kelahiran", axis=1)

    Q1 = df['umur'].quantile(0.25)
    Q3 = df['umur'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    outliers = df[(df['umur'] < lower_bound) | (df['umur'] > upper_bound)]

    df.loc[(df['umur'] < lower_bound) | (df['umur'] > upper_bound), 'umur'] = np.nan
    

    df['tanggal_menjadi_anggota'] = pd.to_datetime(df['tanggal_menjadi_anggota'], errors='coerce')

    current_date = pd.to_datetime('2014-07-01')

    df['keanggotaan'] = ((current_date.year - df['tanggal_menjadi_anggota'].dt.year) * 12 +
                                (current_date.month - df['tanggal_menjadi_anggota'].dt.month))

    df = df.drop("tanggal_menjadi_anggota", axis=1)
    df['pendidikan'] = df['pendidikan'].replace({'Magister': 'Pasca_Sarjana', 'Doktor': 'Pasca_Sarjana'})

    education_order = {
        'SMP': 0, 'SMA': 1, 'Sarjana': 2, 'Pasca_Sarjana': 3
    }
    
    status_order = {
        'Sendiri': 0, 'Rencana Menikah': 1, 'Menikah': 2, 'Cerai': 3, 'Cerai Mati': 4
    }
    df['pendidikan'] = df['pendidikan'].map(education_order)
    df['status_pernikahan'] = df['status_pernikahan'].map(status_order)
    columns_to_correct = ['belanja_kue', 'belanja_daging', 'belanja_ikan', 'belanja_buah']

    for column in columns_to_correct:
        df[column] = df[column].apply(lambda x: max(x, 0))
    drop = ['keanggotaan']
    df = df.drop(drop, axis=1)
    drop = ['keluhan']
    df = df.drop(drop, axis=1)
    df['jumlah_pembelian'] = df['pembelian_diskon'] + df['pembelian_web'] + df['pembelian_toko']
    df['total_belanja'] = df['belanja_daging'] + df['belanja_ikan'] + df['belanja_buah'] + df['belanja_kue']
    df["jumlah_anak"] = df['jumlah_anak_balita'] + df['jumlah_anak_remaja']

    # 8018
    df['winsorized_pendapatan'] = winsorize(df['pendapatan'], limits=[0.01, 0.01])
    # 8038
    df['log_pendapatan'] = np.log(df['pendapatan'] + 1)
    # 8097
    df['ranked_pendapatan'] = df['pendapatan'].rank(method='average')
    # 8101
    df['cbrt_pendapatan'] = np.cbrt(df['pendapatan'])
    # 8108
    df['rasio_pembelian_diskon'] = df['pembelian_diskon']/(df['jumlah_pembelian']+1)
    # 8115
    df['proporsi_kue'] = df['belanja_kue'] / (df['belanja_buah'] + df['belanja_daging'] + df['belanja_ikan'] + df['belanja_kue'] + 1)
    # 8165
    df['pendapatan_kelompok_umur'] = df['winsorized_pendapatan'] / df['umur']
    # 8161
    df['log_shopping_frequency'] = np.log(df['total_belanja']+1)
    # 8140
    df['proporsi_buah'] = df['belanja_buah'] / (df['belanja_buah'] + df['belanja_daging'] + df['belanja_ikan'] + df['belanja_kue'] + 1)
    # 8181
    df['rasio_belanja_ikan'] = df['belanja_ikan'] / df['pendapatan']
    # 8181
    df['rasio_belanja_kue'] = df['belanja_kue'] / df['pendapatan']
    # 8212
    df['total_pembelian_toko_web_pendapatan'] = (df['pembelian_toko'] + df['pembelian_web']) / (df['pendapatan'] + 1)
    # 8199
    df['pendapatan_adjusted_umur_pendidikan'] = df['pendapatan'] / (df['umur'] * (df['pendidikan'].astype(int) + 1))
    # 8205
    df['rasio_pembelian_toko'] = df['pembelian_toko']/(df['jumlah_pembelian']+1)
    # 8214
    df['income_stability_score'] = df['pendapatan'] / (df[['belanja_buah', 'belanja_daging', 'belanja_ikan', 'belanja_kue']].std(axis=1) + 1)
    # 8204
    mean_income = np.log(df['pendapatan'].mean() + 1)
    df['log_pendapatan_vs_mean'] = np.log(df['pendapatan'] + 1) - mean_income
    # 8232
    df['log_age'] = np.log((df['umur']))
    drop = ['jumlah_pembelian', 'total_belanja', 'jumlah_anak']
    df = df.drop(drop, axis=1)
    
    #Prediksi
    prediksi = model.predict(df)[0]
    interpretasi = interpret_prediction(prediksi, df)

    # Menampilkan hasil prediksi dengan tampilan yang lebih menarik
    st.markdown(f"""
        <div style="padding: 10px; border-radius: 10px; background-color: #f0f0f5; border: 1px solid #ccc;">
            <h4 style="color: #333;">Hasil Prediksi (Nomor Promosi): <span style="color: #0073e6;">{prediksi}</span></h4>
            <p style="color: #555;">{interpretasi}</p>
        </div>
    """, unsafe_allow_html=True)
