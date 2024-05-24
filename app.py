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

# Load model
with gzip.open('model.pkl.gz', 'rb') as f:
  model = pickle.load(f)
  
  
st.image('src/Marketing.png', caption='✨ Marketing Campaign - iris tentan ✨', use_column_width=True)

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

buffer = generate_qr_code("https://iristentan-findit2024.streamlit.app/")

# Setup sidebar dengan styling yang lebih menarik
with st.sidebar:

    st.image(buffer, caption='Scan QR Code for Our App', use_column_width=True)
    
    st.markdown("<h2 style='text-align: center; color: #FFA500;'>DATA ANALYTICS COMPETITION</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FFA500;'>FIND IT! 2024</h3>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: #0066CC;'>Tim IRIS Tentan</h3>", unsafe_allow_html=True)
    
    st.image("src/Logo UNAIR.png", use_column_width=True)
    
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
    tahun_kelahiran = st.number_input('Tahun Kelahiran', min_value=1900, max_value=2015, step=1, value=1970)
    tanggal_menjadi_anggota = st.date_input('Tanggal Menjadi Anggota', value=datetime(2012, 9, 14), min_value=datetime(1900, 1, 1))

    st.subheader("Data Kategorikal")
    pendidikan = st.selectbox('Pendidikan', ['SMP', 'SMA', 'Diploma', 'Sarjana', 'Magister', 'Doktor'], index=5)
    status_pernikahan = st.selectbox('Status Pernikahan', ['Sendiri', 'Rencana Menikah', 'Menikah', 'Cerai', 'Cerai Mati'], index=0)
    keluhan = st.selectbox(
        "Keluhan",
        options=[0, 1],
        format_func=lambda x: 'Tidak Ada Keluhan' if x == 0 else 'Ada Keluhan',
        index=0
    )

    st.subheader("Data Numerik")
    pendapatan = st.number_input('Pendapatan', value=172047000, format='%d')
    jumlah_anak_balita = st.slider('Jumlah Anak Balita', min_value=0, max_value=10, value=0)
    jumlah_anak_remaja = st.slider('Jumlah Anak Remaja', min_value=0, max_value=10, value=0)
    terakhir_belanja = st.number_input('Terakhir Belanja (hari)', value=97, format='%d')
    belanja_buah = st.number_input('Belanja Buah', value=204323, format='%d')
    belanja_daging = st.number_input('Belanja Daging', value=1687182, format='%d')
    belanja_ikan = st.number_input('Belanja Ikan', value=265013, format='%d')
    belanja_kue = st.number_input('Belanja Kue', value=210391, format='%d')
    pembelian_diskon = st.number_input('Pembelian Diskon', value=0, format='%d')
    pembelian_web = st.number_input('Pembelian Web', value=6, format='%d')
    pembelian_toko = st.number_input('Pembelian Toko', value=12, format='%d')

    submitted = st.form_submit_button("Submit")
    
    
def interpret_prediction(pred, df):
    if pred == 0:
        return f"""
    <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border: 2px solid #e57373; margin-top: 10px;">
        <h2 style="color: #d32f2f;">🚫 Tidak Menerima Promosi:</h2>
        <h4><strong>Strategi:</strong></h4>
        <ol>
            <li><strong>Penyesuaian Waktu Promosi:</strong> Mengingat frekuensi belanja yang bervariasi, promosi dapat dilakukan pada waktu yang lebih bervariasi untuk menjangkau pelanggan yang berbelanja pada waktu yang berbeda.
            Contoh: "Promo akhir pekan spesial: Dapatkan <strong>diskon 10%</strong> untuk semua produk daging dan ikan."</li>
            <li><strong>Segmentasi Berdasarkan Pendidikan:</strong> Meningkatkan jangkauan promosi kepada pelanggan dengan pendidikan lebih rendah melalui komunikasi yang lebih sederhana dan langsung.
            Contoh: "Promo hemat keluarga: Beli daging dan ikan dengan harga terjangkau."</li>
            <li><strong>Peningkatan Akses Informasi:</strong> Menggunakan berbagai saluran komunikasi, termasuk media sosial, email, dan SMS untuk memastikan semua segmen pelanggan menerima informasi promosi.
            Contoh: "Dapatkan penawaran spesial langsung di ponsel Anda. Daftar untuk menerima notifikasi promo."</li>
            <li><strong>Program Loyalitas dan Insentif:</strong> Mengembangkan program loyalitas yang menawarkan insentif bagi pelanggan yang belum pernah menerima promosi sebelumnya.
            Contoh: "Bergabunglah dengan program loyalitas kami dan dapatkan <strong>diskon 5%</strong> untuk pembelian pertama Anda."</li>
            <li><strong>Flash Sale/Diskon:</strong> Pengurangan harga dalam periode waktu tertentu agar dapat meningkatkan minat belanja/pembelian pelanggan.
            Contoh: "Nikmati diskon kilat hingga <strong>20%</strong> hanya dalam 2 jam berikutnya."</li>
        </ol>
    </div>
    """
    elif pred == 1:
        return f"""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; border: 2px solid #90caf9; margin-top: 10px;">
            <h2 style="color: #1976d2;">🎯 <strong>Strategi Promosi</strong></h2>
            <ol>
                <li><strong>Bundling Barang Esensial:</strong> Mengingat pengeluaran yang sederhana untuk daging dan ikan, promosi bundling untuk barang-barang ini dengan diskon kecil bisa efektif. Misalnya, "Beli daging dan ikan, dapatkan diskon hingga <strong>20,000 IDR</strong>."</li>
                <li><strong>Diskon Pembelian Grosir:</strong> Karena sebagian besar pelanggan sensitif terhadap harga, menawarkan diskon untuk pembelian dalam jumlah besar atau kuantitas yang lebih besar bisa meningkatkan penjualan. Contohnya, "Beli 3kg daging atau ikan dan dapatkan <strong>diskon 10%</strong>."</li>
                <li><strong>Targeted Communication:</strong> Fokus pada pelanggan yang sudah menikah dengan pendapatan menengah, berkomunikasilah dengan menekankan nilai dan penghematan dalam promosi. Misalnya, menekankan bahwa promosi ini memberikan nilai lebih untuk setiap pembelian, sehingga menarik bagi mereka yang mencari harga terbaik.</li>
            </ol>
            <h3 style="color: #1976d2;">👥 <strong>Profil Pelanggan:</strong></h3>
            <ul>
                <li>Termasuk kelompok usia yang cenderung stabil dalam hal pengeluaran dan memiliki penghasilan yang cukup untuk berbelanja karena berusia produktif.</li>
                <li>Sebagian besar sudah menikah, dan memiliki setidaknya gelar sarjana. Promosi yang berfokus pada kebutuhan rumah tangga dan pasangan akan lebih menarik.</li>
                <li>Tingkat pendapatan mereka kebanyakan di bawah rata-rata, menunjukkan sensitivitas terhadap harga dan promosi.</li>
                <li>Sebagian besar pelanggan tidak memiliki anak balita atau remaja, menunjukkan bahwa promosi yang berfokus pada paket keluarga atau barang terkait anak mungkin tidak efektif.</li>
                <li>Pelanggan memiliki kebiasaan belanja yang tidak terlalu sering.</li>
            </ul>
        </div>
        """
    elif pred == 2:
        return f"""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border: 2px solid #a5d6a7; margin-top: 10px;">
            <h2 style="color: #2e7d32;">🎯 <strong>Strategi Promosi</strong></h2>
            <ol>
                <li><strong>Promosi Produk Premium:</strong> Dengan tingkat pendapatan menengah hingga tinggi, promosi untuk produk premium dapat diterapkan. Misalnya, "Dapatkan daging organik dan ikan segar premium dengan <strong>diskon 15%</strong> untuk pembelian di atas 500,000 IDR."</li>
                <li><strong>Loyalty Program:</strong> Mengingat usia dan status pernikahan pelanggan, program loyalitas yang menawarkan diskon tambahan atau hadiah untuk pembelian rutin dapat meningkatkan retensi pelanggan. Contohnya, "Bergabunglah dengan program loyalitas kami dan dapatkan poin untuk setiap pembelian yang bisa ditukar dengan voucher belanja."</li>
                <li><strong>Paket Hemat untuk Pasangan:</strong> Fokus pada kebutuhan pasangan atau rumah tangga tanpa anak balita dapat ditingkatkan dengan promosi paket hemat. Misalnya, "Paket belanja mingguan: Beli daging dan ikan, dapatkan buah gratis untuk pasangan, hanya dengan 250,000 IDR."</li>
                <li><strong>Promosi Kombinasi Produk:</strong> Mengingat preferensi pengeluaran yang lebih tinggi untuk daging dan ikan, menawarkan promosi kombinasi produk dapat menarik perhatian pelanggan. Contoh, "Beli 2kg daging dan 1kg ikan, dapatkan 10% diskon dan gratis buah segar."</li>
            </ol>
            <h3 style="color: #2e7d32;">👥 <strong>Profil Pelanggan:</strong></h3>
            <ul>
                <li>Kelompok pelanggan ini cenderung berada di usia mapan secara finansial dan cenderung setia pada merek tertentu.</li>
                <li>Sebagian besar sudah menikah atau merencanakan pernikahan, dan memiliki setidaknya gelar sarjana. Tingkat pendidikan yang tinggi ini menunjukkan bahwa mereka cenderung mencari produk berkualitas dengan nilai tambah yang jelas dan menunjukkan kebutuhan akan produk yang mendukung kehidupan keluarga dan pasangan.</li>
                <li>Pelanggan tidak memiliki anak balita atau remaja, dengan sebagian kecil yang memiliki satu anak. Hal ini mengisyaratkan bahwa promosi yang berfokus pada kebutuhan rumah tangga yang tidak terkait dengan anak-anak mungkin lebih efektif.</li>
            </ul>
        </div>
        """
    elif pred == 3:
        return f"""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; border: 2px solid #ffcc80; margin-top: 10px;">
            <h2 style="color: #ffa726;">🎯 <strong>Strategi Promosi</strong></h2>
            <ol>
                <li><strong>Promosi Produk Kesehatan:</strong> Mengingat usia pelanggan, promosi produk yang berfokus pada kesehatan dan kebugaran bisa sangat efektif. Misalnya, "Dapatkan <strong>diskon 20%</strong> untuk produk daging rendah lemak dan ikan kaya omega-3."</li>
                <li><strong>Paket Langganan Bulanan:</strong> Menawarkan paket langganan bulanan dengan harga khusus dapat menarik pelanggan yang lebih tua dan mencari kenyamanan. Contohnya, "Berlangganan paket daging dan ikan bulanan dan <strong>hemat hingga 15%</strong> setiap bulan."</li>
                <li><strong>Promosi Khusus untuk Pasangan:</strong> Fokus pada kebutuhan pasangan dengan menawarkan promosi khusus untuk pembelian bersama. Misalnya, "Beli daging dan ikan untuk dua orang dan dapatkan buah gratis untuk pasangan Anda."</li>
                <li><strong>Program Diskon untuk Pelanggan Loyal:</strong> Membuat program diskon untuk pelanggan yang sering berbelanja dapat meningkatkan retensi pelanggan. Misalnya, "Dapatkan poin untuk setiap pembelian dan tukarkan dengan diskon pada kunjungan berikutnya."</li>
            </ol>
            <h3 style="color: #ffa726;">👥 <strong>Profil Pelanggan:</strong></h3>
            <ul>
                <li>Pelanggan berusia menengah hingga tua (50-70 tahun) yang cenderung lebih loyal dan memiliki kebiasaan belanja yang mapan.</li>
                <li>Pelanggan memiliki setidaknya gelar sarjana, mengindikasikan pelanggan yang cenderung mengutamakan kualitas dan memiliki daya beli yang baik.</li>
                <li>Mayoritas pelanggan sudah menikah, diikuti oleh yang merencanakan pernikahan dan yang masih lajang. Ini menunjukkan bahwa promosi yang berfokus pada kebutuhan keluarga dan pasangan akan lebih efektif.</li>
                <li>Sebagian besar pelanggan tidak memiliki anak balita atau remaja, sehingga promosi yang berfokus pada kebutuhan rumah tangga dewasa lebih sesuai.</li>
                <li>Pelanggan cenderung mengeluarkan uang lebih banyak untuk daging dan ikan dibandingkan buah, menunjukkan preferensi untuk protein hewani dalam diet mereka.</li>
                <li>Frekuensi belanja menunjukkan bahwa mereka melakukan belanja secara berkala dan terencana.</li>
            </ul>
        </div> """
        
    elif pred == 4:
        return f"""
        <div style="background-color: #e1f5fe; padding: 20px; border-radius: 10px; border: 2px solid #81d4fa; margin-top: 10px;">
            <h2 style="color: #0288d1;">🎯  <strong>Strategi Promosi</strong></h2>
            <ol>
                <li><strong>Promosi Produk Ramah Lingkungan:</strong> Mengingat tingkat pendidikan dan pendapatan yang tinggi, pelanggan mungkin tertarik pada produk ramah lingkungan dan berkelanjutan. Contohnya, "Dapatkan <strong>diskon 20%</strong> untuk daging dan ikan organik yang sustainable."</li>
                <li><strong>Paket Kesehatan dan Kebugaran:</strong> Menawarkan paket kesehatan yang berisi produk-produk rendah lemak dan kaya nutrisi dapat menarik minat pelanggan yang lebih tua. Misalnya, "Paket sehat bulanan: Dapatkan daging rendah lemak dan ikan kaya omega-3 dengan harga spesial."</li>
                <li><strong>Program Keanggotaan Eksklusif:</strong> Membuat program keanggotaan eksklusif yang menawarkan diskon tambahan dan akses prioritas ke produk baru bisa meningkatkan loyalitas pelanggan. Contohnya, "Bergabunglah dengan keanggotaan premium kami dan nikmati diskon 10% setiap bulan serta akses eksklusif ke produk baru."</li>
                <li><strong>Promosi Festival atau Acara Khusus:</strong> Mengingat pelanggan sering berbelanja dalam periode tertentu, promosi yang terkait dengan festival atau acara khusus bisa menarik perhatian mereka. Contohnya, "Rayakan Hari Keluarga dengan diskon 15% untuk semua pembelian daging dan ikan selama bulan ini."</li>
            </ol>
            <h3 style="color: #0288d1;">👥 <strong>Profil Pelanggan Promosi ke-4:</strong></h3>
            <ul>
                <li>Pelanggan cenderung memiliki preferensi belanja yang stabil dan teratur.</li>
                <li>Mayoritas pelanggan memiliki gelar sarjana, diikuti oleh magister dan SMA. Tingkat pendidikan yang tinggi menunjukkan bahwa mereka cenderung mencari produk berkualitas tinggi dan memiliki daya beli yang baik.</li>
                <li>Mayoritas pelanggan sudah menikah atau merencanakan untuk menikah, sementara yang masih lajang dan bercerai berada di posisi berikutnya. Ini menunjukkan bahwa promosi dapat mencakup hal-hal yang luas.</li>
                <li>Sebagian besar pelanggan tidak memiliki anak balita atau remaja, tetapi ada kelompok yang memiliki satu anak balita. Promosi yang menargetkan kebutuhan rumah tangga dan produk balita mungkin lebih relevan.</li>
                <li>Distribusi pendapatan menunjukkan variasi yang cukup besar dengan sebagian besar berada pada pendapatan menengah hingga tinggi. Ini mengindikasikan adanya potensi untuk promosi produk premium dengan harga yang sesuai.</li>
                <li>Pelanggan cenderung mengeluarkan uang lebih banyak untuk daging dan ikan dibandingkan buah, menunjukkan preferensi untuk protein hewani dalam diet mereka.</li>
                <li>Frekuensi belanja menunjukkan bahwa mereka melakukan belanja secara berkala dan terencana.</li>
            </ul>
        </div>
    
        """
        
    elif pred == 5:
        return f"""
    <div style="background-color: #fce4ec; padding: 20px; border-radius: 10px; border: 2px solid #f8bbd0; margin-top: 10px;">
        <h2 style="color: #ec407a;">🎯  <strong>Strategi Promosi</strong></h2>
        <ol>
            <li><strong>Promosi Produk Gourmet:</strong> Mengingat preferensi pelanggan untuk produk premium, promosi produk gourmet bisa sangat efektif. Contohnya, "Dapatkan <strong>diskon 20%</strong> untuk daging wagyu dan ikan salmon pada pembelian di atas 500,000 IDR."</li>
            <li><strong>Paket Kebutuhan Rumah Tangga:</strong> Menawarkan paket kebutuhan rumah tangga yang berisi daging, ikan, dan buah-buahan dengan harga khusus dapat menarik minat pelanggan yang lebih tua. Misalnya, "Paket kebutuhan rumah tangga bulanan: Dapatkan kebutuhan daging, ikan, dan buah dengan <strong>diskon 15%</strong>."</li>
            <li><strong>Promosi Eksklusif untuk Pasangan:</strong> Fokus pada kebutuhan pasangan dengan menawarkan promosi eksklusif. Misalnya, "Beli dua paket daging dan ikan, dapatkan satu paket buah <strong>gratis</strong> untuk pasangan Anda."</li>
            <li><strong>Program Penghargaan untuk Loyalitas:</strong> Membuat program penghargaan yang menawarkan diskon tambahan dan hadiah untuk pelanggan setia bisa meningkatkan retensi pelanggan. Contohnya, "Bergabunglah dengan program penghargaan kami dan dapatkan poin untuk setiap pembelian yang bisa ditukar dengan <strong>voucher belanja</strong>."</li>
        </ol>
        <h3 style="color: #ec407a;">👥 <strong>Profil Pelanggan Promosi ke-5:</strong></h3>
        <ul>
            <li>Pelanggan memiliki preferensi belanja stabil dan rutin.</li>
            <li>Tingkat pendidikan tinggi menunjukkan pelanggan yang mengutamakan kualitas dan memiliki daya beli yang cukup baik.</li>
            <li>Mayoritas pelanggan sudah menikah atau merencanakan pernikahan, sementara yang masih lajang dan bercerai berada di posisi berikutnya. Promosi yang berfokus pada kebutuhan rumah tangga dan pasangan akan lebih efektif.</li>
            <li>Distribusi pendapatan menunjukkan variasi yang cukup besar dengan sebagian besar berada pada pendapatan menengah hingga tinggi. Ini menunjukkan potensi untuk promosi produk premium dengan harga yang sesuai.</li>
            <li>Pengeluaran untuk buah cenderung menengah, dengan banyak pelanggan menghabiskan antara 100,000 hingga 300,000 IDR.</li>
            <li>Pengeluaran untuk daging cukup tinggi, dengan banyak pelanggan menghabiskan lebih dari 200,000 IDR.</li>
            <li>Pola pengeluaran ikan menunjukkan variasi yang cukup besar, dengan pengeluaran yang cukup tinggi hingga 300,000 IDR.</li>
        </ul>
    </div>
    """
    elif pred == 6:
        return f"""
    <div style="background-color: #fff8e1; padding: 20px; border-radius: 10px; border: 2px solid #ffecb3; margin-top: 10px;">
        <h2 style="color: #ffb300;">🎯 <strong>Strategi Promosi</strong></h2>
        <ol>
            <li><strong>Paket Berlangganan Kesehatan:</strong> Menawarkan paket berlangganan bulanan dengan produk-produk sehat seperti daging rendah lemak dan ikan omega-3 bisa sangat menarik. Contohnya, "Berlangganan paket sehat bulanan dan dapatkan <strong>diskon 10%</strong> untuk daging dan ikan setiap bulan."</li>
            <li><strong>Promosi Khusus Akhir Pekan:</strong> Mengingat banyak pelanggan yang mungkin melakukan belanja pada akhir pekan, tawarkan promosi khusus pada hari-hari tersebut. Misalnya, "Nikmati <strong>diskon 15%</strong> untuk semua pembelian daging dan ikan setiap Sabtu dan Minggu."</li>
            <li><strong>Paket Kombinasi untuk Keluarga:</strong> Meski sebagian besar pelanggan tidak memiliki anak balita atau remaja, paket kombinasi yang mencakup kebutuhan seluruh keluarga bisa efektif. Contohnya, "Paket keluarga: Beli 2kg daging dan 1kg ikan, dapatkan buah segar gratis."</li>
            <li><strong>Voucher Diskon untuk Pembelian Selanjutnya:</strong> Memberikan voucher diskon untuk pembelian selanjutnya bisa mendorong pelanggan untuk kembali berbelanja. Misalnya, "Belanja sekarang dan dapatkan <strong>voucher diskon 10%</strong> untuk pembelian berikutnya."</li>
        </ol>
        <h3 style="color: #ffb300;">👥 <strong>Profil Pelanggan:</strong></h3>
        <ul>
            <li>Pelanggan cenderung mengeluarkan uang lebih banyak untuk daging dan ikan dibandingkan buah, menunjukkan preferensi untuk protein hewani dalam diet mereka.</li>
            <li>Frekuensi belanja menunjukkan bahwa mereka melakukan belanja secara berkala dan terencana.</li>
            <li>Pelanggan berusia menengah hingga tua (50-70 tahun), sebagian besar sudah menikah atau merencanakan pernikahan, dan memiliki setidaknya gelar sarjana.</li>
            <li>Tingkat pendapatan mereka bervariasi dari menengah hingga tinggi, menunjukkan kemampuan untuk membeli produk premium.</li>
            <li>Sebagian besar pelanggan tidak memiliki anak balita atau remaja, sehingga promosi yang berfokus pada kebutuhan rumah tangga dewasa lebih sesuai.</li>
        </ul>
    </div>
    """

    else:
        return "<div style='background-color: #fafafa; padding: 20px; border-radius: 10px; border: 1px solid #ccc; margin-top: 10px;'><h4 style='color: #555;'>Terjadi kesalahan dalam prediksi, mohon periksa kembali data input atau hubungi administrator.</h4></div>"

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

    prediksi = model.predict(df)[0]
    st.markdown(f"<div style='background-color: #bbdefb; padding: 20px; border-radius: 10px; border: 1px solid #90caf9;'><h3 style='color: #333;'>Hasil Prediksi (Nomor Promosi): <span style='color: #2196f3;'>{prediksi}</span></h3></div>", unsafe_allow_html=True)
    interpretasi = interpret_prediction(prediksi, df)
    st.markdown(interpretasi, unsafe_allow_html=True)

# if __name__ == '__main__':
#     main()