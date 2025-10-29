import streamlit as st
import pandas as pd
import pickle
import statsmodels.api as sm
import plotly.graph_objects as go

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Perdagangan Indonesia",
    page_icon="📈",
    layout="wide"
)

# Fungsi Caching untuk Memuat Model
@st.cache_resource
def load_model(path):
    """Memuat model dari file .pkl"""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: File model tidak ditemukan di {path}")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Fungsi Caching untuk Memuat Data Historis
@st.cache_data
def load_historical_data(path_csv):
    """Memuat dan memproses data CSV historis."""
    try:
        df_raw = pd.read_csv(path_csv)
        
        # Lakukan preprocessing yang SAMA PERSIS dengan di Colab
        df_processed = df_raw.copy()
        df_processed['Neraca_Perdagangan'] = df_processed['Total_Ekspor'] - df_processed['Total_Impor']
        
        month_map = {
            'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
            'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
            'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
        }
        df_processed['Bulan_Angka'] = df_processed['Bulan'].map(month_map)
        
        df_processed['Tanggal'] = pd.to_datetime({'year': df_processed['Tahun'], 'month': df_processed['Bulan_Angka'], 'day': 1})
        df_processed.set_index('Tanggal', inplace=True)
        df_processed.sort_index(inplace=True)
        
        # Pilih hanya kolom yang kita butuhkan
        target_cols = ['Total_Ekspor', 'Total_Impor', 'Neraca_Perdagangan']
        df_clean = df_processed[target_cols]
        
        return df_clean
        
    except FileNotFoundError:
        st.error(f"Error: File data historis '{path_csv}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat data historis: {e}")
        return None

# Judul Aplikasi
st.title('📈 Aplikasi Prediksi Perdagangan Indonesia')
st.markdown("Deployment model SARIMA untuk memprediksi Ekspor, Impor, dan Neraca Perdagangan.")

# Muat data historis di awal
historical_df = load_historical_data('Data_Gabungan_Ekspor_Impor_Tahun2012-2025.csv')

# Sidebar untuk Input Pengguna
st.sidebar.header('Panel Kontrol')

bulan_prediksi = st.sidebar.slider(
    'Pilih jumlah bulan untuk diprediksi:',
    min_value=6,
    max_value=36,
    value=12,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info("Model ini dilatih pada data historis hingga **Agustus 2025**.")

# Tombol untuk Menjalankan Prediksi
if st.sidebar.button('Jalankan Prediksi'):
    
    # Cek apakah data historis berhasil dimuat
    if historical_df is not None:
        # Muat Model
        with st.spinner('Memuat 3 model SARIMA...'):
            model_ekspor = load_model('model_ekspor_final.pkl')
            model_impor = load_model('model_impor_final.pkl')
            model_neraca = load_model('model_neraca_final.pkl')

        if model_ekspor and model_impor and model_neraca:
            st.success('Ketiga model berhasil dimuat!')
            
            # Buat Prediksi (Forecast)
            with st.spinner(f'Menjalankan prediksi untuk {bulan_prediksi} bulan ke depan...'):
                pred_ekspor = model_ekspor.forecast(steps=bulan_prediksi)
                pred_impor = model_impor.forecast(steps=bulan_prediksi)
                pred_neraca = model_neraca.forecast(steps=bulan_prediksi)
                
                forecast_df = pd.DataFrame({
                    'Total_Ekspor_Prediksi': pred_ekspor,
                    'Total_Impor_Prediksi': pred_impor,
                    'Neraca_Perdagangan_Prediksi': pred_neraca
                })
                forecast_df.index.name = 'Tanggal'

            st.success(f'Prediksi {bulan_prediksi} bulan selesai!')

            # Tampilkan Hasil (Tabel)
            st.subheader(f'Tabel Prediksi untuk {bulan_prediksi} Bulan ke Depan')
            st.dataframe(
                forecast_df,
                use_container_width=True,
                column_config={
                    "Total_Ekspor_Prediksi": st.column_config.NumberColumn(format="%.2f Juta USD"),
                    "Total_Impor_Prediksi": st.column_config.NumberColumn(format="%.2f Juta USD"),
                    "Neraca_Perdagangan_Prediksi": st.column_config.NumberColumn(format="%.2f Juta USD"),
                }
            )
            
            # Tampilkan Grafik (Data Aktual vs Prediksi)
            st.subheader('Grafik Hasil Prediksi (vs. Data Aktual)')
            
            # Ambil data historis 5 tahun terakhir untuk plot
            history_to_plot = historical_df['2020':]
            
            # Plot 1: Ekspor vs Impor
            fig1 = go.Figure()
            # Plot Total Ekspor (Biru)
            fig1.add_trace(go.Scatter(x=history_to_plot.index, y=history_to_plot['Total_Ekspor'], mode='lines', name='Ekspor Aktual', line=dict(color='blue', dash='solid')))
            fig1.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Total_Ekspor_Prediksi'], mode='lines', name='Ekspor Prediksi', line=dict(color='blue', dash='dash')))
            # Plot Total Impor (Merah)
            fig1.add_trace(go.Scatter(x=history_to_plot.index, y=history_to_plot['Total_Impor'], mode='lines', name='Impor Aktual', line=dict(color='red', dash='solid')))
            fig1.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Total_Impor_Prediksi'], mode='lines', name='Impor Prediksi', line=dict(color='red', dash='dash')))
            
            fig1.update_layout(
                title='Prediksi Ekspor (Biru) vs Impor (Merah)',
                yaxis_title='Juta USD',
                legend_title='Keterangan'
            )
            st.plotly_chart(fig1, use_container_width=True) # Tampilkan plot Plotly

            # Plot 2: Neraca Perdagangan
            fig2 = go.Figure()
            # Plot Neraca Perdagangan (Hijau)
            fig2.add_trace(go.Scatter(x=history_to_plot.index, y=history_to_plot['Neraca_Perdagangan'], mode='lines', name='Neraca Aktual', line=dict(color='green', dash='solid')))
            fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Neraca_Perdagangan_Prediksi'], mode='lines', name='Neraca Prediksi', line=dict(color='green', dash='dash')))
            
            fig2.add_hline(y=0, line_dash="dot", line_color="black") # Garis nol
            
            fig2.update_layout(
                title='Prediksi Neraca Perdagangan (Hijau)',
                yaxis_title='Juta USD',
                legend_title='Keterangan'
            )
            st.plotly_chart(fig2, use_container_width=True) # Tampilkan plot Plotly

else:
    # Tampilkan ini jika data CSV tidak ditemukan
    if historical_df is None:
        st.error("Gagal memuat data historis. Pastikan file 'Data_Gabungan_Ekspor_Impor_Tahun2012-2025.csv' ada di folder aplikasi.")
    else:
        st.info('Silakan klik tombol **"Jalankan Prediksi"** di sidebar untuk memulai.')