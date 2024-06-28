import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def predict_naive_bayes(gender, foto, hiv, diabetes, tcm, probabilities):
    # Calculate posterior probabilities for class "PARU"
    p_paru = probabilities['PARU']['TOTAL']
    for feature, value in {'JENIS KELAMIN': gender, 'FOTO TORAKS': foto, 'STATUS HIV': hiv, 'RIWAYAT DIABETES': diabetes, 'HASIL TCM': tcm}.items():
        if feature != 'HASIL TCM':
            p_paru *= probabilities['PARU'][feature][value]
    
    if tcm is not None:
        p_paru *= probabilities['PARU']['HASIL TCM'][tcm]

    # Calculate posterior probabilities for class "EKSTRA PARU"
    p_ekstra_paru = probabilities['EKSTRA PARU']['TOTAL']
    for feature, value in {'JENIS KELAMIN': gender, 'FOTO TORAKS': foto, 'STATUS HIV': hiv, 'RIWAYAT DIABETES': diabetes, 'HASIL TCM': tcm}.items():
        if feature != 'HASIL TCM':
            p_ekstra_paru *= probabilities['EKSTRA PARU'][feature][value]
    
    if tcm is not None:
        p_ekstra_paru *= probabilities['EKSTRA PARU']['HASIL TCM'][tcm]

    # Normalize probabilities by dividing with the total probabilities
    p_paru /= (p_paru + p_ekstra_paru)
    p_ekstra_paru /= (p_paru + p_ekstra_paru)

    return p_paru, p_ekstra_paru

def main():
    st.set_page_config(page_title="Klasifikasi Penyakit Tuberculosis Dengan menggunakan metode Naive Bayes", layout="wide")

    st.markdown("""
        <style> 
            .main { background-color: #607274; } /* Warna latar belakang utama */
            .stButton>button { background-color: #4CAF50; color: white; } /* Tombol warna hijau */
            .stTextInput>div>div>input { font-size: 1.2rem; } /* Input teks ukuran font */
            h3 { color: #FB8B24; } /* Judul warna biru */
            .orange { color: orange; } /* Teks warna orange */
            .red { color: red; } /* Teks warna merah */
            .split-container { display: flex; }
            .split-container > div { flex: 1; padding: 10px; }
            .split-container > div:first-child { margin-right: 10px; }
            .scrollable { max-height: 400px; overflow-y: auto; }
            .stNavbar { background-color: #333; padding: 10px; } /* Tampilan navbar */
            .stNavbar a { color: white; text-decoration: none; margin-right: 20px; font-size: 18px; } /* Tautan navbar */
            .stNavbar a:hover { color: yellow; } /* Efek hover */
        </style>
    """, unsafe_allow_html=True)

    st.title("Klasifikasi Penyakit Tuberculosis Dengan menggunakan metode Naive Bayes")

    # Load data
    df = pd.read_excel("Data_TB_987_FIK_KEL 7.xlsx")

    # Menu navigasi
    page = st.radio("Navigasi", ["Data", "Preprocessing", "Modeling", "Implementasi"])

    # Halaman Data
    if page == "Data":
        st.header("Data Tuberkulosis")
        st.subheader("Memahami Data & Metode yang Digunakan")

        st.markdown("""
            Dataset ini digunakan untuk mengklasifikasikan Penyakit Tuberkulosis berdasarkan beberapa fitur yang telah dikumpulkan. 
            Berikut adalah penjelasan dari fitur-fitur yang ada:
            - STATUS HIV: Status kesehatan HIV, umumnya diukur dengan kategori "Positif" atau "Negatif".
            - JENIS KELAMIN: Jenis kelamin, yang biasanya diberi label "Laki-laki" atau "Perempuan".
            - FOTO TORAKS: Hasil dari foto rontgen dada, sering kali digunakan untuk memeriksa adanya infeksi paru-paru atau gangguan pernapasan lainnya.
            - RIWAYAT DIABETES: Riwayat kejadian diabetes, biasanya diberi label "Ya" atau "Tidak".
            - HASIL TCM: Hasil dari tes, sering kali digunakan untuk memeriksa keberadaan infeksi bakteri TB.
            
            Metode Naive Bayes adalah salah satu algoritma klasifikasi yang menggunakan Teorema Bayes dengan asumsi bahwa setiap fitur bersifat independen satu sama lain. Meskipun asumsi ini sangat sederhana dan jarang terjadi dalam kehidupan nyata, Naive Bayes sering kali memberikan hasil yang baik dan efisien, terutama untuk dataset dengan jumlah fitur yang besar.
        """)
        st.dataframe(df, height=600)
        st.subheader("Grafik Umur dalam Data")

        # Plot histogram with more bins and smaller figure size
        fig, ax = plt.subplots(figsize=(20, 10))  # Decrease figure size
        df.hist(bins=50, ax=ax)
        st.pyplot(fig)

    # Halaman Preprocessing
    elif page == "Preprocessing":
        st.header("Preprocessing")
        st.write("Data Asli:")
        st.dataframe(df, height=600)

        # Periksa nilai yang hilang
        st.subheader("Nilai yang Hilang:")
        st.markdown(""" Jumlah data yang kosong atau tidak terisi di dalam setiap kolom """)
        missing_values = df.isnull().sum()
        st.write(missing_values)
        
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Label encoding untuk kolom kategorikal
        st.subheader("Data Setelah Label Encoding:")
        st.markdown(""" Data yang sudah diisi dengan menggunakan rumus missing value yaitu (modus) """)
        for column in df_imputed.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_imputed[column] = le.fit_transform(df_imputed[column])

        st.dataframe(df_imputed, height=400)

    # Halaman Modeling
    elif page == "Modeling":
        st.header("Modeling")
        st.markdown("## Penghitungan Akurasi Naive Bayes")

        tb_df = pd.read_excel("Data_TB_987_FIK_KEL 7.xlsx")

        # Preprocessing
        imputer = SimpleImputer(strategy='most_frequent')
        tb_df_imputed = pd.DataFrame(imputer.fit_transform(tb_df), columns=tb_df.columns)

        for column in tb_df_imputed.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            tb_df_imputed[column] = le.fit_transform(tb_df_imputed[column])

        # Split dataset
        X = tb_df_imputed.drop('LOKASI ANATOMI (target/output)', axis=1)
        y = tb_df_imputed['LOKASI ANATOMI (target/output)']

        # Encode target variable if it's categorical
        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write(f"Jumlah data training: {len(X_train)}")
        st.write(f"Jumlah data testing: {len(X_test)}")

        # Train Naive Bayes model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate precision, recall, f1-score
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-score: {fscore:.2f}")

        st.write("Classification Report:")
        class_report = classification_report(y_test, y_pred)
        st.text(class_report)


    # Halaman Implementasi
    elif page == "Implementasi":
        st.header("Implementasi")
        # Probabilitas dari data yang diberikan
        probabilities = {
            'PARU': {
                'JENIS KELAMIN': {0: 0.406162465, 1: 0.593837535},
                'FOTO TORAKS': {0: 0.002801120448, 1: 0.9971988796},
                'STATUS HIV': {0: 0.9705882353, 1: 0.02941176471},
                'RIWAYAT DIABETES': {0: 0.9523809524, 1: 0.04761904762},
                'HASIL TCM': {0: 0.494, 1: 0.506},
                'TOTAL': 0.7248730964
            },
            'EKSTRA PARU': {
                'JENIS KELAMIN': {0: 0.479704797, 1: 0.520295203},
                'FOTO TORAKS': {0: 1.0, 1: 0.0},
                'STATUS HIV': {0: 0.9815498155, 1: 0.0184501845},
                'RIWAYAT DIABETES': {0: 0.9557195572, 1: 0.0442804428},
                'HASIL TCM': {0: 1.0, 1: 0.0},
                'TOTAL': 0.2751269036
            }
        }

        gender = st.selectbox("JENIS KELAMIN (0: Laki-laki, 1: Perempuan):", [0, 1])
        foto = st.selectbox("FOTO TORAKS (0: Negatif, 1: Positif):", [0, 1])
        hiv = st.selectbox("STATUS HIV (0: Negatif, 1: Positif):", [0, 1])
        diabetes = st.selectbox("RIWAYAT DIABETES (0: Tidak, 1: Ya):", [0, 1])
        tcm = st.selectbox("HASIL TCM (0: Tidak, 1: Ya):", [0, 1])

        if st.button("Prediksi"):
            p_paru, p_ekstra_paru = predict_naive_bayes(gender, foto, hiv, diabetes, tcm, probabilities)
            
            # Plotting the results
            labels = ['PARU', 'EKSTRA PARU']
            probabilities = [p_paru, p_ekstra_paru]

            fig, ax = plt.subplots()
            ax.bar(labels, probabilities, color=['blue', 'orange'])
            ax.set_ylabel('Probabilitas')
            ax.set_title('Probabilitas Prediksi')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
