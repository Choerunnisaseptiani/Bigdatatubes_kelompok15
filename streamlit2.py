import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Data Cuaca", layout="wide", page_icon="🌤️")
sns.set(style="whitegrid")

# Fungsi untuk memuat dataset
def load_data():
    try:
        data = pd.read_csv("top100cities_weather_data.csv")
        numeric_cols = data.select_dtypes(include=['number']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        non_numeric_cols = data.select_dtypes(exclude=['number']).columns
        data[non_numeric_cols] = data[non_numeric_cols].fillna(data[non_numeric_cols].mode().iloc[0])
        return data
    except FileNotFoundError:
        st.error("File 'top100cities_weather_data.csv' tidak ditemukan.")
        return None

# Muat dataset
data = load_data()

# Sidebar untuk navigasi
st.sidebar.title("🌍 Menu Navigasi")
menu_options = ["Beranda", "Dataset", "Visualisasi", "Evaluasi Model", "Word Cloud"]
selected_menu = st.sidebar.radio("Pilih Halaman", menu_options)

if selected_menu == "Beranda":
    st.title("🌤️ Analisis Data Cuaca")
    st.markdown(""" 
        **Aplikasi ini membantu Anda memahami data cuaca dengan cara yang interaktif dan menarik.**  
        Jelajahi dataset, buat visualisasi, evaluasi model prediktif, dan temukan pola cuaca menggunakan Word Cloud!
    """)
    st.image("https://via.placeholder.com/800x300?text=Analisis+Data+Cuaca", caption="Gambaran Analisis Cuaca")

elif selected_menu == "Dataset":
    st.title("Dataset")
    st.markdown("### Unggah Dataset Anda:")
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset berhasil diunggah!")
    if data is not None:
        st.write(data.head())
        st.markdown("### Info Dataset:")
        st.write(data.info())
        st.markdown("### Jumlah nilai kosong (NaN):")
        st.write(data.isnull().sum())
    else:
        st.warning("Belum ada dataset yang dimuat.")

elif selected_menu == "Visualisasi":
    st.title("Visualisasi Data")
    numeric_features = data.select_dtypes(include=['number']).columns
    if numeric_features.empty:
        st.error("Tidak ada fitur numerik untuk divisualisasikan.")
    else:
        selected_feature = st.selectbox("Pilih Fitur untuk Visualisasi", numeric_features)
        st.markdown(f"### Distribusi {selected_feature}")
        fig = px.histogram(data, x=selected_feature, nbins=20, title=f"Histogram {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

        if 'Country' in data.columns:
            st.markdown(f"### Distribusi {selected_feature} Berdasarkan Negara")
            fig = px.box(data, x='Country', y=selected_feature, color='Country', title=f"Distribusi {selected_feature} Berdasarkan Negara")
            st.plotly_chart(fig, use_container_width=True)

elif selected_menu == "Evaluasi Model":
    st.title("Evaluasi Model")
    if 'Description' not in data.columns:
        st.error("Kolom 'Description' tidak ditemukan. Tidak dapat melatih model.")
    else:
        features = ['Temperature (Celsius)', 'Wind Speed (m/s)', 'Latitude', 'Longitude']
        X = data[features]
        y = data['Description']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        cm = confusion_matrix(y_test, predictions)
        st.markdown("### Confusion Matrix:")
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        accuracy = accuracy_score(y_test, predictions)
        st.markdown(f"### Akurasi Model: **{accuracy:.2f}**")

        st.markdown("### Laporan Klasifikasi:")
        report = classification_report(y_test, predictions, target_names=label_encoder.classes_, zero_division=0)
        st.text(report)

elif selected_menu == "Word Cloud":
    st.title("Word Cloud")
    if 'Country' in data.columns and 'Description' in data.columns:
        st.markdown("### Word Cloud - Negara")
        country_counts = data['Country'].value_counts()
        wordcloud_country = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(country_counts)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_country, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        st.markdown("### Word Cloud - Deskripsi Cuaca")
        description_counts = data['Description'].value_counts()
        wordcloud_description = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(description_counts)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud_description, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.error("Kolom 'Country' atau 'Description' tidak ditemukan.")
