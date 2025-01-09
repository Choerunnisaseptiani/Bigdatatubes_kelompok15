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

# Page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Analisis Data Cuaca", layout="wide")

# Sidebar for navigation with emojis as icons
st.sidebar.title("📚 Menu Navigasi")

menu_options = {
    "Beranda": "🏠",
    "Dataset": "📊",
    "Visualisasi": "📉",
    "Evaluasi Model": "🔬",
    "Word Cloud": "🧠"
}

selected_menu = st.sidebar.radio("Pilih Halaman", list(menu_options.keys()), format_func=lambda x: f"{menu_options[x]} {x}")

# Set style for seaborn
sns.set(style="whitegrid")

# Function to load and preprocess the dataset
def load_data():
    try:
        data = pd.read_csv("top100cities_weather_data.csv")
        numeric_cols = data.select_dtypes(include=['number']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        non_numeric_cols = data.select_dtypes(exclude=['number']).columns
        data[non_numeric_cols] = data[non_numeric_cols].fillna(data[non_numeric_cols].mode().iloc[0])
        return data
    except FileNotFoundError:
        st.error("File 'top100cities_weather_data.csv' tidak ditemukan. Pastikan file tersedia di direktori.")
        return None

# Load the dataset
data = load_data()
if data is None:
    st.stop()

# Main content based on selected menu
if selected_menu == "Beranda":
    st.title("☀️ Analisis Data Cuaca")
    st.write(""" 
        Selamat datang di aplikasi analisis data cuaca. Di sini, Anda dapat mengeksplorasi data cuaca, 
        memvisualisasikan tren, dan menganalisis model menggunakan berbagai metode yang menarik.
    """)
    st.image("images.jpg", caption="Weather Insights", use_container_width=True)

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
    selected_feature = st.selectbox("Pilih Fitur untuk Visualisasi", numeric_features)
    st.subheader(f"Distribusi {selected_feature}")
    fig = px.histogram(data, x=selected_feature, nbins=20, title=f"Histogram {selected_feature}", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig, use_container_width=True)

    if 'Country' in data.columns:
        st.subheader("Distribusi Berdasarkan Negara")
        fig = px.box(data, x='Country', y=selected_feature, color='Country', title=f"Distribusi {selected_feature} Berdasarkan Negara")
        st.plotly_chart(fig, use_container_width=True)

elif selected_menu == "Evaluasi Model":
    st.title("Evaluasi Model")
    if 'Description' not in data.columns:
        st.error("Kolom 'Description' tidak ditemukan pada dataset. Tidak dapat melatih model.")
    else:
        features = ['Temperature (Celsius)', 'Wind Speed (m/s)', 'Latitude', 'Longitude']
        X = data[features]
        y = data['Description']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)  # Encode labels

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        cm = confusion_matrix(y_test, predictions)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        accuracy = accuracy_score(y_test, predictions)
        st.write(f"### Akurasi: {accuracy:.2f}")

        # Classification Report
        st.write("### Classification Report:")
        unique_classes_in_test = sorted(set(y_test))  # Classes in y_test
        report = classification_report(
            y_test, 
            predictions, 
            target_names=label_encoder.classes_,  # Classes from the label encoder
            labels=unique_classes_in_test,  # Only include classes from y_test
            zero_division=0
        )
        st.text(report)

        st.subheader("Pentingnya Fitur")
        importance = model.feature_importances_
        feature_imp = pd.DataFrame(importance, index=features, columns=["Penting"]).sort_values(by="Penting", ascending=False)
        st.dataframe(feature_imp.style.background_gradient(cmap="Blues"))

elif selected_menu == "Word Cloud":
    st.title("Word Cloud")
    if 'Country' in data.columns and 'Description' in data.columns:
        country_counts = data['Country'].value_counts()
        wordcloud_country = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate_from_frequencies(country_counts)
        st.subheader("Word Cloud - Negara")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_country, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        description_counts = data['Description'].value_counts()
        wordcloud_description = WordCloud(width=800, height=400, background_color="white", colormap="magma").generate_from_frequencies(description_counts)
        st.subheader("Word Cloud - Deskripsi Cuaca")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud_description, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.error("Kolom 'Country' atau 'Description' tidak ditemukan pada dataset.")
