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

# Set page configuration
st.set_page_config(page_title="Weather Data Analysis", layout="wide")
sns.set(style="whitegrid")

# Fungsi untuk memuat dan memproses dataset
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

# Load dataset
data = load_data()
if data is None:
    st.stop()

# Sidebar untuk navigasi
st.sidebar.title("🌍 Menu Navigasi")
menu_options = ["Beranda", "Dataset", "Visualisasi", "Model Evaluasi", "Word Cloud"]
selected_menu = st.sidebar.radio("Pilih Halaman", menu_options)

if selected_menu == "Beranda":
    st.title("☀️ Weather Data Analysis")
    st.write("""
        Selamat datang di aplikasi analisis data cuaca. Di sini, Anda dapat mengeksplorasi data cuaca, 
        memvisualisasikan tren, dan menganalisis model menggunakan berbagai metode yang menarik.
    """)

    st.image("https://source.unsplash.com/800x400/?weather", caption="Weather Insights", use_column_width=True)

elif selected_menu == "Dataset":
    st.title("Dataset")
    st.write("### Tampilkan Dataset")
    st.write(data)

    st.write("Jumlah nilai NaN setelah diatasi:")
    st.write(data.isnull().sum())

elif selected_menu == "Visualisasi":
    st.title("Visualisasi Data")

    selected_feature = st.selectbox("Pilih Fitur untuk Visualisasi", ["Temperature (Celsius)", "Wind Speed (m/s)"])
    st.subheader(f"Distribusi {selected_feature}")
    fig = px.histogram(data, x=selected_feature, nbins=20, title=f"Histogram {selected_feature}", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribusi Kecepatan Angin Berdasarkan Negara")
    fig = px.box(data, x='Country', y='Wind Speed (m/s)', color='Country', title="Distribusi Berdasarkan Negara")
    st.plotly_chart(fig, use_container_width=True)

elif selected_menu == "Model Evaluasi":
    st.title("Evaluasi Model")

    features = ['Temperature (Celsius)', 'Wind Speed (m/s)', 'Latitude', 'Longitude']
    X = data[features]
    y = data['Description']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    accuracy = accuracy_score(y_test, predictions)
    st.write(f"### Accuracy: {accuracy:.2f}")

    st.write("### Classification Report:")
    report = classification_report(y_test, predictions, target_names=label_encoder.classes_)
    st.text(report)

    st.subheader("Feature Importance")
    importance = model.feature_importances_
    feature_imp = pd.DataFrame(importance, index=features, columns=["Importance"]).sort_values(by="Importance", ascending=False)
    st.dataframe(feature_imp.style.background_gradient(cmap="Blues"))

elif selected_menu == "Word Cloud":
    st.title("Word Cloud")

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
