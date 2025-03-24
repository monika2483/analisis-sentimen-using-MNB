from lib import *
import streamlit as st
from preprocessing import (
    remove,
    tokenize,
    normalize,
    remove_stopwords,
    stemming,
    combine_sentiment_phrases,
)

# Konfigurasi layout
st.set_page_config(layout="wide")

# Load model
tfidf_idf_values = joblib.load("tfidf_idf_values.pkl")
selected_features = joblib.load("mi_selector.pkl")
nb_model = joblib.load("mnb_model.pkl")


# Fungsi preprocessing
def preprocess_text(text):
    cleaned_text = remove(text)
    tokens = tokenize(cleaned_text)
    normalized_tokens = normalize(tokens)
    filtered_tokens = remove_stopwords(normalized_tokens)
    stemmed_tokens = stemming(filtered_tokens)
    final_tokens = combine_sentiment_phrases(stemmed_tokens)
    return " ".join(final_tokens)


# TF-IDF
def compute_tf(words):
    word_count = len(words)
    word_freq = {word: words.count(word) / word_count for word in words}
    return word_freq


def compute_tfidf(text):
    words = text.split()
    tf_values = compute_tf(words)
    tfidf_vector = {
        word: tf_values[word] * tfidf_idf_values.get(word, 0) for word in words
    }
    selected_tfidf_vector = {
        word: tfidf_vector[word] for word in selected_features if word in tfidf_vector
    }
    return selected_tfidf_vector


# Prediksi
def predict_sentiment(text):
    tfidf_vector = compute_tfidf(text)
    class_prior = nb_model["class_prior"]
    word_prob = nb_model["word_prob"]

    class_scores = {}
    for c in class_prior.keys():
        prob = np.log(class_prior[c])
        for word, tfidf_value in tfidf_vector.items():
            if word in word_prob[c]:
                prob += np.log(word_prob[c][word]) * tfidf_value
        class_scores[c] = prob

    return max(class_scores, key=class_scores.get)


# Tampilan antarmuka
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.markdown("<h4>Pilih Mode Analisis</h4>", unsafe_allow_html=True)
menu = st.sidebar.radio("", ["Analisis Teks", "Analisis File"])

# Mode Analisis Teks
if menu == "Analisis Teks":
    st.markdown(
        "<h2 style='text-align:center;'>Analisis Sentimen Review Toko</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Sistem ini digunakan untuk mengetahui sentimen dari ulasan pelanggan apakah termasuk positif, netral, atau negatif.</p>",
        unsafe_allow_html=True,
    )
    st.write("Masukkan teks ulasan/review di sini :")
    user_input = st.text_area("", "", height=150)
    pred_col, _ = st.columns([1, 5])
    with pred_col:
        if st.button("Prediksi"):
            if user_input:
                processed = preprocess_text(user_input)
                result = predict_sentiment(processed)
                label = (
                    "Negatif"
                    if result == -1
                    else "Netral" if result == 0 else "Positif"
                )
                st.write("**Hasil prediksi :**", f"**{label}**")

# Mode Analisis File
elif menu == "Analisis File":
    st.markdown(
        "<h2 style='text-align:center;'>Analisis Sentimen Review Toko</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;'>Sistem ini digunakan untuk mengetahui sentimen dari ulasan pelanggan apakah termasuk positif, netral, atau negatif.</p>",
        unsafe_allow_html=True,
    )
    st.write("Upload file ulasan di sini :")
    st.write("Pastikan terdapat header 'text' pada kolom yang berisi ulasan")
    uploaded_file = st.file_uploader("", type=["csv", "xlsx"])

    if st.button("Prediksi") and uploaded_file:
        df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )

        if "text" not in df.columns:
            st.write("Error: File harus memiliki kolom 'text'")
        else:
            df["Processed_Text"] = df["text"].astype(str).apply(preprocess_text)
            df["Prediction"] = df["Processed_Text"].apply(predict_sentiment)
            label_map = {-1: "Negatif", 0: "Netral", 1: "Positif"}
            df["Prediction_Label"] = df["Prediction"].map(label_map)

            total = len(df)
            dist = df["Prediction_Label"].value_counts(normalize=True) * 100

            st.write(f"Total ulasan dalam file : **{total}**")

            col1, col2 = st.columns([1.5, 3.5])

            with col1:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie(
                    dist,
                    labels=dist.index,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=["#ff6b6b", "#4dabf7", "#ffe066"],
                )
                ax.axis("equal")
                st.pyplot(fig)

            with col2:
                st.dataframe(df[["text", "Prediction_Label"]], height=600, width=1100)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh hasil prediksi", csv, "hasil_prediksi.csv", "text/csv"
            )
