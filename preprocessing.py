from lib import *
import importlib
import spellchecker


# Reload spellchecker untuk memastikan fungsi diperbarui
importlib.reload(spellchecker)

# Membaca daftar stopword dari file eksternal
with open("preprocessing/stopword-list.txt", "r") as file:
    stopwords = set(
        file.read().splitlines()
    )  # Menggunakan set untuk optimasi pencarian

# Membuat objek stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# Case Folding & Cleaning Data
def remove(sentence):
    """Menghapus angka, karakter non-alfanumerik, dan mengubah ke huruf kecil."""
    sentence = re.sub(r"[0-9]", " ", sentence)
    sentence = re.sub(r"[^\w\s]", " ", sentence)
    sentence = re.sub(r"[^A-Za-z\s]", "", sentence)
    sentence = sentence.lower().strip()
    sentence = sentence.strip()
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = sentence.replace("\n", "").replace("_", "")
    return sentence


# Tokenization
def tokenize(sentence):
    """Melakukan tokenisasi teks menjadi daftar kata."""
    return sentence.split()


# Normalization
def normalize(tokens):
    """Menerapkan koreksi ejaan menggunakan spellchecker."""
    return spellchecker.normalize_and_correct(tokens)


# Stopword Removal
def remove_stopwords(tokens):
    """Menghapus kata-kata umum yang tidak memiliki makna signifikan."""
    return [token for token in tokens if token not in stopwords]


# Stemming
def stemming(tokens):
    """Melakukan stemming pada setiap token menggunakan Sastrawi."""
    return [stemmer.stem(token) for token in tokens]


# Penanganan Negasi dan Intensifier
def combine_sentiment_phrases(words):
    """Menggabungkan frasa sentimen agar lebih bermakna dalam analisis sentimen."""
    text = " ".join(words)
    # positif
    text = re.sub(r"\bsangat bagus\b", "sangat_bagus", text)
    text = re.sub(r"\bbagus sekali\b", "bagus_sekali", text)
    text = re.sub(r"\bbagus banget\b", "bagus_banget", text)
    text = re.sub(r"\bsuka banget\b", "suka_banget", text)
    text = re.sub(r"\blanggan banget\b", "langgan_banget", text)
    text = re.sub(r"\bsangat bagus\b", "sangat_bagus", text)
    text = re.sub(r"\bsangat baik\b", "sangat_baik", text)
    text = re.sub(r"\bsangat puas\b", "sangat_puas", text)
    text = re.sub(r"\btidak kurang\b", "tidak_kurang", text)
    text = re.sub(r"\btidak rusak\b", "tidak_rusak", text)
    text = re.sub(r"\btidak luntur\b", "tidak_luntur", text)
    text = re.sub(r"\btidak terawang\b", "tidak_terawang", text)
    text = re.sub(r"\btidak murah\b", "tidak_murah", text)
    text = re.sub(r"\btidak kecewa\b", "tidak_kecewa", text)
    text = re.sub(r"\btidak melar\b", "tidak_melar", text)
    text = re.sub(r"\btidak sesal\b", "tidak_sesal", text)
    text = re.sub(r"\btidak sobek\b", "tidak_sobek", text)
    text = re.sub(r"\btidak panas\b", "tidak_panas", text)
    # negatif
    text = re.sub(r"\btidak bagus\b", "tidak_bagus", text)
    text = re.sub(r"\btidak sesuai\b", "tidak_sesuai", text)
    text = re.sub(r"\bsangat kecewa\b", "sangat_kecewa", text)
    text = re.sub(r"\bsangat tidak sesuai\b", "sangat_tidak_sesuai", text)
    text = re.sub(r"\bsedih banget\b", "sedih_banget", text)
    text = re.sub(r"\bkecewa banget\b", "kecewa_banget", text)
    text = re.sub(r"\bkecewa sekali\b", "kecewa_sekali", text)
    text = re.sub(r"\btipis sekali\b", "tipis_sekali", text)
    text = re.sub(r"\bsangat jelek\b", "sangat_jelek", text)
    text = re.sub(r"\bsangat buruk\b", "sangat_buruk", text)
    text = re.sub(r"\bsangat lama\b", "sangat_lama", text)
    text = re.sub(r"\bsangat pendek\b", "sangat_pendek", text)
    text = re.sub(r"\bsangat tipis\b", "sangat_tipis", text)
    text = re.sub(r"\bsangat tidak puas\b", "sangat_tidak_puas", text)
    text = re.sub(r"\bsangat tidak ramah\b", "sangat_tidak_ramah", text)
    text = re.sub(r"\bsangat kurang sekali\b", "sangat_kurang_sekali", text)
    text = re.sub(r"\bjelek banget\b", "jelek_banget", text)
    text = re.sub(r"\btidak suka\b", "tidak_suka", text)
    text = re.sub(r"\btidak nyaman\b", "tidak_nyaman", text)
    text = re.sub(r"\btidak amanah\b", "tidak_amanah", text)
    text = re.sub(r"\btidak puas\b", "tidak_puas", text)
    text = re.sub(r"\btidak ramah\b", "tidak_ramah", text)
    text = re.sub(r"\btidak rapi\b", "tidak_rapi", text)
    text = re.sub(r"\btidak serap\b", "tidak_serap", text)
    text = re.sub(r"\btidak kirim\b", "tidak_kirim", text)
    text = re.sub(r"\btidak teliti\b", "tidak_teliti", text)
    text = re.sub(r"\btidak fungsi\b", "tidak_fungsi", text)
    text = re.sub(r"\btidak jangkau\b", "tidak_jangkau", text)
    # netral
    text = re.sub(r"\btidak apa\b", "tidak_apa", text)
    text = re.sub(r"\bagak kecewa\b", "agak_kecewa", text)
    text = re.sub(r"\blumayan banget\b", "lumayan_banget", text)
    text = re.sub(r"\blumayan lah\b", "lumayan_lah", text)
    text = re.sub(r"\blumayan bagus\b", "lumayan_bagus", text)
    text = re.sub(r"\bkurang suka\b", "kurang_suka", text)
    text = re.sub(r"\bkurang rapi\b", "kurang_rapi", text)
    text = re.sub(r"\bkurang amanah\b", "kurang_amanah", text)
    text = re.sub(r"\bkurang nyaman\b", "kurang_nyaman", text)
    text = re.sub(r"\bkurang sesuai\b", "kurang_sesuai", text)
    text = re.sub(r"\bkurang pas\b", "kurang_pas", text)
    text = re.sub(r"\btidak seperti\b", "tidak_seperti", text)
    text = re.sub(r"\btidak mirip\b", "tidak_mirip", text)
    text = re.sub(r"\bbagus sih\b", "bagus_sih", text)
    text = re.sub(r"\bterima kasih\b", "terima_kasih", text)
    return text.split()
