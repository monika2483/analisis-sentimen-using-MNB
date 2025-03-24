import pandas as pd
import re
from collections import Counter
import math


# Fungsi untuk menghapus huruf yang berulang dan berurutan
def hapus_karakter_berulang(sentence):
    hasil = [sentence[0]]  # Tambahkan karakter pertama ke hasil
    for i in range(1, len(sentence)):
        if (
            sentence[i] != sentence[i - 1]
        ):  # Jika huruf berbeda dengan sebelumnya, tambahkan
            hasil.append(sentence[i])
    return "".join(hasil)


# Memuat kamus kata dari file indonesian-words.txt
def load_dictionary():
    with open(r"preprocessing\indonesian-words.txt", "r", encoding="utf-8") as f:
        return set(f.read().splitlines())


dictionary = load_dictionary()  # Muat kamus kata saat program dijalankan


# Fungsi untuk memeriksa apakah kata ada dalam kamus
def ada_di_kamus(sentence):
    return sentence in dictionary


# Fungsi utama untuk memproses kata input
def handling_karakter_berulang(sentence):
    # Periksa apakah kata ada di dalam kamus
    if ada_di_kamus(sentence):
        return sentence  # Kembalikan kata apa adanya
    else:
        return hapus_karakter_berulang(sentence)  # Proses penghapusan karakter berulang


# NORMALISASI KATA SINGKATAN YANG SANGAT BERBEDA DENGAN KAMUS


# Fungsi untuk memuat normalisasi kata dari file normalization.txt
def load_normalization_words():
    normalize_words = {}
    with open(r"preprocessing\normalization.txt", "r") as f:
        for line in f:
            key, value = line.strip().split(":")  # Pisahkan kata dan penggantinya
            normalize_words[key] = value
    return normalize_words


# Fungsi normalisasi kalimat
def normalization(sentence):
    normalize_words = load_normalization_words()  # Memuat kata-kata normalisasi
    # Jika input berupa string, pecah menjadi list token dan ubah ke huruf kecil
    if isinstance(sentence, str):
        sentence = sentence.lower().split()

    # Iterasi setiap kata di dalam list token
    normalized_sentence = []
    for word in sentence:
        word = handling_karakter_berulang(word)  # Penghapusan huruf berulang
        normalized_word = normalize_words.get(
            word, word
        )  # Normalisasi berdasarkan kamus
        normalized_sentence.extend(
            normalized_word.split()
        )  # Pisahkan jika lebih dari satu kata
    return normalized_sentence  # Mengembalikan sebagai list token


# NORMALISASI DENGAN PROBABILITAS DENGAN KAMUS


def words(text):
    return re.findall(r"\w+", text.lower())


WORDS = Counter(words(open(r"preprocessing\indonesian-words.txt").read()))


def P(word, N=sum(WORDS.values())):
    # "Probability of `word`."
    return WORDS[word] / N


def correction(word):
    # "Most probable spelling correction for word."
    return max(candidates(word), key=P)


def candidates(word):
    # "Generate possible spelling corrections for word."
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]


def known(words):
    # "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    # "All edits that are one edit away from `word`."
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [
        (word[:i], word[i:]) for i in range(len(word) + 1)
    ]  # [('', 'kemarin'), ('k', 'emarin'), ('ke', 'marin'), dst]
    deletes = [L + R[1:] for L, R in splits if R]  # ['emarin', 'kmarin', 'kearin', dst]
    transposes = [
        L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1
    ]  # ['ekmarin', 'kmearin', 'keamrin', dst]
    replaces = [
        L + c + R[1:] for L, R in splits if R for c in letters
    ]  # ['aemarin', 'bemarin', 'cemarin', dst]
    inserts = [
        L + c + R for L, R in splits for c in letters
    ]  # ['akemarin', 'bkemarin', 'ckemarin', dst]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    # "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# Fungsi untuk koreksi tokenisasi (list kata)
def correct_tokens(tokens):
    return [correction(word) for word in tokens]


# Menggabungkan fungsi
def normalize_and_correct(tokens):
    # 1. Langkah pertama: Normalisasi token
    normalized_tokens = normalization(tokens)

    # 2. Langkah kedua: Koreksi ejaan token yang telah dinormalisasi
    corrected_tokens = correct_tokens(normalized_tokens)

    return corrected_tokens  # Kembalikan hasil sebagai list token
