import re
import string
import tensorflow as tf

STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'adalah', 'untuk', 'dengan', 'itu',
    'ini', 'karena', 'sebagai', 'juga', 'tidak', 'dalam', 'oleh', 'atau', 'agar'
}

def custom_standardization(input_text):
    lowercase = tf.strings.lower(input_text)
    # Tambahkan handling untuk karakter khusus Indonesia
    cleaned = tf.strings.regex_replace(lowercase, f"[{re.escape(string.punctuation)}]", " ")
    cleaned = tf.strings.regex_replace(cleaned, r'\d+', '')
    cleaned = tf.strings.regex_replace(cleaned, r'\s+', ' ')
    cleaned = tf.strings.strip(cleaned)  # Tambahkan strip
    return cleaned


def create_text_vectorizer(texts, max_tokens=10000, max_len=100):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=max_len,
        standardize=custom_standardization,
        split='whitespace',
    )
    vectorizer.adapt(texts)
    return vectorizer
