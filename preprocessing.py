# preprocessing.py
# Modul preprocessing yang sama persis dengan notebook
# Di-extract ke file terpisah agar bisa dipakai app.py
# ============================================================

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (hanya jika belum ada)
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}' if 'punkt' in r else f'corpora/{r}')
        except LookupError:
            nltk.download(r, quiet=True)

download_nltk_resources()


def fix_hashtags(text: str) -> str:
    """Expand hashtag ke kata-kata (misal: #FreePalestine -> free palestine)"""
    try:
        from wordsegment import load, segment
        load()
        hashtags = re.findall(r'#(\w+)', text)
        for ht in hashtags:
            broken_camel = re.sub(r'([a-z])([A-Z])', r'\1 \2', ht)
            segmented = segment(broken_camel)
            combined = ' '.join(segmented)
            text = text.replace(f'#{ht}', combined)
    except ImportError:
        # Fallback: hapus simbol # saja jika wordsegment tidak tersedia
        text = re.sub(r'#(\w+)', r'\1', text)
    return text


def clean_tweet(tweet: str) -> str:
    """
    Bersihkan tweet:
    - Expand hashtag
    - Hapus URL
    - Hapus karakter non-alphanumeric
    - Normalisasi whitespace
    """
    tweet = fix_hashtags(tweet)
    tweet = re.sub(r'https\S+', '', tweet, flags=re.MULTILINE)  # hapus URL
    tweet = re.sub(r'\W', ' ', tweet)                            # hapus non-word
    tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)               # hapus simbol sisa
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet


def tokenize_and_lemmatize(tweet: str) -> str:
    """
    Tokenisasi, lowercase, hapus stopwords, lemmatisasi
    Sama persis dengan TweetProcessor.tokenize_and_lemmatize di notebook
    """
    tokens = word_tokenize(tweet)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def preprocess_single_tweet(text: str) -> str:
    """
    Pipeline lengkap untuk satu tweet input.
    Output: string yang siap dipakai oleh TF-IDF / RoBERTa tokenizer.
    """
    if not isinstance(text, str) or text.strip() == '':
        return ''
    cleaned = clean_tweet(text)
    lemmatized = tokenize_and_lemmatize(cleaned)
    return lemmatized
