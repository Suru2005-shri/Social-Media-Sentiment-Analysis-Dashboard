"""
src/cleaner.py
--------------
Text cleaning pipeline for social media posts.
Removes noise, normalises text, and prepares it for NLP.
"""

import re
import string

# ── Optional imports with graceful fallback ──────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Download required NLTK data silently on first use
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    _STOP_WORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except ImportError:
    _STOP_WORDS = {"i","me","my","we","our","you","he","she","it","they","is","are",
                   "was","were","be","been","being","have","has","had","do","does",
                   "did","will","would","could","should","may","might","a","an","the",
                   "and","but","or","nor","so","yet","both","either","not","no","at",
                   "by","for","in","of","on","to","up","as","if","with"}
    NLTK_AVAILABLE = False


# ── Contraction map ──────────────────────────────────────────────────────────
CONTRACTIONS = {
    "won't":"will not","can't":"cannot","don't":"do not","doesn't":"does not",
    "didn't":"did not","isn't":"is not","aren't":"are not","wasn't":"was not",
    "weren't":"were not","haven't":"have not","hasn't":"has not","hadn't":"had not",
    "wouldn't":"would not","couldn't":"could not","shouldn't":"should not",
    "i'm":"i am","i've":"i have","i'll":"i will","i'd":"i would",
    "it's":"it is","that's":"that is","there's":"there is","they're":"they are",
    "we're":"we are","you're":"you are","let's":"let us","he's":"he is",
    "she's":"she is","who's":"who is","what's":"what is",
}


def expand_contractions(text: str) -> str:
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b")
    return pattern.sub(lambda m: CONTRACTIONS[m.group(0)], text.lower())


def clean_text(text: str, remove_stops: bool = True, lemmatize: bool = True) -> str:
    """
    Full cleaning pipeline:
      1. Lowercase
      2. Expand contractions
      3. Remove URLs, mentions, hashtag symbols, HTML entities
      4. Remove punctuation and digits
      5. Strip extra whitespace
      6. Remove stopwords  (optional)
      7. Lemmatize         (optional, requires NLTK)

    Returns cleaned string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase + contractions
    text = expand_contractions(text)

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove @mentions and #hashtag symbols (keep word)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    # 4. Remove HTML entities and emojis (keep ASCII only)
    text = re.sub(r"&\w+;", " ", text)
    text = text.encode("ascii", "ignore").decode()

    # 5. Remove punctuation and digits
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    text = re.sub(r"\d+", " ", text)

    # 6. Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Tokenise
    tokens = text.split()

    # 8. Remove stopwords
    if remove_stops:
        tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]

    # 9. Lemmatize
    if lemmatize and NLTK_AVAILABLE:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_series(series, **kwargs):
    """Apply clean_text to a pandas Series."""
    return series.astype(str).apply(lambda x: clean_text(x, **kwargs))


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "I can't believe how AMAZING @Zomato's delivery was!!! 🔥 https://zomato.com",
        "Worst app ever! #Frustrated with @Swiggy's customer service. They don't help.",
        "It's okay, nothing special. The product does what it's supposed to do.",
    ]
    print("=== Text Cleaning Demo ===\n")
    for s in samples:
        print(f"  RAW   : {s}")
        print(f"  CLEAN : {clean_text(s)}")
        print()
