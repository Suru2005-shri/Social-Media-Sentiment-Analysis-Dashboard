"""
src/features.py
---------------
Feature extraction pipeline.
  - TF-IDF vectorisation (primary feature set)
  - VADER sentiment scores as supplementary numeric features
  - Combined feature matrix builder
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# VADER – graceful fallback
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    VADER_OK = True
except ImportError:
    VADER_OK = False


# ── VADER feature extraction ─────────────────────────────────────────────────

def vader_features(texts) -> np.ndarray:
    """
    Returns (n_samples, 4) array of VADER scores:
    [compound, pos, neu, neg]
    Falls back to zeros if vaderSentiment not installed.
    """
    if not VADER_OK:
        return np.zeros((len(texts), 4))

    rows = []
    for t in texts:
        sc = _VADER.polarity_scores(str(t))
        rows.append([sc["compound"], sc["pos"], sc["neu"], sc["neg"]])
    return np.array(rows)


def vader_label(text: str) -> str:
    """Single-text VADER label for the live analyser."""
    if not VADER_OK:
        return "neutral"
    sc = _VADER.polarity_scores(str(text))
    if sc["compound"] >= 0.05:
        return "positive"
    if sc["compound"] <= -0.05:
        return "negative"
    return "neutral"


def vader_scores(text: str) -> dict:
    """Return full VADER dict for a single text (for the live UI)."""
    if not VADER_OK:
        return {"compound": 0.0, "pos": 0.33, "neu": 0.34, "neg": 0.33}
    return _VADER.polarity_scores(str(text))


# ── TF-IDF vectoriser ────────────────────────────────────────────────────────

def build_tfidf(max_features: int = 8000,
                ngram_range: tuple = (1, 2),
                sublinear_tf: bool = True) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=2,
        max_df=0.92,
        strip_accents="unicode",
        analyzer="word",
    )


# ── Combined feature builder ─────────────────────────────────────────────────

class FeatureBuilder:
    """
    Combines TF-IDF + VADER into a single sparse/dense feature matrix.

    Usage
    -----
    fb = FeatureBuilder()
    X_train = fb.fit_transform(train_texts_clean, train_texts_raw)
    X_test  = fb.transform(test_texts_clean, test_texts_raw)
    """

    def __init__(self, max_tfidf: int = 8000):
        self.tfidf = build_tfidf(max_features=max_tfidf)
        self._fitted = False

    def fit_transform(self, clean_texts, raw_texts=None):
        tfidf_mat = self.tfidf.fit_transform(clean_texts)
        self._fitted = True

        if raw_texts is not None and VADER_OK:
            vf = csr_matrix(vader_features(raw_texts))
            return hstack([tfidf_mat, vf])
        return tfidf_mat

    def transform(self, clean_texts, raw_texts=None):
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        tfidf_mat = self.tfidf.transform(clean_texts)

        if raw_texts is not None and VADER_OK:
            vf = csr_matrix(vader_features(raw_texts))
            return hstack([tfidf_mat, vf])
        return tfidf_mat

    def vocabulary_size(self) -> int:
        return len(self.tfidf.vocabulary_)

    def top_features(self, n: int = 20) -> list:
        names = np.array(self.tfidf.get_feature_names_out())
        return names[:n].tolist()


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    texts_raw   = ["Absolutely love this product!", "Terrible experience, never again.", "It's okay I guess."]
    texts_clean = ["absolutely love product", "terrible experience never", "okay guess"]

    fb = FeatureBuilder(max_tfidf=100)
    X  = fb.fit_transform(texts_clean, texts_raw)
    print(f"Feature matrix shape : {X.shape}")
    print(f"VADER ok             : {VADER_OK}")
    for t in texts_raw:
        print(f"  '{t[:40]}' → {vader_label(t)}  scores={vader_scores(t)}")
