"""Karakteristike dostupne iz samog naslova, i tokom treninga i pri unosu."""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TitleStats(BaseEstimator, TransformerMixin):
    """Šest jednostavnih numeričkih karakteristika naslova."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for title in X:
            words = title.split()
            result.append([len(title), len(words), sum(c.isdigit() for c in title),
                           sum(not c.isalnum() and not c.isspace() for c in title),
                           sum(w.isupper() and len(w) > 1 for w in words),
                           max(map(len, words), default=0)])
        return np.asarray(result, dtype=float)


def title_key(title):
    """Normalizacija za uklanjanje duplikata; ne menja tekst za model."""
    return re.sub(r"\s+", " ", title.casefold()).strip()
