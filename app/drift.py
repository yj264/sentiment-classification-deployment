import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DriftDetector:
    def __init__(self, ref_file="mlruns/latest_model/train_word_dist.csv"):
        self.ref_dist = pd.read_csv(ref_file).set_index("word")["freq"]
        self.ref_dist = self.ref_dist / self.ref_dist.sum()

    def check_drift(self, texts):
        vectorizer = TfidfVectorizer(vocabulary=self.ref_dist.index)
        freq = vectorizer.fit_transform(texts).sum(axis=0).A1
        curr_dist = pd.Series(freq, index=self.ref_dist.index)
        curr_dist = curr_dist / (curr_dist.sum() + 1e-9)

        # Kullback-Leibler divergence
        kl_div = np.sum(self.ref_dist * np.log((self.ref_dist + 1e-9) / (curr_dist + 1e-9)))
        return kl_div
