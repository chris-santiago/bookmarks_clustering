"""

Much faster than lemmatizing (takes 10seconds), but not really needed.
SKlearn TFIDF Vectorizer can take a tokenizer as parameter, making this module moot.
"""

import time
import pathlib
import pickle

from gensim.parsing.preprocessing import preprocess_string, preprocess_documents
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from bookmarks.utils.types import Website

HERE = pathlib.Path(__file__)
CURR_DIR = HERE.parent
PARENT = CURR_DIR.parent


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self,):
        self.tokenize = preprocess_string

    @staticmethod
    def doc(website):
        if website.content:
            return website.title + website.content
        return website.title

    def fit(self, X, y=None):
        documents = [self.doc(x) for x in X]
        self.output_features_ = [self.tokenize(doc) for doc in documents]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.output_features_


if __name__ == '__main__':
    with open(PARENT.joinpath('websites.p'), 'rb') as fp:
        sites = pickle.load(fp)
    start = time.time()
    norm = TextNormalizer()
    norm.fit_transform(sites)
    duration = time.time() - start
    print(f"Process took {round(duration / 60, 1)} minutes.")
    print(norm.output_features_[65])
