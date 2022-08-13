import pickle
import pathlib

import pandas as pd
from sklearn.cluster import (DBSCAN, AffinityPropagation,
                             AgglomerativeClustering, KMeans,
                             SpectralClustering)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from gensim.parsing.preprocessing import preprocess_string, preprocess_documents
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse

HERE = pathlib.Path(__file__)
CURR_DIR = HERE.parent
PARENT = CURR_DIR.parent


def get_data(filepath):
    data = pd.read_pickle(filepath)
    return data.fillna('').drop_duplicates().reset_index(drop=True)


def get_bookmark_text(data):
    return data['title'] + data['url_text']


class BookmarkCluster(BaseEstimator, TransformerMixin):
    def __init__(self, cluster=KMeans(40)):
        self.vec = TfidfVectorizer(tokenizer=preprocess_string)
        self.cluster = cluster
        self.pca = TruncatedSVD(n_components=100)  # TruncatedSVD for sparse matrix
        self.pl = Pipeline(
            [
                ('vectorize', self.vec),
                # ('pca', self.pca),
                ('cluster', self.cluster)
            ]
        )

    def fit(self, X, y=None):
        self.pl.fit(X)

    def transform(self, X, y=None):
        self.pl.transform(X)

    def predict(self, X, y=None):
        self.pl.predict(X)

    def to_dataframe(self, X, y=None):
        check_is_fitted(self.vec)
        return pd.DataFrame(
            self.vec.transform(X).toarray(),
            columns=sorted(self.vec.vocabulary_.keys())
        )

    def label_data(self, data):
        results = data.copy().drop(['url_text'], axis=1)
        results['label'] = self.cluster.labels_
        return results

    def labeled_to_file(self, data, filepath='clusters.txt'):
        results = self.label_data(data)
        with open(filepath, 'w') as file:
            for cluster in range(self.cluster.n_clusters):
                file.write(f'Cluster: {cluster}\n')
                file.write('\n'.join(results['title'][results['label'] == cluster].values))
                file.write('\n')
                file.write('============================================================================\n')


if __name__ == '__main__':
    bookmark_file = 'bookmarks_df.p'
    bookmark_data = pd.read_pickle(bookmark_file).drop_duplicates().dropna().reset_index(drop=True)
    bookmark_text = bookmark_data['url_text']
    bc = BookmarkCluster()
    bc.fit(bookmark_text)
    bc.labeled_to_file(bookmark_data)
