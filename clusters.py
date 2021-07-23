import pickle

import pandas as pd
from sklearn.cluster import (DBSCAN, AffinityPropagation,
                             AgglomerativeClustering, KMeans,
                             SpectralClustering)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from gensim.parsing.preprocessing import preprocess_string, preprocess_documents


input_fn = 'bookmarks_df.p'
data = pd.read_pickle(input_fn)
data = data.fillna('').drop_duplicates().reset_index(drop=True)
data['text'] = data['title'] + data['url_text']

vectorizer = TfidfVectorizer(tokenizer=preprocess_string)
vectorizer.fit(data['text'])
X = pd.DataFrame(
    vectorizer.transform(data['text']).toarray(),
    columns=sorted(vectorizer.vocabulary_.keys())
)

pca = PCA(n_components=0.999)
X_pca = pca.fit_transform(X)

# mod = AgglomerativeClustering(n_clusters=None, distance_threshold=3)
# mod.fit(X_pca)
#
# mod = KMeans(n_clusters=50)
# mod.fit(X_pca)

# mod = SpectralClustering(n_clusters=50)
# mod.fit(X_pca)
#
# mod = AffinityPropagation()
# mod.fit(X_pca)

results = data.copy().drop(['url_text', 'text'], axis=1)
results['label'] = mod.labels_
results.to_csv('test.csv', index=False)

with open('clusters.txt', 'w') as file:
    for cluster in range(mod.n_clusters):
        file.write(f'Cluster: {cluster}\n')
        file.write('\n'.join(results['title'][results['label'] == cluster].values))
        file.write('\n')
        file.write('============================================================================\n')
