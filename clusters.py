import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

input_fn = 'bookmarks_data.p'
with open(input_fn, 'rb') as file:
    data = pickle.load(file)

data = data.dropna().drop_duplicates().reset_index(drop=True)

vectorizer = TfidfVectorizer()
vectorizer.fit(data['url_text'])
X = pd.DataFrame(
    vectorizer.transform(data['title']).toarray(),
    columns=sorted(vectorizer.vocabulary_.keys())
)

pca = PCA(n_components=0.999)
X_pca = pca.fit_transform(X)

mod = AgglomerativeClustering(n_clusters=None, distance_threshold=2)
mod.fit(X_pca)

# mod = KMeans(n_clusters=50)
# mod.fit(X_pca)

# mod = SpectralClustering(n_clusters=50)
# mod.fit(X_pca)

results = data.copy().drop('url_text', axis=1)
results['label'] = mod.labels_
results.to_csv('test.csv', index=False)

with open('clusters.txt', 'w') as file:
    for cluster in range(mod.n_clusters_):
        file.write(f'Cluster: {cluster}\n')
        file.write('\n'.join(results['title'][results['label'] == cluster].values))
        file.write('\n')
        file.write('============================================================================\n')
