import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
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

mod = AgglomerativeClustering(n_clusters=35)
mod.fit(X_pca)

results = data.copy().drop('url_text', axis=1)
results['label'] = mod.labels_
results.to_csv('test.csv', index=False)

for cluster in range(mod.n_clusters_):
    print(f'Cluster: {cluster}')
    print(results['title'][results['label'] == cluster])
    print()

with open('clusters.txt', 'w') as file:
    for cluster in range(mod.n_clusters_):
        file.write(f'Cluster: {cluster}\n')
        file.write('\n'.join(results['title'][results['label'] == cluster].values))
        file.write('\n')
        file.write('===========================================\n')
