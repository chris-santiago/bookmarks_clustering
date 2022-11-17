import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE

bookmark_file = 'bookmarks_df.p'
bookmark_data = pd.read_pickle(bookmark_file).drop_duplicates().dropna().reset_index(drop=True)
bookmark_text = bookmark_data['url_text']

sentences = bookmark_text.to_list()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences, show_progress_bar=True)

with open('embeddings.p', 'wb') as fp:
    pickle.dump(embeddings, fp)

# cluster = KMeans(40)
# cluster = AgglomerativeClustering(
#     n_clusters=None, affinity='cosine', distance_threshold=.95, linkage='complete'
# )
cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=2)
cluster.fit(embeddings)

results = bookmark_data.copy().drop(['url_text'], axis=1)
results['label'] = cluster.labels_

with open('embedding-clusters-hier.txt', 'w') as file:
    for cluster in range(cluster.n_clusters_):
        file.write(f'Cluster: {cluster}\n')
        file.write('\n'.join(results['title'][results['label'] == cluster].values))
        file.write('\n')
        file.write('============================================================================\n')
