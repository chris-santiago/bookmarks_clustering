import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from transformers import pipeline
import tqdm


def get_summaries(data, model=None, write=True, max_length=1024):
    data = data.where(bookmark_data['url_text'].str.len() > 15, bookmark_data['title'], axis=0)
    sentences = data['url_text'].to_list()
    if model is None:
        model = pipeline("summarization", model="facebook/bart-large-cnn")

    summaries = []
    for i, sent in tqdm.tqdm(enumerate(sentences), position=0):
        if len(sent) > max_length:
            sent = sent[:max_length]
        try:
            summaries.append({'index': i, 'summary': model(sent)[0]['summary_text']})
        except Exception as e:
            print(e)
            summaries.append({'index': i, 'summary': None})
    data['summary'] = [x['summary'] for x in summaries]
    if not write:
        return data
    data.to_pickle('bookmarks_summaries_df.p')
    return data


bookmark_file = 'bookmarks_df.p'
bookmark_data = pd.read_pickle(bookmark_file).drop_duplicates().dropna().reset_index(drop=True)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
bookmark_data = get_summaries(bookmark_data, model=summarizer)

bookmark_text = bookmark_data['summary']
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

with open('embedding-clusters-hier-summaries.txt', 'w') as file:
    for cluster in range(cluster.n_clusters_):
        file.write(f'Cluster: {cluster}\n')
        file.write('\n'.join(results['title'][results['label'] == cluster].values))
        file.write('\n')
        file.write('============================================================================\n')
