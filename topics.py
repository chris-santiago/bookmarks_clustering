import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import re

bookmark_file = 'bookmarks_df.p'
bookmark_data = pd.read_pickle(bookmark_file).drop_duplicates().dropna().reset_index(drop=True)
titles = bookmark_data['title']
contents = bookmark_data['url_text']
# text = bookmark_data['title'] + bookmark_data['url_text']
# text = [' '.join(re.findall(r'\w+', t.lower())) for t in text]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(contents, show_progress_bar=True)

red = umap.UMAP(n_components=64, metric='euclidean')
red_embed = red.fit_transform(embeddings)

sc = StandardScaler()
red_embed = sc.fit_transform(red_embed)

clust = hdbscan.HDBSCAN(
    min_cluster_size=3,
    alpha=.01,
    # cluster_selection_epsilon=0.2,
)
clust.fit(red_embed)

res = pd.DataFrame({
    'title': titles,
    'cluster': clust.labels_
})

res.groupby('cluster').size()


c = CountVectorizer(
    strip_accents='ascii',
    stop_words='english',
    ngram_range=(1, 3)
)

mask = res['cluster'] == 40
x = c.fit_transform([p for p in res.loc[mask, 'title']])
pd.DataFrame(x.toarray(), columns=c.get_feature_names_out()).sum(0).sort_values(ascending=False).head(5)


def get_topic(text):
    cv = CountVectorizer(
        strip_accents='ascii',
        stop_words='english',
        ngram_range=(1, 3)
    )
    counts = cv.fit_transform(text)
    df = pd.DataFrame(
        counts.toarray(),
        columns=cv.get_feature_names_out()
    ).sum(0).sort_values(ascending=False).head(5)
    return list(df.index)


topics = res.groupby('cluster').agg(get_topic).to_dict()['title']
res['topics'] = res['cluster'].map(topics)