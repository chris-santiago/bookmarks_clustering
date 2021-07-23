import pickle

from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

stop_words = stopwords.words('english')

input_fn = 'bookmarks_data.p'
with open(input_fn, 'rb') as file:
    data = pickle.load(file)

data = data.dropna().drop_duplicates().reset_index(drop=True)

count_vec = CountVectorizer(stop_words=stop_words)
X = count_vec.fit_transform(data['url_text'])

lda = LatentDirichletAllocation(n_components=50)
lda.fit(X)
feature_names = count_vec.get_feature_names()


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


print_top_words(lda, feature_names, 5)

topic_values = lda.transform(X)
data['topic'] = topic_values.argmax(axis=1)

with open('topics.txt', 'w') as file:
    for topic in range(lda.n_components):
        file.write(f'Topic: {topic} ({" ".join([feature_names[i] for i in lda.components_[topic].argsort()[:-3 - 1:-1]])})\n')
        file.write('\n'.join(data['title'][data['topic'] == topic].values))
        file.write('\n')
        file.write('============================================================================\n')
