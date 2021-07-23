import json
import pickle
import unicodedata

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BookmarkProcessor:
    def __init__(self, bookmark_data: str):
        self.file = bookmark_data

    def raw(self):
        for page in self.loader():
            yield page

    def loader(self):
        stem = self.file.split('.')[-1]
        loaders = {
            'json': self.json_loader,
            'p': self.pickle_loader
        }
        return loaders[stem](self.file)

    @staticmethod
    def pickle_loader(file: str):
        with open(file, 'r') as fp:
            data = pickle.load(fp)
        for line in data:
            yield line

    @staticmethod
    def json_loader(file: str):
        with open(file, 'r') as fp:
            data = json.load(fp)
        for line in data:
            yield line

    def sents(self):
        for bookmark in self.raw():
            for sentence in sent_tokenize(bookmark['url_text']):
                yield sentence

    def words(self):
        for sentence in self.sents():
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self):
        for bookmark in self.raw():
            try:
                yield [pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(bookmark['url_text'])]
            except TypeError:
                continue


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self,):
        self.stopwords = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        documents = X
        self.output_features_ = [self.normalize(doc) for doc in documents]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        return self.output_features_

    @staticmethod
    def is_punc(token: str):
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token: str):
        return token.lower() in self.stopwords

    def lemmatize(self, token, tag):
        tags = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ,
        }
        return self.lemmatizer.lemmatize(word=token, pos=tags.get(tag[0], wordnet.NOUN))

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for sentence in document
            for (token, tag) in sentence
            if not self.is_punc(token) and not self.is_stopword(token)
        ]


if __name__ == '__main__':
    file = './bookmarks_data.json'
    p = BookmarkProcessor(file)
    docs = [*p.tokenize()]
    normalizer = TextNormalizer()
    normalizer.fit(docs)