"""
Using Spacy to lemmatize takes WAY too long-- 18 minutes.

"""

from typing import Iterable, List, Optional, Union
import pathlib
import pickle
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

from bookmarks.utils.types import Bookmark, Website, Tokens

d = ['Base baseball base then and if throw throwing catch catching throw.']


def bookmark_tokenizer(doc: str, model='en_core_web_sm') -> List[str]:
    nlp = spacy.load(model, disable=['tok2vec'])
    return [token.lemma_.lower() for token in nlp(doc) if not token.is_stop and token.is_alpha]


vec = TfidfVectorizer(tokenizer=bookmark_tokenizer)
# vec.build_tokenizer()
vec.fit(d)
X = pd.DataFrame(
    vec.transform(d).toarray(),
    columns=sorted(vec.vocabulary_.keys())
)

HERE = pathlib.Path(__file__)
CURR_DIR = HERE.parent
PARENT = CURR_DIR.parent


class BookmarkProcessor:
    def __init__(self, bookmarks: Optional[List] = None, filepath: Optional[Union[pathlib.Path, str]] = None):
        if not bookmarks and not filepath:
            raise ValueError('Must pass either bookmarks or filepath parameter.')
        self.bookmarks = (b for b in bookmarks) if bookmarks else self.from_pickle(filepath)
        self.vec = TfidfVectorizer(tokenizer=bookmark_tokenizer)

    @staticmethod
    def from_pickle(filepath: Union[pathlib.Path, str]) -> List[Website]:
        with open(filepath, 'rb') as file:
            websites = pickle.load(file)
            for web in websites:
                yield web

    @staticmethod
    def to_dataframe(bookmarks: List) -> pd.DataFrame:
        return pd.DataFrame.from_records(bookmarks, columns=['title', 'url', 'content'])

    def docs(self):
        for web in self.bookmarks:
            if web.content:
                yield Website(web.title, web.url, web.title + web.content)
            yield Website(web.title, web.url, web.title)

    def tokenize(self):
        for doc in self.docs():
            yield Website(doc.title, doc.url, bookmark_tokenizer(doc.content))

    def to_pickle(self, filepath: Optional[Union[pathlib.Path, str]]):
        tokenized = [*self.tokenize()]
        with open(filepath, 'wb') as fp:
            pickle.dump(tokenized, fp)

    def transform(self):
        docs = [*self.docs()]
        return pd.DataFrame(
            self.vec.fit_transform(docs).toarray(),
            columns=sorted(self.vec.vocabulary_.keys())
        )


if __name__ == '__main__':
    start = time.time()
    processor = BookmarkProcessor(filepath=PARENT.joinpath('websites.p'))
    processor.to_pickle('tokenized.p')
    duration = time.time() - start
    print(f"Process took {round(duration / 60, 1)} minutes.")
