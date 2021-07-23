import string

import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()
    for token in nltk.word_tokenize(text):
        if token in string.punctuation:
            continue
        yield stem.stem(token)


def get_model(corpus):
    corpus = [list(tokenize(doc)) for doc in corpus]
    corpus = [TaggedDocument(words, ['d{}'.format(idx)]) for idx, words in enumerate(corpus)]
    return Doc2Vec(corpus)


TEXT = """
Bank of America Corp. BAC -4.67% said the economic rebound helped to more than double its profit, but low rates weighed on its revenue.

The nation’s second-largest bank by assets posted earnings Wednesday of $9.22 billion in the second quarter, up from $3.53 billion a year earlier. The bank earned $1.03 per share, beating the 77 cents forecast by analysts polled by FactSet.

Bank of America’s bottom line was lifted by its decision to release $2.2 billion of reserves it had set aside during the depths of the coronavirus pandemic to protect against a wave of soured loans. Like peers including JPMorgan Chase & Co., Bank of America has been releasing its loan-loss stockpiles as the U.S. economy rebounds.

“Consumer spending has significantly surpassed pre-pandemic levels, deposit growth is strong and loan levels have begun to grow,” CEO Brian Moynihan said.

Bank of America and its peers sit at the center of the U.S. economy. They have benefited from the reopening of businesses, record-high stock prices and people’s increased willingness to spend after a year of hunkering down.
"""

model = get_model([TEXT])
