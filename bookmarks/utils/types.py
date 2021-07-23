from collections import namedtuple

Bookmark = namedtuple('Bookmark', ['title', 'url'])
Website = namedtuple('Website', ['title', 'url', 'content'])
Tokens = namedtuple('Tokens', ['text', 'lemma'])
