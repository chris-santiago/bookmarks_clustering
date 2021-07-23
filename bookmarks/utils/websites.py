from collections import namedtuple
import pickle

from bookmarks.utils.types import Website

with open('bookmarks_data.p', 'rb') as fp:
    bookmarks = pickle.load(fp)

data = [Website(d['title'], d['url'], d['url_text']) for d in bookmarks]

with open('websites.p', 'wb') as fp:
    pickle.dump(data, fp)
