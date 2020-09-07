from bs4 import BeautifulSoup
from collections import namedtuple
import pickle

Bookmark = namedtuple('Bookmark', ['title', 'url'])

html_file = 'bookmarks_9_5_20.html'
with open(html_file, 'r') as file:
    soup = BeautifulSoup(file, 'lxml')

all_bookmarks = []
for bookmark in soup.find_all('a'):
    all_bookmarks.append(Bookmark(bookmark.text, bookmark.get('href')))

ouput_fn = 'bookmarks.p'
with open(ouput_fn, 'wb') as file:
    pickle.dump(all_bookmarks, file)
