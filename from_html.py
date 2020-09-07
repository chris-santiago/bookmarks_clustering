from bs4 import BeautifulSoup
from collections import namedtuple
import pickle

Bookmark = namedtuple('Bookmark', ['title', 'url'])


def get_folder(folder_name, soup):
    res = []
    for bookmark in soup.find('h3', text=folder_name).find_next('dl').find_all('a'):
        res.append(Bookmark(bookmark.text, bookmark.get('href')))
    return res


def get_all(soup):
    res = []
    for bookmark in soup.find_all('a'):
        res.append(Bookmark(bookmark.text, bookmark.get('href')))
    return res


html_file = 'bookmarks_9_5_20.html'
with open(html_file, 'r') as file:
    soup = BeautifulSoup(file, 'lxml')

edu = get_folder('EDU', soup)
work = get_folder('Work', soup)
all_bookmarks = edu + work

ouput_fn = 'bookmarks.p'
with open(ouput_fn, 'wb') as file:
    pickle.dump(all_bookmarks, file)


