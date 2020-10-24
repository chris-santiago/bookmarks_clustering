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


def bookmarks_to_pickle(html_file, out_file='bookmarks.p'):
    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, 'lxml')
    edu = get_folder('EDU', soup)
    work = get_folder('Work', soup)
    all_bookmarks = edu + work
    with open(out_file, 'wb') as file:
        pickle.dump(all_bookmarks, file)


if __name__ == '__main__':
    html_file = 'bookmarks_10_24_20.html'
    bookmarks_to_pickle(html_file)
