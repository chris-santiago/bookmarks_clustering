import pickle
from typing import Optional, List, Union
import pathlib

from bs4 import BeautifulSoup

from bookmarks.utils.types import Bookmark

HERE = pathlib.Path(__file__)
CURR_DIR = HERE.parent
PARENT = CURR_DIR.parent


class BookmarkReader:
    def __init__(self, filepath: str):
        self.filename = filepath
        self.soup = None
        self.bookmarks = []

    def make_soup(self) -> "BookmarkReader":
        with open(self.filename, 'r') as file:
            self.soup = BeautifulSoup(file, 'lxml')
        return self

    def get_folder(self, folder_name: str):
        return [
            Bookmark(bookmark.text, bookmark.get('href'))
            for bookmark in self.soup.find('h3', text=folder_name).find_next('dl').find_all('a')
        ]

    def get_all(self):
        return [
            Bookmark(bookmark.text, bookmark.get('href')) for bookmark in self.soup.find_all('a')
        ]

    def get(self, folders: Optional[List[str]] = None) -> "BookmarkReader":
        if not self.soup:
            self.make_soup()
        if folders:
            for folder in folders:
                self.bookmarks.extend(self.get_folder(folder))
            return self
        self.bookmarks.extend(self.get_all())
        return self

    def to_pickle(self, filepath: Union[pathlib.Path, str]):
        with open(filepath, 'wb') as file:
            pickle.dump(self.bookmarks, file)


if __name__ == '__main__':
    html_file = PARENT.joinpath('bookmarks_10_24_20.html')
    out_file = PARENT.joinpath('bookmarks.p')
    reader = BookmarkReader(html_file)
    reader.get(folders=['EDU', 'Work']).to_pickle(out_file)
