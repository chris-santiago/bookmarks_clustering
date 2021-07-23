import pickle
from typing import Union, List, Optional
import pathlib
from concurrent.futures import ThreadPoolExecutor
import time

import requests
from bs4 import BeautifulSoup

from bookmarks.utils.types import Bookmark, Website

HERE = pathlib.Path(__file__)
CURR_DIR = HERE.parent
PARENT = CURR_DIR.parent


class HtmlReader:
    def __init__(
            self, bookmarks: Optional[List[Bookmark]] = None,
            filepath: Optional[Union[pathlib.Path, str]] = None
    ):
        if not bookmarks and not filepath:
            raise ValueError('Must pass either bookmarks or filepath parameter.')
        self.bookmarks = bookmarks if bookmarks else self.from_pickle(filepath)
        self.html = []
        self.websites = []

    @staticmethod
    def from_pickle(filepath: Union[pathlib.Path, str]) -> List[Bookmark]:
        with open(filepath, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def fetch(bookmark: Bookmark) -> Optional[Website]:
        try:
            return Website(
                bookmark.title,
                bookmark.url,
                requests.get(bookmark.url).text
            )
        except Exception:
            return Website(
                bookmark.title,
                bookmark.url,
                None
            )

    def fetch_all(self) -> "HtmlReader":
        with ThreadPoolExecutor(max_workers=200) as executor:
            response = executor.map(self.fetch, self.bookmarks)
            self.html.extend([*response])
        return self

    @staticmethod
    def parse(html):
        try:
            soup = BeautifulSoup(html, 'lxml')
            text = [x.text for x in soup.find_all('p')]
            text.insert(0, soup.title.text)
            return ' '.join(text)
        except Exception:
            return None

    def parse_all(self) -> "HtmlReader":
        self.websites.extend(
            [
                Website(web.title, web.url, self.parse(web.content))
                if web.content else web for web in self.html
            ]
        )
        return self

    def to_pickle(self, filepath: Union[pathlib.Path, str]):
        with open(filepath, 'wb') as file:
            pickle.dump(self.websites, file)


if __name__ == '__main__':
    start = time.time()
    html = HtmlReader(filepath=PARENT.joinpath('bookmarks.p'))
    html.fetch_all()
    duration = time.time() - start
    print(f"Downloaded {len(html.bookmarks)} sites in {round(duration/60, 1)} minutes.")
    html.parse_all().to_pickle(PARENT.joinpath('data_test.p'))
