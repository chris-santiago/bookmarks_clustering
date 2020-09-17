import pandas as pd
from from_html import Bookmark
import pickle
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import time


input_fn = 'bookmarks.p'
with open(input_fn, 'rb') as file:
    bookmarks = pickle.load(file)

data = pd.DataFrame.from_records(
    bookmarks,
    columns=Bookmark._fields
)


def get_text_content(html):
    try:
        soup = BeautifulSoup(html, 'lxml')
        text = [x.text for x in soup.find_all('p')]
        text.insert(0, soup.title.text)
        return ' '.join(text)
    except Exception:
        return None


def fetch(url):
    try:
        return requests.get(url).text
    except Exception:
        return None


def fetch_all(urls):
    with ThreadPoolExecutor(max_workers=200) as executor:
        response = executor.map(fetch, urls)
        return [*response]


def dump(data, file):
    with open(file, 'wb') as fp:
        pickle.dump(data, fp)


if __name__ == '__main__':
    start = time.time()
    html = fetch_all(data['url'])
    duration = time.time() - start
    print(f"Downloaded {len(data['url'])} sites in {duration / 60} minutes.")
    data['url_text'] = [get_text_content(x) for x in html]
    ouput_fn = 'bookmarks_data.p'
    dump(data, ouput_fn)
