import pandas as pd
from from_html import Bookmark
import pickle
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


input_fn = 'bookmarks.p'
with open(input_fn, 'rb') as file:
    bookmarks = pickle.load(file)

data = pd.DataFrame.from_records(
    bookmarks,
    columns=Bookmark._fields
)


def get_text_content(url):
    url_content = requests.get(url).text
    soup = BeautifulSoup(url_content, 'lxml')
    text = [x.text for x in soup.find_all('p')]
    text.insert(0, soup.title.text)
    return ' '.join(text)


def safe_get_text_content(url):
    try:
        return get_text_content(url)
    except Exception as e:
        return None

# this still takes 3-4 min; try asyncio or aiohttp
with ThreadPoolExecutor(max_workers=100) as executor:
    response = tqdm(executor.map(safe_get_text_content, data['url']))

text = [x for x in response]
data['url_text'] = text

ouput_fn = 'bookmarks_data.p'
with open(ouput_fn, 'wb') as file:
    pickle.dump(data, file)
