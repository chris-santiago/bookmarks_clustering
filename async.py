"""Takes 272 seconds"""

import asyncio
import aiohttp
import time
import pandas as pd
import pickle
from from_html import Bookmark


input_fn = 'bookmarks.p'
with open(input_fn, 'rb') as file:
    bookmarks = pickle.load(file)

data = pd.DataFrame.from_records(
    bookmarks,
    columns=Bookmark._fields
)


async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()


async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(fetch(session, url))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == '__main__':
    urls = data['url']
    start = time.time()
    asyncio.get_event_loop().run_until_complete(fetch_all(urls))
    duration = time.time() - start
    print(f"Downloaded {len(urls)} sites in {duration} seconds.")
