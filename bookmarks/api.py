import configparser
import pathlib

import requests

HOME = pathlib.Path.home()


def query(text, model=None):
    config = configparser.ConfigParser()
    config.read(HOME.joinpath('.hugging-face', 'config'))

    if model:
        endpoint = f"https://api-inference.huggingface.co/models/{model}"
    else:
        endpoint = config['bookmarks']['url']

    payload = {"inputs": text}
    response = requests.post(
        url=endpoint,
        headers={"Authorization": f"Bearer {config['bookmarks']['token']}"},
        json=payload
    )
    return response.json()
