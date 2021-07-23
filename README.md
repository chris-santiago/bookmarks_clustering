## Files
|Filename|Contents|Container|
|--------|--------|--------------|
|bookmarks.p|Pickled list of bookmark titles and URLs|Bookmark|
|bookmarks_data.json|List of bookmark titles, URLs and content|Dictionary|
|bookmarks_data.p|Pickled list of bookmark titles, URLs and content|Dictionary|
|bookmarks_df.p|Pickled dataframe of bookmark titles, URLs and content|DataFrame|
|websites.p|Pickled list of bookmark titles, URLs and content|Website|

## Collections

```python
Bookmark = namedtuple('Bookmark', ['title', 'url'])
Website = namedtuple('Website', ['title', 'url', 'content'])
```



