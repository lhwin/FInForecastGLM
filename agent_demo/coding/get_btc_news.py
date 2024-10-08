# filename: get_btc_news.py

import requests

def get_btc_news(date, api_key):
    url = f'https://newsapi.org/v2/everything?q=Apple&sortBy=popularity&apiKey={api_key}'
    
    response = requests.get(url)
    data = response.json()
    
    articles = data['articles']
    for article in articles:
        title = article['title']
        source = article['source']['name']
        published_at = article['publishedAt']
        print(f'Title: {title}')
        print(f'Source: {source}')
        print(f'Published At: {published_at}')
        print('---')

date = '2024-09-29'
api_key = 'fcc5120a43084356afb4643aa4128d1f'
get_btc_news(date, api_key)