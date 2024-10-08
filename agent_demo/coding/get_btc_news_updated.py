# filename: get_btc_news_updated.py

import requests

def get_btc_news(date, api_key):
    url = f'https://newsapi.org/v2/everything?q=bitcoin&from={date}&to={date}&sortBy=publishedAt&apiKey={api_key}'
    
    response = requests.get(url)
    data = response.json()
    
    if 'articles' in data:
        articles = data['articles']
        for article in articles:
            title = article['title']
            source = article['source']['name']
            published_at = article['publishedAt']
            print(f'Title: {title}')
            print(f'Source: {source}')
            print(f'Published At: {published_at}')
            print('---')
    else:
        print("No articles found for the specified date.")

date = '2024-09-01'
api_key = 'fcc5120a43084356afb4643aa4128d1f'
get_btc_news(date, api_key)