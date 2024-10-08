# filename: stock_info.py
import requests

def get_stock_info(stock_code):
    api_key = 'YOUR_API_KEY'  # Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock_code}&apikey={api_key}'
    
    response = requests.get(url)
    data = response.json()
    
    if 'Global Quote' in data:
        stock_info = data['Global Quote']
        return stock_info
    else:
        return "Stock information not found"

stock_code = '600519'  # Replace with the actual stock code
print(get_stock_info(stock_code))