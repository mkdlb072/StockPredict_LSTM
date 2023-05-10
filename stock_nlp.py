import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime
from datetime import timedelta
import gensim
from gensim.models import Word2Vec
import requests
import json
from GoogleNews import GoogleNews
import math

stock_name = 'AAPL'
news_period = 300


def get_news_headlines(stock_name, news_period):
    # Set API key and endpoint
    api_key = '0aa59831ff90443b997a71e5e4aa07fe'
    endpoint = 'https://newsapi.org/v2/everything'

    # Set stock symbol and date range
    query = stock_name
    from_date = datetime.now() - timedelta(days=news_period)
    to_date = datetime.now()

    # Format date range for API query
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')

    # Set query parameters
    params = {
        'q': query,
        'searchIn': 'title',
        'from': from_date_str,
        'to': to_date_str,
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': api_key,
    }

    # Make API request
    response = requests.get(endpoint, params=params)

    # Parse JSON response
    json_data = json.loads(response.text)

    # Extract headlines
    headlines_title = []
    headlines_date = []
    for article in json_data['articles']:
        title = article['title']
        headlines_title.append(title)
        publish_date = article['publishedAt']
        headlines_date.append(publish_date)

    stock_news = pd.DataFrame({'News_title': headlines_title, 'News_date': headlines_date})
    stock_news['News_date'] = pd.to_datetime(stock_news['News_date'])
    stock_news['News_date'] = stock_news['News_date'].dt.strftime('%d-%m-%Y')
    return stock_news


def get_mediastack_headlines(stock_name, news_period):
    # Set up API key and base URL for MediaStack API
    api_key = 'abb92ad755982808ce855486780e4460'
    base_url = "http://api.mediastack.com/v1/news"
    from_date = (datetime.date.today() - datetime.timedelta(days=news_period)).strftime('%Y-%m-%d')
    to_date = datetime.date.today().strftime('%Y-%m-%d')

    # Set up parameters for API request
    params = {
        "access_key": api_key,
        "keywords": f'{stock_name} -HotOptions',
        "languages": "en",
        "sort": "published_desc",
        "date": f'{from_date},{to_date}',
        "limit": 100,
    }

    # Send API requests and store articles in a list
    response = requests.get(base_url, params=params)

    # Parse JSON response
    json_data = json.loads(response.text)

    # Extract headlines
    headlines_title = []
    headlines_date = []
    for article in json_data['data']:
        title = article['title']
        headlines_title.append(title)
        publish_date = article['published_at']
        headlines_date.append(publish_date)

    stock_news = pd.DataFrame({'News_title': headlines_title, 'News_date': headlines_date})
    stock_news['News_date'] = pd.to_datetime(stock_news['News_date'])
    stock_news['News_date'] = stock_news['News_date'].dt.strftime('%d-%m-%Y')
    return stock_news


# def get_google_news(stock_name, news_period):
#     googlenews = GoogleNews(lang='en', region='US', period=f'{news_period}d')
#     googlenews.get_news(stock_name)
#     googlenews.results(sort=True)


# df_stock_news = get_news_headlines(stock_name, news_period)

googlenews = GoogleNews(lang='en', region='US', period=f'{news_period}d')
googlenews.search(stock_name)
result_cnt = googlenews.total_count()
result = googlenews.results()

df_stock_news = get_news_headlines(stock_name, news_period)
# print(df_stock_news)

# # Scrape financial news headlines from Reuters
# url = 'https://www.reuters.com/news/archive/businessNews?view=page&page={page}&pageSize=10'
# headlines = []
# for i in range(1, 3):
#     page = requests.get(url.format(page=i))
#     soup = BeautifulSoup(page.content, 'html.parser')
#     headlines += [h.get_text() for h in soup.find_all('h3')]
#
# # Text preprocessing
# nltk.download('stopwords')
# nltk.download('wordnet')
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# clean_headlines = []
# for headline in headlines:
#     words = nltk.word_tokenize(headline)
#     words = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]
#     words = [lemmatizer.lemmatize(w) for w in words]
#     clean_headlines.append(' '.join(words))
#
# # Feature extraction
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(clean_headlines).toarray()
#
# # Model training
# y = [1, 0] * (len(X) // 2)  # Dummy labels for demonstration
# model = LinearRegression()
# model.fit(X, y)
#
# # Predict whether today's news headlines indicate a positive or negative stock market
# today = datetime.datetime.now().strftime('%Y-%m-%d')
# page = requests.get(url.format(page=1))
# soup = BeautifulSoup(page.content, 'html.parser')
# today_headlines = [h.get_text() for h in soup.find_all('h3') if today in h.get_text()]
# today_clean_headlines = []
# for headline in today_headlines:
#     words = nltk.word_tokenize(headline)
#     words = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]
#     words = [lemmatizer.lemmatize(w) for w in words]
#     today_clean_headlines.append(' '.join(words))
# today_X = vectorizer.transform(today_clean_headlines).toarray()
# today_prediction = model.predict(today_X)
# print(today_prediction)
