import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime

# Scrape financial news headlines from Reuters
url = 'https://www.reuters.com/news/archive/businessNews?view=page&page={page}&pageSize=10'
headlines = []
for i in range(1, 3):
    page = requests.get(url.format(page=i))
    soup = BeautifulSoup(page.content, 'html.parser')
    headlines += [h.get_text() for h in soup.find_all('h3')]

# Text preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
clean_headlines = []
for headline in headlines:
    words = nltk.word_tokenize(headline)
    words = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    clean_headlines.append(' '.join(words))

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_headlines).toarray()

# Model training
y = [1, 0] * (len(X) // 2)  # Dummy labels for demonstration
model = LinearRegression()
model.fit(X, y)

# Predict whether today's news headlines indicate a positive or negative stock market
today = datetime.datetime.now().strftime('%Y-%m-%d')
page = requests.get(url.format(page=1))
soup = BeautifulSoup(page.content, 'html.parser')
today_headlines = [h.get_text() for h in soup.find_all('h3') if today in h.get_text()]
today_clean_headlines = []
for headline in today_headlines:
    words = nltk.word_tokenize(headline)
    words = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    today_clean_headlines.append(' '.join(words))
today_X = vectorizer.transform(today_clean_headlines).toarray()
today_prediction = model.predict(today_X)
print(today_prediction)

