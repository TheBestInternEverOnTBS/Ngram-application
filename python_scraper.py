import requests
from bs4 import BeautifulSoup
from collections import Counter
from pythainlp.tokenize import word_tokenize
import nltk

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove HTML tags and attributes
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize into words
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Lemmatization (optional, can skip or replace with stemming)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def get_ngram_data(url, n):
    response = requests.get(url)
    if not response.ok:
        print(f"Failed to get data for {url}")
        return Counter()
    
    soup = BeautifulSoup(response.content, "html.parser")
    text = ' '.join([element.get_text().strip() for element in soup.find_all("p")])
    text = clean_text(text)  # Assuming you have a suitable cleaning function for Thai
    
    # Use pythainlp for tokenizing Thai language text
    tokens = word_tokenize(text, engine='newmm')  # 'newmm' is a popular engine for Thai
    
    # Generate n-grams
    ngrams = nltk.ngrams(tokens, n)
    ngram_freq = Counter(ngrams)
    
    return ngram_freq

get_ngram_data("https://anzphotobookaward.com/", 1)