import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure the punkt tokenizer is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

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

def get_unigram_heatmap_data(url):
    response = requests.get(url)
    if not response.ok:
        print(f"Failed to get data for {url}")
        return None, None
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract titles from the webpage
    titles = [title.get_text().strip() for title in soup.find_all("p")]

    # Join the titles into a single string for tokenization
    all_titles_text = ' '.join(titles)

    # Tokenize the text
    tokens = word_tokenize(clean_text(all_titles_text))

    # Define N-gram length as 1 for unigrams
    N = 1

    # Count unigrams with frequency
    unigram_freq = Counter(tokens)

    # Take only the 50 most common unigrams
    most_common_unigrams = unigram_freq.most_common(50)
    if not most_common_unigrams:
        print(f"No unigrams found for {url}")
        return None, None

    # Collect all unique words in the most common unigrams
    words, freqs = zip(*most_common_unigrams)

    # Create an array from the frequencies for plotting
    freq_array = np.array(freqs).reshape(-1, 1)

    return freq_array, words


# Define your URLs
urls = ["https://anzphotobookaward.com/", "https://www.theamericancollege.edu/"] 
colormaps = ['Set1', 'Set2']  # Using different colormaps

# Setup the figure for the subplots
fig, axes = plt.subplots(1, len(urls), figsize=(20, len(urls) * 10))

for ax, url, cmap in zip(axes.flat, urls, colormaps):
    freq_array, words = get_unigram_heatmap_data(url)
    if freq_array is not None and words is not None:
        # Create a seaborn heatmap
        sns.heatmap(freq_array, annot=True, fmt=".0f", cmap=cmap, yticklabels=words, ax=ax, cbar=False)
        ax.set_title(f'50 Most Common Unigrams for {url}')
        ax.set_yticklabels(words, rotation=0)
        ax.set_xticklabels(["Frequency"], rotation=0)
    else:
        ax.set_visible(False)  # Hide the axis if no data is available

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()