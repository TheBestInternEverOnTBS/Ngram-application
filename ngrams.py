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
    most_common_unigrams = unigram_freq.most_common(10)
    if not most_common_unigrams:
        print(f"No unigrams found for {url}")
        return None, None

    # Collect all unique words in the most common unigrams
    words, freqs = zip(*most_common_unigrams)

    # Create an array from the frequencies for plotting
    freq_array = np.array(freqs).reshape(-1, 1)

    return freq_array, words

def get_ngram_data(url, n):
    """
    Gets n-gram data from the given URL where 'n' specifies the size of the n-gram.
    For example, 1 would be unigrams, 2 would be bigrams, etc.
    """
    response = requests.get(url)
    if not response.ok:
        print(f"Failed to get data for {url}")
        return Counter()
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extract text from the webpage
    text = ' '.join([element.get_text().strip() for element in soup.find_all("p")])
    
    # Clean and tokenize the text
    tokens = word_tokenize(clean_text(text))
    
    # Generate n-grams
    ngrams = nltk.ngrams(tokens, n)
    ngram_freq = Counter(ngrams)
    
    return ngram_freq

def create_unique_ngram_table(ngrams_1, ngrams_2, top_n=10):
    unique_1 = Counter({ngram: ngrams_1[ngram] for ngram in ngrams_1 if ngram not in ngrams_2})
    unique_2 = Counter({ngram: ngrams_2[ngram] for ngram in ngrams_2 if ngram not in ngrams_1})
    
    # Take the top 'top_n' from each unique Counter
    top_unique_ngrams = (unique_1 + unique_2).most_common(top_n)
    
    # We create a list of ngrams with frequency in URL1 if it's unique to URL1 otherwise 0 and vice versa.
    unique_ngrams = [(ngram, unique_1[ngram] if ngram in unique_1 else 0, unique_2[ngram] if ngram in unique_2 else 0) for ngram, _ in top_unique_ngrams]
    
    # Sort based on the sum of frequencies
    unique_ngrams.sort(key=lambda x: x[1] + x[2], reverse=True)
    
    # Convert to a structured format for plotting
    words, freqs_1, freqs_2 = zip(*unique_ngrams)
    return words, freqs_1, freqs_2

def create_shared_unigram_table(ngrams_1, ngrams_2, top_n=10):
    combined_ngrams = ngrams_1 + ngrams_2
    top_combined_ngrams = Counter(combined_ngrams).most_common(top_n)
    
    # We take the top `top_n` combined ngrams and count frequencies separately.
    shared_ngrams = [(ngram, ngrams_1[ngram], ngrams_2[ngram]) for ngram, _ in top_combined_ngrams if ngram in ngrams_1 and ngram in ngrams_2]
    
    # Sort based on frequencies
    shared_ngrams.sort(key=lambda x: x[1] + x[2], reverse=True)
    
    # Take only the top_n shared ngrams for visualization
    shared_ngrams = shared_ngrams[:top_n]
    
    # Convert to a structured format for plotting
    words, freqs_1, freqs_2 = zip(*shared_ngrams)
    return words, freqs_1, freqs_2

# Function to plot a heatmap of unigram frequencies
def plot_unigram_heatmap(freq_array, words, title, ax):
    sns.heatmap(freq_array, annot=True, fmt=".0f", cmap="coolwarm", xticklabels=["Frequency"], yticklabels=words, ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_yticklabels(words, rotation=0)
    ax.set_xticklabels(["Frequency"], rotation=90)

# User input to choose the n-gram type
n = int(input("Enter 1 for Unigrams, 2 for Bigrams, or 3 for Trigrams: "))

# Retrieve and process the n-gram data for both websites
ngram_data_1 = get_ngram_data("https://www.nbcnews.com/news/world/earthquake-taiwan-tsunami-rcna146140", n)
ngram_data_2 = get_ngram_data("https://edition.cnn.com/asia/live-news/taiwan-earthquake-hualien-tsunami-warning-hnk-intl/index.html", n)

# Get the top 10 most common n-grams for each URL
top_ngrams_1 = ngram_data_1.most_common(10)
top_ngrams_2 = ngram_data_2.most_common(10)

# Create the shared n-gram table with only the top 10 n-grams
words_shared, freqs_1_shared, freqs_2_shared = create_shared_unigram_table(ngram_data_1, ngram_data_2, top_n=10)

# Create the unique n-gram table with only the top 10 unique n-grams
words_unique, freqs_1_unique, freqs_2_unique = create_unique_ngram_table(ngram_data_1, ngram_data_2, top_n=10)

# Setup the figure with 4 vertical subplots
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(5, 10))

# Fetch the two URLs
url_1 = "https://www.blognone.com/node/139037"
url_2 = "https://brandinside.asia/ais-business-martech-expo/"

fig.subplots_adjust(hspace=0.8)

# URL 1 n-grams
words1, values1 = zip(*top_ngrams_1) if top_ngrams_1 else ([], [])
words1 = [' '.join(word) for word in words1]
sns.heatmap(np.array(values1).reshape(-1, 1), annot=True, fmt="d", cmap="Blues", yticklabels=words1, ax=axes[0], cbar=True)
axes[0].set_title(f"Top 10 N-grams from {url_1}")
axes[0].set_xticklabels(['Frequency'], rotation=0)
axes[0].set_yticklabels(words1, rotation=0)

# URL 2 n-grams
words2, values2 = zip(*top_ngrams_2) if top_ngrams_2 else ([], [])
words2 = [' '.join(word) for word in words2]
sns.heatmap(np.array(values2).reshape(-1, 1), annot=True, fmt="d", cmap="Greens", yticklabels=words2, ax=axes[1], cbar=True)
axes[1].set_title(f"Top 10 N-grams from {url_2}")
axes[1].set_xticklabels(['Frequency'], rotation=0)
axes[1].set_yticklabels(words2, rotation=0)

# Shared n-grams heatmap
shared_freqs_array = np.array([freqs_1_shared, freqs_2_shared]).T  # Transpose for vertical orientation
sns.heatmap(shared_freqs_array, annot=True, fmt="d", cmap="Purples", yticklabels=words_shared, ax=axes[2], cbar=True)
axes[2].set_title(f"Top 10 Shared N-grams between URLs")
axes[2].set_xticklabels([url_1, url_2], rotation=0)
axes[2].set_yticklabels(words_shared, rotation=0)

# Unique n-grams heatmap
unique_freqs_array = np.array([freqs_1_unique, freqs_2_unique]).T  # Transpose for vertical orientation
sns.heatmap(unique_freqs_array, annot=True, fmt="d", cmap="Oranges", yticklabels=words_unique, ax=axes[3], cbar=True)
axes[3].set_title("Top 10 Unique N-grams from both URLs")
axes[3].set_xticklabels([f'Unique to {url_1}', f'Unique to {url_2}'], rotation=0)
axes[3].set_yticklabels(words_unique, rotation=0)

plt.tight_layout()
plt.show()