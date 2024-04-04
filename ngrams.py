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

def get_unigram_data(url):
    response = requests.get(url)
    if not response.ok:
        print(f"Failed to get data for {url}")
        return Counter()
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract text from the webpage
    text = ' '.join([element.get_text().strip() for element in soup.find_all("p")])

    # Clean and tokenize the text
    tokens = word_tokenize(clean_text(text))

    # Define N-gram length as 1 for unigrams
    N = 1

    # Count unigrams with frequency
    unigram_freq = Counter(tokens)

    return unigram_freq

def create_shared_unigram_table(unigrams_1, unigrams_2, top_n=10):
    # Get the unique set of words from both URLs
    all_words = set(unigrams_1.keys()) | set(unigrams_2.keys())

    # Initialize a plot-friendly format for the shared table
    freq_table = []

    for word in all_words:
        freq_1 = unigrams_1.get(word, 0)
        freq_2 = unigrams_2.get(word, 0)
        
        # Append to the table
        freq_table.append([word, freq_1, freq_2])
    
    # Sort the table based on the sum of frequencies from both URLs and select the top 'n'
    freq_table.sort(key=lambda x: x[1] + x[2], reverse=True)
    freq_table = freq_table[:top_n]  # Take only the top 'n' frequencies
    
    # Convert to a structured format for plotting
    words, freqs_1, freqs_2 = zip(*freq_table)
    return words, freqs_1, freqs_2

# Function to plot a heatmap of unigram frequencies
def plot_unigram_heatmap(freq_array, words, title, ax):
    sns.heatmap(freq_array, annot=True, fmt=".0f", cmap="coolwarm", xticklabels=["Frequency"], yticklabels=words, ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_yticklabels(words, rotation=0)
    ax.set_xticklabels(["Frequency"], rotation=90)

# Retrieve and process the unigram data for both websites
unigram_data_1 = get_unigram_data("https://www.nbcnews.com/news/world/earthquake-taiwan-tsunami-rcna146140")
unigram_data_2 = get_unigram_data("https://edition.cnn.com/asia/live-news/taiwan-earthquake-hualien-tsunami-warning-hnk-intl/index.html")

# Get the top 10 most common unigrams for each URL
freqs_url1 = unigram_data_1.most_common(10)
freqs_url2 = unigram_data_2.most_common(10)

# Create the shared unigram table with only the top 10 unigrams
words_shared, freqs_1_shared, freqs_2_shared = create_shared_unigram_table(unigram_data_1, unigram_data_2, top_n=10)

# The `freqs_url1` and `freqs_url2` are lists of tuples, we need to separate the words and the frequencies.
words_url1, values_url1 = zip(*freqs_url1) if freqs_url1 else ([], [])
words_url2, values_url2 = zip(*freqs_url2) if freqs_url2 else ([], [])

# Setup the figure with 3 vertical subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 15))

# URL 1 Unigrams
sns.heatmap(np.array(values_url1).reshape(-1, 1), annot=True, fmt="d", cmap="Blues", yticklabels=words_url1, ax=axes[0])
axes[0].set_title("URL 1 Unigrams")
axes[0].set_xticklabels(['Frequency'], rotation=90)
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

# URL 2 Unigrams
sns.heatmap(np.array(values_url2).reshape(-1, 1), annot=True, fmt="d", cmap="Greens", yticklabels=words_url2, ax=axes[1])
axes[1].set_title("URL 2 Unigrams")
axes[1].set_xticklabels(['Frequency'], rotation=90)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

# Shared Unigrams
shared_freqs_array = np.array([freqs_1_shared, freqs_2_shared]).T  # Transpose for vertical orientation
sns.heatmap(shared_freqs_array, annot=True, fmt="d", cmap="Oranges", yticklabels=words_shared, ax=axes[2], cbar=True)
axes[2].set_title("Shared Unigrams")
axes[2].set_xticklabels(['URL 1', 'URL 2'], rotation=90)
axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()