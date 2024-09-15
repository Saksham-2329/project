import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Load the CSV file
data = pd.read_csv('spam_emails.csv')

# Plot the distribution of ham and spam emails
sns.countplot(x='label', data=data)
plt.show()

# Remove punctuation and convert to lowercase
data['text'] = data['text'].str.replace('[^\w\s]', '')
data['text'] = data['text'].str.lower()

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Create a word cloud for ham and spam emails
ham_msg = data[data['label'] == 'ham']
spam_msg = data[data['label'] == 'spam']

def plot_word_cloud(data, typ):
    email_corpus = ' '.join(data['text'])
    plt.figure(figsize=(7, 7))
    wc = WordCloud(background_color='black', max_words=100, width=800, height=400, collocations=False).generate(email_corpus)
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {typ} emails', fontsize=15)
    plt.axis('off')
    plt.show()

plot_word_cloud(ham_msg, typ='Non-Spam')
plot_word_cloud(spam_msg, typ='Spam')