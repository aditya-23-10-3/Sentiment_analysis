import numpy as np
import pandas as pd
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
import nltk
nltk.download()

# Download necessary NLTK packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
data = pd.read_csv('data/youtube_comments.csv')

# Drop unnecessary columns
data1 = data.drop(['Likes', 'Time', 'user', 'UserLink'], axis=1)


# Sentiment Analysis with Vader for initial labeling
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sentiments = SentimentIntensityAnalyzer()
data1["Compound"] = [sentiments.polarity_scores(i)["compound"] for i in data1["Comment"]]

# Label data based on compound scores
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

data1["Sentiment"] = data1["Compound"].apply(classify_sentiment)
data2 = data1.drop(['Compound'], axis=1)

# Preprocessing function for text
stop_words = stopwords.words('english')
lzr = WordNetLemmatizer()

def text_processing(text):
    """
    Function to clean and preprocess text data
    """
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'\W', ' ', text)
    text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text) if word not in stop_words])
    return text


# Apply preprocessing
data_copy = data2.copy()
data_copy['Comment'] = data_copy['Comment'].apply(text_processing)

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])

# Data balancing
df_negative = data_copy[data_copy['Sentiment'] == 0]
df_neutral = data_copy[data_copy['Sentiment'] == 1]
df_positive = data_copy[data_copy['Sentiment'] == 2]

df_negative_upsampled = resample(df_negative, replace=True, n_samples=205, random_state=42)
df_neutral_upsampled = resample(df_neutral, replace=True, n_samples=205, random_state=42)

final_data = pd.concat([df_negative_upsampled, df_neutral_upsampled, df_positive])

# Feature extraction with TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(final_data['Comment']).toarray()
y = final_data['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a Naive Bayes Classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Save the model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

print("Model and vectorizer saved successfully!")
