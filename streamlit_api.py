import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import os
from nltk import FreqDist, ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string
from nltk.corpus import wordnet
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import io
from tensorflow import summary
from sklearn.metrics.pairwise import linear_kernel
nltk.download('punkt')
nltk.download('stopwords')
import emoji


# Load the trained models from the notebook
rating_model = joblib.load("rating_model.pkl")
#sentiment_model = joblib.load("sentiment_model.pkl")

# Load dataframe to access to restaurant names
df = pd.read_csv('preprocessed_data_250_each.csv')

# Function to handle negation using WordNet
def handle_negation(sentence):
    result_sentence = list(sentence)
    
    for i in range(len(result_sentence)):
        if result_sentence[i - 1] in ['not', "n't"]:
            antonyms = []
            for syn in wordnet.synsets(result_sentence[i]):
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())

            if antonyms:
                result_sentence[i] = antonyms[0]

    result_sentence = [word for word in result_sentence if word != '']
    return result_sentence


# Preprocessing definition 
def process_text(text):

    # Convert emojis to text
    text = emoji.demojize(text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation and special characters
    tokens = [token for token in tokens if token.isalnum()]
    
    # Remove numbers
    tokens = [token for token in tokens if not token.isdigit()]
    
    # Handle negation
    tokens = handle_negation(tokens)
    
    # Correction 
    blob = TextBlob(' '.join(tokens))
    tokens = blob.correct().words
     
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming 
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join the tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text

# Function to get top restaurants from a query 

def get_top_restaurants(query, restaurant_df, top_n=5):
    # Use TF-IDF vectorization to represent the restaurants' information and the query
    vectorizer = TfidfVectorizer()
    corpus = restaurant_df['Information'].fillna('') + [query]
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate cosine similarity between the query and each restaurant
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Get the indices of top N similar restaurants
    top_indices = similarities.argsort()[0][::-1][:top_n]

    # Get the top N similar restaurants with the highest average rating
    top_restaurants = restaurant_df.iloc[top_indices].sort_values(by='Average Rating', ascending=False).head(top_n)

    return top_restaurants

def question_answer(user_query):
    # Colonnes à utiliser pour la recherche de texte
    text_columns = ['Information', 'Processed Review Title', 'Processed Review Text']

    # Fonction de tokenisation
    def tokenize(text):
        return word_tokenize(text.lower())  # Ajoutez ici d'autres étapes de prétraitement si nécessaire

    # Agréger les avis et les Average Rating pour chaque restaurant
    aggregated_reviews = df.groupby('Restaurant Name')[text_columns + ['Average Rating']].agg(lambda x: ' '.join(x.dropna()) if x.name in text_columns else x.mean()).reset_index()

    # Concaténer les colonnes de texte pour construire les documents
    aggregated_reviews['Combined_Text'] = aggregated_reviews[text_columns].apply(lambda x: ' '.join(x), axis=1)

    # Appliquer la tokenisation aux documents
    aggregated_reviews['Combined_Text_Tokenized'] = aggregated_reviews['Combined_Text'].apply(tokenize)

    # Appliquer la tokenisation à la requête de l'utilisateur
    user_query_tokenized = tokenize(user_query)

    # Utiliser TF-IDF pour vectoriser les documents
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(aggregated_reviews['Combined_Text_Tokenized'].apply(lambda x: ' '.join(x)))

    # Vectoriser la requête de l'utilisateur
    user_query_vector = tfidf_vectorizer.transform([' '.join(user_query_tokenized)])

    # Calculer la similarité cosine entre la requête de l'utilisateur et tous les documents
    cosine_similarities = linear_kernel(user_query_vector, tfidf_matrix).flatten()

    # Pondérer la similarité cosine par l'Average Rating
    weighted_similarities = cosine_similarities * aggregated_reviews['Average Rating'].values

    # Obtenir les indices des restaurants les plus similaires (top 5)
    top_indices = weighted_similarities.argsort()[:-6:-1]

    # Afficher les informations des restaurants les plus similaires
    top_restaurants = aggregated_reviews.loc[top_indices, ['Restaurant Name', 'Information', 'Average Rating']]
    return top_restaurants

# Streamlit app
st.title("Review and Restaurant Information")

selected_option = st.selectbox("Select an option:", ["Give a review to a restaurant", "Find restaurants"])


if selected_option == "Give a review to a restaurant":
    # Option 1: Give a review to a restaurant
    st.subheader("Give a Review")

    # User input for review text and restaurant
    review_text = st.text_area("Enter your review text:")
    selected_restaurant = st.selectbox("Select a restaurant:", df['Restaurant Name'].unique())

    if st.button("Predict"):
        # Check if review text is provided
        if not review_text:
            st.warning("Please enter a review text.")
        else:
            # Make predictions for rating
            processed_text = process_text(review_text)
            rating_prediction = rating_model.predict([review_text])[0]

            # Display predictions
            st.success(f"Predicted Rating: {rating_prediction}")


elif selected_option == "Find restaurants":
    # Option 2: Find restaurants based on a query
    st.subheader("Find Restaurants")

    # User input for query
    search_query = st.text_input("Enter your search query:")

    if st.button("Search"):
        # Check if search query is provided
        if not search_query:
            st.warning("Please enter a search query.")
        else:
            # Find top restaurants based on the closest distance using NLP
            top_restaurants = question_answer(search_query)
            st.table(top_restaurants)