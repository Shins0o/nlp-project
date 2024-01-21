import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the trained models from the notebook
rating_model = joblib.load("rating_model.pkl")
sentiment_model = joblib.load("sentiment_model.pkl")

# Load dataframe to access to restaurant names
df = pd.read_csv('restaurant_uk_reviews_v3.csv')


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

# Streamlit app
st.title("Review and Restaurant Information")

selected_option = st.selectbox("Select an option:", ["Give a review to a restaurant", "Find restaurants"])

# User input for review text
review_text = st.text_area("Enter your review text:")

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
            # Make predictions for rating and sentiment
            processed_text = process_text(review_text)
            rating_prediction = rating_model.predict([review_text])[0]
            sentiment_prediction = sentiment_model.predict([review_text])[0]

            # Display predictions
            st.success(f"Predicted Rating: {rating_prediction}")
            st.success(f"Predicted Sentiment: {'Positive' if sentiment_prediction == 1 else 'Negative'}")


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
            top_restaurants = get_top_restaurants(search_query, df)
            st.table(top_restaurants)