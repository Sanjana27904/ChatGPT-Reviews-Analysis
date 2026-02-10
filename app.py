import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Step 1: Load data with caching
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Review Date'] = pd.to_datetime(df['Review Date'], errors='coerce')
    df['Ratings'] = pd.to_numeric(df['Ratings'], errors='coerce')
    df.dropna(inplace=True)
    return df

# Step 2: Preprocess with sentiment analysis
@st.cache_data
def preprocess_sentiment(df):
    df['Sentiment'] = df['Review'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    df['Sentiment Category'] = df['Sentiment'].apply(
        lambda polarity: "Positive" if polarity > 0.2 else
        ("Negative" if polarity < -0.2 else "Neutral")
    )
    df['Month'] = df['Review Date'].dt.to_period('M').dt.to_timestamp()
    df['Review Length'] = df['Review'].astype(str).str.len()
    return df

# Word cloud generation
def generate_wordcloud(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return WordCloud(
        width=800,
        height=400,
        background_color='black'
    ).generate(text)

# ✅ FIXED FILE PATH (RELATIVE PATH)
FILE_PATH = "ChatGPT_Reviews.csv"

# Load and preprocess data
df = load_data(FILE_PATH)
df = preprocess_sentiment(df)

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose an Analysis:",
    [
        "Dashboard Overview",
        "Ratings Distribution",
        "Sentiment Analysis",
        "Word Cloud",
        "Review Trends",
        "Sentiment vs Ratings",
        "Review Length Analysis"
    ]
)

st.title("ChatGPT Reviews Analysis")

# Dashboard View
if option == "Dashboard Overview":
    st.header("📊 Dashboard Overview")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x='Ratings', nbins=5)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        sentiment_counts = df['Sentiment Category'].value_counts()
        fig2 = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        review_counts = df.groupby('Month').size()
        fig3 = px.bar(
            x=review_counts.index.astype(str),
            y=review_counts.values,
            labels={'x': 'Month', 'y': 'Number of Reviews'}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        heatmap_data = df.pivot_table(
            index='Ratings',
            columns='Sentiment Category',
            aggfunc='size',
            fill_value=0
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='d')
        st.pyplot(plt, clear_figure=True)

elif option == "Ratings Distribution":
    ratings_counts = df['Ratings'].value_counts().sort_index()
    fig = px.bar(
        x=ratings_counts.index,
        y=ratings_counts.values,
        labels={'x': 'Rating', 'y': 'Count'}
    )
    st.plotly_chart(fig)

elif option == "Sentiment Analysis":
    sentiment_counts = df['Sentiment Category'].value_counts()
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={'x': 'Sentiment', 'y': 'Count'}
    )
    st.plotly_chart(fig)

elif option == "Word Cloud":
    col1, col2 = st.columns(2)

    with col1:
        positive_text = " ".join(
            df[df['Sentiment Category'] == "Positive"]['Review'].astype(str)
        )
        st.pyplot(generate_wordcloud(positive_text).to_array())

    with col2:
        negative_text = " ".join(
            df[df['Sentiment Category'] == "Negative"]['Review'].astype(str)
        )
        st.pyplot(generate_wordcloud(negative_text).to_array())

elif option == "Review Trends":
    review_counts = df.groupby('Month').size()
    fig = px.line(
        x=review_counts.index.astype(str),
        y=review_counts.values,
        labels={'x': 'Month', 'y': 'Review Count'}
    )
    st.plotly_chart(fig)

elif option == "Sentiment vs Ratings":
    heatmap_data = df.pivot_table(
        index='Ratings',
        columns='Sentiment Category',
        aggfunc='size',
        fill_value=0
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='d')
    st.pyplot(plt, clear_figure=True)

elif option == "Review Length Analysis":
    df['Length Category'] = df['Review Length'].apply(
        lambda x: 'Short' if x < 50 else
        ('Medium' if x < 150 else 'Long')
    )
    length_sentiment = pd.crosstab(
        df['Length Category'],
        df['Sentiment Category']
    )
    length_sentiment.plot(kind='bar', stacked=True)
    st.pyplot(plt)

st.subheader("🔎 Conclusion")
st.write(
    "This dashboard provides an interactive analysis of ChatGPT reviews, "
    "sentiment trends, and user feedback insights."
)


