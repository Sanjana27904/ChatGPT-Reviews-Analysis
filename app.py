import nltk
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.corpus import stopwords
import string
import nltk

# Ensure stopwords are downloaded
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

# Step 2: Preprocess with sentiment analysis and caching
@st.cache_data
def preprocess_sentiment(df):
    df['Sentiment'] = df['Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['Sentiment Category'] = df['Sentiment'].apply(lambda polarity: (
        "Positive" if polarity > 0.2 else ("Negative" if polarity < -0.2 else "Neutral")
    ))
    df['Month'] = df['Review Date'].dt.to_period('M').dt.to_timestamp()
    df['Review Length'] = df['Review'].str.len()
    return df

# Word cloud generation
def generate_wordcloud(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return WordCloud(width=800, height=400, background_color='black').generate(text)

# Load and preprocess data
filepath = r"C:\Users\Sanjana\chatgpt_review\ChatGPT_Reviews.csv"
df = load_data(filepath)
df = preprocess_sentiment(df)

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an Analysis:", 
    ["Dashboard Overview", "Ratings Distribution", "Sentiment Analysis", "Word Cloud", "Review Trends", "Sentiment vs Ratings", "Review Length Analysis"]
)

st.title("ChatGPT Reviews Analysis")

# Dashboard View
if option == "Dashboard Overview":
    st.header("📊 Dashboard Overview: All Graphs")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ratings Distribution")
        fig1 = px.histogram(df, x='Ratings', nbins=5, color_discrete_sequence=['skyblue'])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['Sentiment Category'].value_counts()
        fig2 = px.pie(values=sentiment_counts.values, names=sentiment_counts.index)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Review Trends Over Time")
        review_counts = df.groupby('Month').size()
        fig3 = px.bar(x=review_counts.index.astype(str), y=review_counts.values, labels={'x': 'Month', 'y': 'Number of Reviews'})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Ratings vs Sentiment Heatmap")
        heatmap_data = df.pivot_table(index='Ratings', columns='Sentiment Category', aggfunc='size', fill_value=0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='coolwarm')
        st.pyplot(plt, clear_figure=True)

    st.subheader("Word Clouds for Sentiments")
    col5, col6 = st.columns(2)

    with col5:
        st.write("Positive Reviews")
        positive_text = " ".join(df[df['Sentiment Category'] == "Positive"]['Review'].astype(str))
        wordcloud_pos = generate_wordcloud(positive_text)
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt, clear_figure=True)

    with col6:
        st.write("Negative Reviews")
        negative_text = " ".join(df[df['Sentiment Category'] == "Negative"]['Review'].astype(str))
        wordcloud_neg = generate_wordcloud(negative_text)
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt, clear_figure=True)

    st.subheader("Review Length Analysis")
    fig5, ax = plt.subplots()
    sns.histplot(df['Review Length'], bins=30, kde=True, ax=ax)
    st.pyplot(fig5)

# Individual: Ratings Distribution
elif option == "Ratings Distribution":
    st.header("📊 Ratings Distribution")
    chart_type = st.selectbox("Select Chart Type", ["Bar", "Pie", "Line"])

    ratings_counts = df['Ratings'].value_counts().sort_index()

    if chart_type == "Bar":
        fig = px.bar(x=ratings_counts.index, y=ratings_counts.values, labels={'x': 'Rating', 'y': 'Count'}, color_discrete_sequence=['skyblue'])
    elif chart_type == "Pie":
        fig = px.pie(values=ratings_counts.values, names=ratings_counts.index)
    elif chart_type == "Line":
        fig = px.line(x=ratings_counts.index, y=ratings_counts.values, labels={'x': 'Rating', 'y': 'Count'})

    st.plotly_chart(fig)

# Individual: Sentiment Analysis
elif option == "Sentiment Analysis":
    st.header("😊 Sentiment Analysis")
    chart_type = st.selectbox("Select Chart Type", ["Pie", "Bar"])

    sentiment_counts = df['Sentiment Category'].value_counts()

    if chart_type == "Pie":
        fig2 = px.pie(values=sentiment_counts.values, names=sentiment_counts.index)
    else:
        fig2 = px.bar(x=sentiment_counts.index, y=sentiment_counts.values, labels={'x': 'Sentiment', 'y': 'Count'}, color_discrete_sequence=['orange'])

    st.plotly_chart(fig2)

# Individual: Word Cloud
elif option == "Word Cloud":
    st.header("🌍 Most Commonly Used Words in Reviews")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Positive Reviews")
        positive_text = " ".join(df[df['Sentiment Category'] == "Positive"]['Review'].astype(str))
        wordcloud_pos = generate_wordcloud(positive_text)
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt, clear_figure=True)

    with col2:
        st.subheader("Negative Reviews")
        negative_text = " ".join(df[df['Sentiment Category'] == "Negative"]['Review'].astype(str))
        wordcloud_neg = generate_wordcloud(negative_text)
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt, clear_figure=True)

# Individual: Review Trends
elif option == "Review Trends":
    st.header("📅 Review Trends Over Time")
    chart_type = st.selectbox("Select Chart Type", ["Bar", "Line"])

    review_counts = df.groupby('Month').size()

    if chart_type == "Bar":
        fig3 = px.bar(x=review_counts.index.astype(str), y=review_counts.values, labels={'x': 'Month', 'y': 'Review Count'})
    else:
        fig3 = px.line(x=review_counts.index.astype(str), y=review_counts.values, labels={'x': 'Month', 'y': 'Review Count'})

    st.plotly_chart(fig3)

# Individual: Sentiment vs Ratings Heatmap
elif option == "Sentiment vs Ratings":
    st.header("🔥 Ratings vs Sentiment Categories")
    heatmap_data = df.pivot_table(index='Ratings', columns='Sentiment Category', aggfunc='size', fill_value=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='coolwarm')
    st.pyplot(plt, clear_figure=True)

# Individual: Review Length
elif option == "Review Length Analysis":
    st.header("📏 Review Length vs Sentiment Category")
    df['Length Category'] = df['Review'].apply(lambda x: 'Short' if len(str(x)) < 50 else ('Medium' if len(str(x)) < 150 else 'Long'))
    length_sentiment = pd.crosstab(df['Length Category'], df['Sentiment Category'])
    fig4, ax = plt.subplots(figsize=(10, 6))
    length_sentiment.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
    plt.title("Review Length vs Sentiment Category")
    plt.xlabel("Length Category")
    plt.ylabel("Number of Reviews")
    plt.legend(title="Sentiment Category")
    st.pyplot(fig4)

st.title("🔎 Conclusion")
st.write("This dashboard provides an interactive analysis of ChatGPT reviews, sentiment trends, and user feedback insights.")

