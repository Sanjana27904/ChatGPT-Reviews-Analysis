# ChatGPT Reviews Analysis 🤖📊

## 🚀 Live Application
🔗 https://chatgpt-reviews-analysis-fihttddkmyaznd9jsa9swe.streamlit.app/

---

## 📌 Project Overview
This project focuses on analyzing **ChatGPT user reviews** to extract meaningful insights related to **user sentiment, satisfaction levels, and recurring pain points** using **Data Analytics and Natural Language Processing (NLP)** techniques.

The dataset contains **textual reviews, ratings, and timestamps**, enabling sentiment analysis, trend identification, and interactive visual exploration.  
The final output is an **interactive Streamlit dashboard** that converts unstructured feedback into **actionable insights**.

---

## 🎯 Objectives
- Analyze user sentiment toward ChatGPT  
- Identify common positive feedback and critical issues  
- Study the relationship between ratings and sentiment  
- Visualize trends in user reviews over time  
- Build an interactive analytics dashboard  

---

## 🧠 Problem Statement
User reviews provide valuable insights into the strengths and limitations of AI systems.  
However, raw textual feedback is unstructured and difficult to interpret at scale.

This project aims to **process, analyze, and visualize large volumes of user feedback** to support **data-driven improvements** in conversational AI systems.

---

## 🗂 Dataset Description
- **Type:** ChatGPT User Reviews  
- **Attributes:**
  - Review Text  
  - User Rating  
  - Review Date / Timestamp  
- **Nature:** Unstructured and semi-structured data  

---

## 🔍 Analysis Workflow

### 1️⃣ Data Loading & Cleaning
- Loaded dataset using **Pandas**
- Converted review dates to `datetime` format
- Converted ratings to numeric values
- Removed null, duplicate, and inconsistent records

---

### 2️⃣ Data Preprocessing
- Text normalization (lowercasing, formatting)
- Stopword handling using **NLTK**
- Review length calculation
- Monthly aggregation of reviews

---

### 3️⃣ Sentiment Analysis
- Applied **TextBlob** polarity scoring
- Sentiment classification:
  - **Positive:** Polarity > 0.2  
  - **Neutral:** Polarity between -0.2 and 0.2  
  - **Negative:** Polarity < -0.2  

---

### 4️⃣ Exploratory Data Analysis (EDA)
- Ratings distribution analysis
- Sentiment distribution analysis
- Review trends over time
- Review length vs sentiment study

---

### 5️⃣ Visualization & Dashboard
- Interactive charts using **Plotly**
- Heatmap for **Sentiment vs Ratings**
- Word clouds for positive and negative reviews
- Fully interactive **Streamlit dashboard**

---

## 📊 Key Insights
- ✅ Majority of users express **positive sentiment**
- ⚠️ Negative reviews highlight:
  - Response inconsistency  
  - Contextual understanding gaps  
  - Difficulty with complex queries  
- 📈 Strong correlation between **higher ratings and positive sentiment**
- 🕒 Review volume and sentiment patterns change over time
- ✍️ Longer reviews often contain detailed criticism or mixed sentiment

---

## 🖥 Dashboard Features
- 📊 Ratings Distribution (Bar / Pie / Line charts)
- 😊 Sentiment Analysis (Positive / Neutral / Negative)
- 📅 Review Trends Over Time
- 🔥 Sentiment vs Ratings Heatmap
- ☁️ Word Clouds for User Feedback
- 📏 Review Length vs Sentiment Analysis

---

## 🛠 Tools & Technologies
- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **NLP:** NLTK, TextBlob  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Dashboard Framework:** Streamlit  

---

## 🚀 How to Run the Project

### 🔹 Step 1: Clone the Repository
git clone https://github.com/your-username/ChatGPT-Reviews-Analysis.git
cd ChatGPT-Reviews-Analysis


### 🔹 Step 2: Install Required Dependencies
pip install streamlit numpy pandas plotly matplotlib seaborn wordcloud textblob nltk

### 🔹Step 3: Download NLP Resources
python -m textblob.download_corpora

### 🔹Step 4: Run the Streamlit Application 🚀
streamlit run app.py

### 🔹Step 5: Open in Browser
http://localhost:8501

---
### 6. Tools & Technologies

Language: Python

Data Processing: Pandas, NumPy

NLP: NLTK, TextBlob

Visualization: Matplotlib, Seaborn, Plotly

Dashboard: Streamlit

---

## 7📌Applications & Use Cases

Analyze user satisfaction for AI-based applications

Identify improvement areas in conversational AI systems

Monitor sentiment trends over time

Serve as a reference project for NLP + Data Analytics dashboards

---

## 8.🔮Future Scopee

Integrate advanced NLP models (BERT, Transformers) for improved sentiment accuracy

Enable real-time review ingestion

Add multilingual sentiment analysis

Predict future sentiment trends using time-series models

---

## ⭐ Conclusion

This project demonstrates how data analytics and NLP techniques can be effectively applied to real-world user feedback to extract insights, visualize trends, and support data-driven decision-making for AI systems like ChatGPT.
