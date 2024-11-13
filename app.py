import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import string
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  
    text = re.sub(r'\w*\d\w*', '', text) 
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\n', '', text) 
    text = ' '.join([word for word in text.split() if word not in stop_words])  
    return text

fake_df = pd.read_csv(r'C:\Users\Lenovo\Desktop\New folder\Fake.csv')  
real_df = pd.read_csv(r'C:\Users\Lenovo\Desktop\New folder\True.csv')  

fake_df['label'] = 1  
real_df['label'] = 0  

news_df = pd.concat([fake_df, real_df], axis=0)

news_df['text'] = news_df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(news_df['text'], news_df['label'], test_size=0.2, random_state=42)


model = Pipeline([('tfidf', TfidfVectorizer(max_df=0.7, min_df=0.1)),
                  ('nb', MultinomialNB())])

model.fit(X_train, y_train)

def predict_news(news_text):
    cleaned_text = clean_text(news_text)
    prediction = model.predict([cleaned_text])
    label = 'Real' if prediction[0] == 0 else 'Fake'
    return label

def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1E1E1E;  /* Dark background */
        }
        h1 {
            color: #FFCC00;  /* Bright yellow for visibility */
            font-family: 'Arial Black', sans-serif;
            font-size: 3em;
        }
        h2 {
            color: #FFFFFF;  /* White text for headings */
            font-family: 'Verdana', sans-serif;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;  /* Green button */
            color: white;
            font-size: 1.1em;
        }
        .stButton>button:hover {
            background-color: #45a049;  /* Darker green on hover */
        }
        .stTextInput>div>input {
            background-color: #2E2E2E;  /* Darker input field */
            color: #FFFFFF;  /* White text in input field */
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("🌟 Fake News Detection NLP Project 🌟")

    news_text = st.text_area("📰 Enter the news article text below:")

    if st.button("🔍 Predict"):
        if news_text:
            result = predict_news(news_text)
           
            st.markdown(f"<h2 style='color: #FFCC00;'>The news is predicted to be: **{result}**</h2>", unsafe_allow_html=True)
        else:
            st.error("Please enter some text to analyze.")

    st.sidebar.title("📊 Data Exploration")
    if st.sidebar.checkbox("Show Dataset"):
        st.write(news_df.head())

    if st.sidebar.checkbox("Show Pie chart"):
        labels = ['Real', 'Fake']  
        sizes = [news_df['label'].value_counts()[0], news_df['label'].value_counts()[1]]  
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal') 
        plt.title('Proportion of Real vs Fake News')
        st.pyplot(plt)  
        
if __name__ == "__main__":
    main()