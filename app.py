import streamlit as st
import pandas as pd
from pathlib import Path

# ----------------------
# Data Loading
# ----------------------
def load_data():
    """Load data from Customer_Sentiment.csv."""
    data_file = Path("data/Customer_Sentiment.csv")
    if not data_file.exists():
        st.error("Could not find data/Customer_Sentiment.csv")
        return pd.DataFrame()
    return pd.read_csv(data_file)

# ----------------------
# ML Models (TF-IDF + Logistic Regression)
# ----------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_sentiment_model = None

def train_models(df: pd.DataFrame):
    global _sentiment_model

    if "review_text" not in df.columns or "sentiment" not in df.columns:
        st.error("CSV must contain 'review_text' and 'sentiment' columns to train model.")
        return

    # Train sentiment classifier
    _sentiment_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("logreg", LogisticRegression(max_iter=300))
    ])
    _sentiment_model.fit(df["review_text"], df["sentiment"])

# ----------------------
# Predictors
# ----------------------
def predict_sentiment(text: str):
    if _sentiment_model is None:
        return "model_not_trained"
    return _sentiment_model.predict([text])[0]

def predict_gender(text: str):
    # Placeholder for actual gender classifier
    return "unknown"

def predict_age_group(text: str):
    # Placeholder for actual age group classifier
    return "unknown"

# ----------------------
# Processing Pipeline
# ----------------------
def process_dataframe(df: pd.DataFrame):
    if "review_text" not in df.columns:
        st.error("CSV must contain a 'review_text' column.")
        return df

    if "sentiment" not in df.columns:
        df["sentiment"] = df["review_text"].apply(predict_sentiment)

    df["gender_pred"] = df["review_text"].apply(predict_gender)
    df["age_group_pred"] = df["review_text"].apply(predict_age_group)
    return df

# ----------------------
# Streamlit App
# ----------------------
st.set_page_config(page_title="Customer Sentiment Classifier", layout="wide")
st.title("📊 Customer Sentiment, Gender & Age-Group Classifier")

st.write("This app loads the **Customer_Sentiment.csv** file, predicts sentiment, gender, and age group, and displays the results.")

# Load + process
df = load_data()
if df.empty:
    st.stop()

processed = process_dataframe(df.copy())

st.subheader("Processed Data")
st.dataframe(processed, use_container_width=True)

# Text box for manual classification
st.subheader("Try Your Own Review Text")
user_input = st.text_area("Enter review text:")
if user_input:
    st.write("**Sentiment:**", predict_sentiment(user_input))
    st.write("**Predicted Gender:**", predict_gender(user_input))
    st.write("**Predicted Age Group:**", predict_age_group(user_input))
