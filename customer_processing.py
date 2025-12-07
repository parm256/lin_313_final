import pandas as pd

def load_data():
    """Load data from the Customer_Sentiment.csv file."""
    data_file = "data/Customer_Sentiment.csv"
    return pd.read_csv(data_file)

def classify_gender(text):
    """Dummy gender classifier stub."""
    # Replace with real model
    return "unknown"

def classify_age_group(text):
    """Dummy age group classifier stub."""
    # Replace with real model
    return "unknown"

def process_data(df):
    """Add sentiment, gender, and age-group predictions."""
    # sentiment already in file
    df['gender_pred'] = df['review_text'].apply(classify_gender)
    df['age_group_pred'] = df['review_text'].apply(classify_age_group)
    return df

if __name__ == "__main__":
    df = load_data()
    df = process_data(df)
    print(df.head())
