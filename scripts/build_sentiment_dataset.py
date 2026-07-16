"""
build_sentiment_dataset.py
--------------------------
Recreates the cleaned Sentiment140 dataset with columns: sentence, sentiment (0/1).

SETUP (one time):
    1. Create a free Kaggle account at https://www.kaggle.com if you don't have one.
    2. Install dependencies: pip install kagglehub pandas scikit-learn
    3. Get the raw Sentiment140 CSV via
       kagglehub.dataset_download("kazanova/sentiment140") and place it at
       ml/data/raw/sentiment.csv
    4. Run: python -m scripts.build_sentiment_dataset

The first time you use kagglehub it will prompt for your Kaggle API token
(Kaggle Account -> Settings -> Create New Token, which downloads kaggle.json).
"""

import pandas as pd
from scripts.data_preprocessing import clean_text
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# STEP 1: Load the raw Sentiment140 CSV
# ---------------------------------------------------------------------------
cols = ["sentiment", "id", "date", "query", "user", "sentence"]
df = pd.read_csv("ml/data/raw/sentiment.csv", encoding="latin-1", header=None, names=cols)

# ---------------------------------------------------------------------------
# STEP 2: Keep only the columns needed
# ---------------------------------------------------------------------------
df = df[["sentence", "sentiment"]]

# ---------------------------------------------------------------------------
# STEP 3: Remap labels (raw file uses 0=neg, 4=pos; remap 4 -> 1)
# ---------------------------------------------------------------------------
df["sentiment"] = df["sentiment"].replace(4, 1)

# ---------------------------------------------------------------------------
# STEP 4: Clean the text
# ---------------------------------------------------------------------------
df["sentence"] = df["sentence"].apply(clean_text)


sentiment = df

sentiment_train, sentiment_test = train_test_split(sentiment, test_size=0.2, random_state=42)

sentiment_test, sentiment_val = train_test_split(sentiment_test, test_size=0.5, random_state=42)

sentiment_train.to_csv("ml/data/processed/sentiment_train.csv", index=False)
sentiment_test.to_csv("ml/data/processed/sentiment_test.csv", index=False)
sentiment_val.to_csv("ml/data/processed/sentiment_val.csv", index=False)

print(df.head())
