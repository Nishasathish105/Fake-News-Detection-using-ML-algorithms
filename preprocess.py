import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load scraped data
df = pd.read_csv("scraped_data.csv")

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"[^A-Za-z\s]", "", text)  # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.lower().strip()

df["clean_text"] = df["text"].apply(clean_text)

# Split into train/test
train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

print("✅ Cleaned data saved to train.csv and test.csv")
