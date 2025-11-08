# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("📂 Loading dataset...")
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

true['label'] = 'real'
fake['label'] = 'fake'

# Combine and shuffle data
data = pd.concat([true, fake]).sample(frac=1, random_state=42)

# Use title + text for better accuracy
data['content'] = data['title'] + " " + data['text']

X = data['content']
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🔠 Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

print("🤖 Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# Evaluate
y_pred = model.predict(X_test_vect)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n💾 Model and vectorizer saved successfully!")
