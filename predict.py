# predict.py
import joblib

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("🤖 Fake News Detector is ready!")
print("Type 'exit' to stop.\n")

while True:
    text = input("Enter a news headline: ")
    if text.lower() == "exit":
        print("👋 Exiting...")
        break

    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]

    if pred == "real":
        print("✅ Prediction: Real News\n")
    else:
        print("🚨 Prediction: Fake News\n")
