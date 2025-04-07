from fastapi import FastAPI
import joblib

# Load trained model and vectorizer
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Spam Classifier API!"}

@app.post("/predict/")
def predict_spam(message: str):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)[0]
    result = "Spam" if prediction == 1 else "Ham"
    return {"message": message, "prediction": result}
