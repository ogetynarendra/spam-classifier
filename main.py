from fastapi import FastAPI
import joblib

model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Spam Classifier API!"}

@app.post("/predict/")
    prediction = model.predict(message_vectorized)[0]
    result = "Spam" if prediction == 1 else "Ham"
