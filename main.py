from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

app = FastAPI()

class Message(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Spam Classifier API!"}

@app.post("/predict/")
def predict_spam(data: Message):
    message_vectorized = vectorizer.transform([data.message])
    prediction = model.predict(message_vectorized)[0]
    result = "Spam" if prediction == 1 else "Ham"
    return {"message": data.message, "prediction": result}
