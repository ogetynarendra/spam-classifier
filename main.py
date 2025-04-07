from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained model and vectorizer
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(msg: Message):
    X = vectorizer.transform([msg.text])
    prediction = model.predict(X)[0]
    label = "spam" if prediction == 1 else "ham"
    return {"prediction": label}
