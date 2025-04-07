from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

app = FastAPI()

# ✅ Add this middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ogetynarendra.github.io"],  # Only your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Spam Classifier API!"}

@app.post("/predict/")
def predict_spam(message: str):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)[0]
    result = "Spam" if prediction == 1 else "Ham"
    return {"message": message, "prediction": result}
