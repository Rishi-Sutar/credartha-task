from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json

# Load fine-tuned model and tokenizer
MODEL_PATH = "models/distilbert_transaction_classifier"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Load label_to_category mapping
with open("data/label_to_category.json", "r") as f:
    label_to_category = json.load(f)

# FastAPI setup
app = FastAPI()

class Transaction(BaseModel):
    description: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the nlp categorization service!"}

@app.post("/predict")
def predict(transaction: Transaction):
    inputs = tokenizer(transaction.description, return_tensors="pt", padding=True, truncation=True, max_length=64)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    predicted_category = label_to_category[str(predicted_class)]
    
    return {"description": transaction.description, "predicted_category": predicted_category}

# uvicorn app.main:app --reload
