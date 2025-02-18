import json
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def init():
    global model, tokenizer
    model_path = "distilbert-finetuned"
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        inputs = tokenizer(data["text"], return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        return {"output": outputs.logits.tolist()}
    except Exception as e:
        return {"error": str(e)}
