import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from huggingface_hub import HfApi

import os
from dotenv import load_dotenv

load_dotenv()

# Load dataset
dataset = load_from_disk("data/transaction_dataset")

# Load tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["description"], padding="max_length", truncation=True, max_length=10)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model
num_labels = len(set(dataset["train"]["label"]))
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# Compute evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="models/distilbert_transaction_classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    push_to_hub=True,  # Enable pushing to hub
    hub_model_id="richie28/distilbert_transaction_classifier",  # Replace with your model ID
    hub_token= os.environ['HF_TOKEN']  # Replace with your Hugging Face token
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Save trained model
trainer.save_model("models/distilbert_transaction_classifier")
tokenizer.save_pretrained("models/distilbert_transaction_classifier")
print("Fine-tuned DistilBERT model saved!")

# Push model and tokenizer to Hugging Face Hub
trainer.push_to_hub()
