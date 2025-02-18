import pandas as pd
from datasets import Dataset
import logging
import json

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load synthetic transactions dataset
df = pd.read_csv("data/transactions.csv")

# Map categories to numerical labels
categories = list(df["Category"].unique())
category_to_label = {cat: i for i, cat in enumerate(categories)}
label_to_category = {i: cat for cat, i in category_to_label.items()}

# Save label_to_category mapping
with open("data/label_to_category.json", "w") as f:
    json.dump(label_to_category, f)

df["label"] = df["Category"].map(category_to_label)

logging.info("Categories mapped to numerical labels!")

# Ensure all columns exist and handle numerical data properly
df["description"] = df["Merchant"]  + " " + df["Payment Method"] + " " + df["Amount"].astype(str)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df[["description", "label"]], preserve_index=False)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Save dataset
dataset.save_to_disk("data/transaction_dataset")

print("Dataset saved!")
