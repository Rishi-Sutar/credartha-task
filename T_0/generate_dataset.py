import pandas as pd
import numpy as np
from faker import Faker
import logging
import random
import os
from typing import List

# Initialize Faker and random seed for reproducibility
fake = Faker()
fake.seed_instance(43)
np.random.seed(43)

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create 'data' folder if it does not exist
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Transaction categories and merchants dictionary
CATEGORIES = [
    "salary", "cash Withdrawals", "transfers", "groceries", "utilities", "transport",
    "shopping", "entertainment", "food and dining", "health", "insurance",
    "loan EMI Payment", "others"
]

MERCHANT_DICT = {
    "groceries": ["bigbasket", "d-mart", "reliance fresh", "more", "grofers", "local store"],
    "utilities": ["electricity", "water", "broadband", "telephone", "gas"],
    "transport": ["ola", "uber", "auto", "metro", "bus", "train", "flight"],
    "shopping": ["amazon", "flipkart", "myntra", "jabong", "snapdeal", "club factory", "aliexpress"],
    "entertainment": ["bookmyshow", "netflix", "prime", "hotstar", "zee5", "sonyliv", "spotify", "gaana", "youtube", "tiktok"],
    "food and dining": ["restaurants", "cafes", "zomato", "swiggy", "starbucks", "fast food"],
    "health": ["pharmacy", "hospital", "clinic", "gym", "yoga", "spa"],
    "insurance": ["life insurance", "health insurance", "vehicle insurance", "travel insurance"],
    "loan EMI Payment": ["home loan", "personal loan", "car loan", "education loan", "gold loan", "two wheeler loan", "consumer durable loan"],
    "others": ["others"]
}

def generate_transactions(num_customers: int, transactions_per_customer: int) -> pd.DataFrame:
    """
    Generate synthetic transaction data for customers.
    """
    transactions = []
    
    for customer_id in range(1, num_customers + 1):
        num_transactions = random.randint(50, transactions_per_customer)
        customer_id = f'cust_{customer_id}'
        
        for i in range(num_transactions):
            transaction_id = f't_{i+1}'
            category = random.choice(CATEGORIES)
            merchant_name = random.choice(MERCHANT_DICT.get(category, ["miscellaneous"]))
            
            amount = round(random.uniform(50, 5000), 2) if category != "salary" else round(random.uniform(30000, 100000), 2)
            
            date_format = random.choice(["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"])
            date = fake.date_between(start_date='-3y', end_date='today').strftime(date_format)
            payment_method = random.choice(["credit card", "debit card", "upi", "cash"])
            
            transactions.append([customer_id, transaction_id, category, merchant_name, amount, date, payment_method])
    
    columns = ["Customer ID", "Transaction ID", "Category", "Merchant", "Amount", "Date", "Payment Method"]
    return pd.DataFrame(transactions, columns=columns)

def generate_bureau_report(num_customers: int) -> pd.DataFrame:
    """
    Generate synthetic customer bureau report data.
    """
    bureau_data = []
    
    for customer_id in range(1, num_customers + 1):
        age = random.randint(21, 60)
        existing_loans = random.randint(0, 5)
        utilization = round(random.uniform(0, 1), 2)
        missed_payments = random.randint(0, 6) if existing_loans > 0 else 0
        total_outstanding_debt = round(random.uniform(1000, 500000), 2)
        debt_to_income_ratio = round(random.uniform(0.1, 0.8), 2)
        
        credit_score = 900
        if existing_loans > 2:
            credit_score -= 50
        if utilization > 0.5:
            credit_score -= 100
        if missed_payments > 0:
            credit_score -= missed_payments * 30
        if total_outstanding_debt > 250000:
            credit_score -= 50
        if debt_to_income_ratio > 0.5:
            credit_score -= 50
        
        credit_score = max(300, min(credit_score, 900))
        
        bureau_data.append([customer_id, age, credit_score, existing_loans, utilization, missed_payments, total_outstanding_debt, debt_to_income_ratio])
    
    columns = ["Customer ID", "Age", "Credit Score", "Existing Loans", "Utilization", "Missed Payments (12M)", "Total Outstanding Debt", "Debt-to-Income Ratio"]
    return pd.DataFrame(bureau_data, columns=columns)

def save_dataframe_to_csv(df: pd.DataFrame, filename: str):
    """
    Save a pandas DataFrame as a CSV file.
    """
    file_path = os.path.join(DATA_FOLDER, filename)
    df.to_csv(file_path, index=False)
    logging.info(f"Dataset saved at: {file_path}")

def main():
    """
    Main function to generate datasets and save them as CSV files.
    """
    num_customers = 100
    transactions_per_customer = 100

    logging.info("Generating synthetic transaction data...")
    transaction_df = generate_transactions(num_customers, transactions_per_customer)
    save_dataframe_to_csv(transaction_df, "transactions.csv")

    logging.info("Generating synthetic bureau report data...")
    bureau_df = generate_bureau_report(num_customers)
    save_dataframe_to_csv(bureau_df, "bureau_report.csv")

    logging.info("Data generation process completed successfully!")

if __name__ == "__main__":
    main()
