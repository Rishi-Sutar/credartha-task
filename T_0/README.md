# Synthetic Data Generation for Transactions and Bureau Reports

## Overview
This project generates synthetic transaction and bureau report datasets for financial analysis, machine learning models, and testing environments. The datasets mimic real-world financial data while maintaining privacy and compliance.

## Features
- **Synthetic Transaction Data**: Generates transaction records for customers with categories, merchant names, payment methods, and timestamps.
- **Synthetic Bureau Report Data**: Creates financial profiles with credit scores, outstanding debts, and other metrics.
- **CSV Export**: Saves generated datasets as CSV files for easy use.
- **Customizable Parameters**: Adjust the number of customers and transactions per customer.

## Installation & Requirements
This script requires Python and the following dependencies:

```sh
pip install pandas numpy faker
```

Alternatively, you can install dependencies from the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

### Create `requirements.txt`
Save the following content in a `requirements.txt` file:

```
pandas
numpy
faker
```

## Usage
To generate the datasets, simply run:

```sh
python generate_data.py
```

## Output
The script generates two CSV files in the `data` directory:
- `transactions.csv`: Contains transaction details for multiple customers.
- `bureau_report.csv`: Includes customer financial profiles with credit scores and debt details.

## Data Fields
### Transactions Dataset
| Column            | Description |
|------------------|-------------|
| Customer ID      | Unique identifier for a customer |
| Transaction ID   | Unique transaction identifier |
| Category        | Transaction type (e.g., groceries, salary) |
| Merchant       | Name of the merchant involved in the transaction |
| Amount          | Transaction amount |
| Date           | Date of the transaction |
| Payment Method | Payment method used (e.g., credit card, UPI) |

### Bureau Report Dataset
| Column                    | Description |
|--------------------------|-------------|
| Customer ID             | Unique identifier for a customer |
| Age                     | Customer's age |
| Credit Score           | Credit score (300-900) |
| Existing Loans         | Number of existing loans |
| Utilization            | Credit utilization ratio |
| Missed Payments (12M) | Number of missed payments in the last 12 months |
| Total Outstanding Debt | Total debt outstanding |
| Debt-to-Income Ratio   | Ratio of debt to income |

## Logging
The script logs progress messages to indicate the status of dataset generation.

## License
This project is licensed under the MIT License.