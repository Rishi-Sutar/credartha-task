# NLP-Based Categorization

This project is an NLP-based text categorization system. It includes scripts for preprocessing data, training a bert model, and serving the trained model via a FastAPI application.

## Description

This project is designed to categorize text data using Natural Language Processing (NLP) techniques. It consists of the following components:

1. **Preprocessing**: The `preprocess.py` script cleans and prepares the text data for training.
2. **Model Training**: The `train_model.py` script trains a machine learning model using the preprocessed data and saves the trained model.
3. **FastAPI App**: The `main.py` script serves the trained model via a FastAPI application, allowing users to make predictions through API endpoints.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nlp-categorization.git
   cd nlp-categorization

   ```

2. Install the required dependencies:

    ```bash
    Copy
    pip install -r requirements.txt
    ```
#### Preprocessing and Training
3.  Navigate to the script folder:

    ```bash
    cd script
    Run the preprocessing script:
    ````

    ```bash
    python preprocess.py
    Train the model:
    ```

    ````bash
    python train_model.py
    ````

    It will fine tuned distiled-bert model and save it in `model` folder

4. Run app/main.py to use model using FASTAPI

    ```bash
    uvicorn app.main:app --reload
    ```

## Logging
The script logs progress messages to indicate the status of dataset generation.