import logging
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LogisticRegressionTrainer:
    def __init__(self, experiment_name="Credit Risk Assessment"):
        # Initialize the base model and hyperparameter grid
        self.model = LogisticRegression(solver='liblinear', random_state=42)
        self.param_grid = {
            "penalty": ['l1', 'l2'],
            "C": [0.01, 0.1, 1, 10, 100],
            "max_iter": [500, 1000, 2000]
        }
        self.best_model = None
        self.best_params = None
        mlflow.set_experiment(experiment_name)
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, cv=3, n_iter=10):
        """Train Logistic Regression using RandomizedSearchCV, then refit with best parameters."""
        try:
            with mlflow.start_run(run_name="Logistic Regression (RandomizedSearch)") as run:
                logging.info("Starting hyperparameter tuning with RandomizedSearchCV...")
                
                # Hyperparameter tuning
                random_search = RandomizedSearchCV(
                    self.model,
                    self.param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    random_state=42
                )
                random_search.fit(X_train, y_train)
                
                # Retrieve best hyperparameters
                self.best_params = random_search.best_params_
                logging.info(f"Best hyperparameters found: {self.best_params}")
                
                # Refit the model manually with the best parameters on the full training set
                self.best_model = LogisticRegression(
                    penalty=self.best_params["penalty"],
                    C=self.best_params["C"],
                    max_iter=self.best_params["max_iter"],
                    solver='liblinear',
                    random_state=42
                )
                self.best_model.fit(X_train, y_train)
                
                # Evaluate the refitted model
                y_pred = self.best_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                logging.info(f"Refitted Model - Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
                
                # Log parameters, metrics, and the model with MLflow
                mlflow.log_params(self.best_params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                mlflow.sklearn.log_model(self.best_model, "best_logistic_regression_model")
                
                # Save the best model locally
                os.makedirs("saved_models", exist_ok=True)
                joblib.dump(self.best_model, "saved_models/best_logistic_regression_model.pkl")
                logging.info("Best Logistic Regression model saved as 'best_logistic_regression_model.pkl'")
                
        
        except Exception as e:
            logging.error(f"Model training failed due to: {e}")
            raise

if __name__ == "__main__":
    # Example usage:
    # Assume you have already loaded your data and performed a train-test split.
    # For example:
    # from sklearn.model_selection import train_test_split
    # import pandas as pd
    #
    # df = pd.read_csv("your_dataset.csv")
    # X = df.drop(columns=['Target'])
    # y = df['Target']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # trainer = LogisticRegressionTrainer()
    # best_model, best_params = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    pass
