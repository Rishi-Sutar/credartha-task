import pickle
import json
import numpy as np
import os

import logging

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_logistic_regression_model.pkl')
    logging.info(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

def run(data):
    try:
        data = json.loads(data)
        data = np.array(data['data'])
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})