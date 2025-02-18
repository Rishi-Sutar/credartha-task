import logging
import numpy as np
import pandas as pd
import os
import joblib

from sklearn.preprocessing import StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureTransformationConfig:
    scaled_data_path: str = os.path.join('artifacts', "transform_data.csv")
    scaler_path: str = os.path.join('artifacts', "standard_scaler.pkl")

class FeatureTransformation(FeatureTransformationConfig):
    def __init__(self):
        self.feature_scaling_config = FeatureTransformationConfig()
        
    def map_risk_classification(self, data):
        logging.info("Entered the map risk classification method or component")
        try:
            logging.info(f"Data head before mapping: \n{data.head()}")
            mapping = {'Low Risk': 0, 'High Risk': 1}
            data['Risk_Classification'] = data['Risk_Classification'].map(mapping)
    
            logging.info("Mapping of the Risk Classification column is completed")
            logging.info(f"Data head after mapping: \n{data.head()}")
            return data
        
        except Exception as e:
            raise Exception(f"Mapping failed due to {e}")
        
    def standard_scaling(self, data):
        logging.info("Entered the standard scaling method or component")
        try:
            logging.info(f"Data head before scaling: \n{data.head()}")
            # Exclude 'Risk Classification' from scaling
            features = data.drop(columns=['Risk_Classification'])
            risk_classification = data['Risk_Classification']
            
            standard_scaler = StandardScaler()
            scaled_features = standard_scaler.fit_transform(features)
            logging.info("Standard scaling of the data is completed")
            
            # Save the scaler
            joblib.dump(standard_scaler, self.feature_scaling_config.scaler_path)
            logging.info(f"Standard scaler saved at {self.feature_scaling_config.scaler_path}")
            
            scaled_data_df = pd.DataFrame(scaled_features, columns=features.columns)
            scaled_data_df['Risk_Classification'] = risk_classification
            logging.info(f"Data head after scaling: \n{scaled_data_df.head()}")
            return scaled_data_df
        
        except Exception as e:
            raise Exception(f"Standard scaling failed due to {e}")

    def encode_and_scale(self, file_path):
        logging.info("Entered the encode and scale method or component")
        try:
            logging.info(f"Reading data from {file_path}")
            data = pd.read_csv(file_path)
            logging.info(f"Data head: \n{data.head()}")
            
            # Perform mapping for 'Risk Classification'
            data = self.map_risk_classification(data)
            
            # Perform standard scaling
            scaled_data = self.standard_scaling(data)
            
            # Save the transformed data to a new file
            transformed_file_path = os.path.join('artifacts', "encoded_scaled_data.csv")
            scaled_data.to_csv(transformed_file_path, index=False)
            logging.info(f"Encoded and scaled data saved to {transformed_file_path}")
            
            return scaled_data
        
        except Exception as e:
            raise Exception(f"Encode and scale failed due to {e}")

if __name__ == "__main__":
    pass