import os
import sys
import pandas as pd
import logging 

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataIngestionConfig:
    data_path: str=os.path.join('artifacts',"dataset.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('../data/bureau_report_risk_cleaned.csv')
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.data_path,index=False)
            logging.info("Ingestion of the data is completed")

            return self.ingestion_config.data_path,

        except Exception as e:
            raise Exception(f"Data ingestion failed due to {e}")
        
if __name__=="__main__":
    pass