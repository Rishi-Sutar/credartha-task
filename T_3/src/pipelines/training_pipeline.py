from src.components.data_ingestion import DataIngestion
from src.components.feature_transformation import FeatureTransformation
from src.components.data_spliter import TrainTestSpliter
from src.components.model_trainer import LogisticRegressionTrainer

def ml_pipeline():
    data_ingestion = DataIngestion()
    data_path = data_ingestion.initiate_data_ingestion()
    
    feature_transform = FeatureTransformation()
    transform_data = feature_transform.encode_and_scale(data_path[0])
    
    data_splitter = TrainTestSpliter()
    X_train, X_test, y_train, y_test = data_splitter.split_data(transform_data, "Risk_Classification") 
    
    trainer = LogisticRegressionTrainer()
    trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
if __name__ == "__main__":
    ml_pipeline()