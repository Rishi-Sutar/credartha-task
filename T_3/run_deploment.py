from azureml.core import Workspace, Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
import os
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the workspace
ws = Workspace.from_config()

logging.info(f"Workspace information: {ws.name}, {ws.location}, {ws.resource_group}, {ws.subscription_id}, {ws.location}")

if os.path.exists("saved_models/best_logistic_regression_model.pkl"):
    model_filename = "best_logistic_regression_model.pkl"
    model_path = f"saved_models/{model_filename}"

    # Register the model in Azure ML
    logging.info("Registering the model in Azure ML...")
    model = Model.register(workspace=ws, model_path=model_path, model_name='best_logistic_regression_model')
    logging.info(f"Model registered: {model.name} with id {model.id}")
else:
    print("Model not found. Run the training pipeline first.")
    exit(1)

# Load the registered model
model = Model(ws, name='best_logistic_regression_model')

# Create an environment
env = Environment.from_conda_specification(name='inference-env', file_path='environment.yml')

# Create an inference configuration
inference_config = InferenceConfig(entry_script='src/pipelines/deployment_pipeline.py', environment=env)

# Define the deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, location='eastus')

# Deploy the model
service = Model.deploy(workspace=ws,
                        name='risk-classification-service',
                        models=[model],
                        inference_config=inference_config,
                        deployment_config=deployment_config
                    )

service.wait_for_deployment(show_output=True)

# Print the logs
logs = service.get_logs()
print(logs)

print(f"Service state: {service.state}")
print(f"Scoring URI: {service.scoring_uri}")

service.delete()



