
from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Load the workspace
ws = Workspace.from_config()

# Register the model
model = Model.register(
    workspace=ws,
    model_name="distilbert_transaction_classifier",
    model_path="models/distilbert_transaction_classifier"
)

# Define the environment
env = Environment(name="distilbert-env")
env.python.conda_dependencies.add_pip_package("torch")
env.python.conda_dependencies.add_pip_package("transformers")
env.python.conda_dependencies.add_pip_package("azureml-defaults")

# Define inference configuration
inference_config = InferenceConfig(
    entry_script="score.py",
    environment=env
)

# Deploy the model as a web service
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=2)
service = Model.deploy(
    workspace=ws,
    name="distilbert-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)
print(f"Service deployed at {service.scoring_uri}")
