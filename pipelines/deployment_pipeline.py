import numpy as np
import pandas as pd

# from materializer.custom_materializer import cs_materializer
from zenml import pipeline, step
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.ingest_data import run
from steps.clean_data import clean_data
from steps.split_data import split_data
from steps.train_and_test_split import train_and_test_split
from steps.model_train import train_model
from steps.model_eval import model_eval


docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger config."""

    min_accuracy: float = 0.5

@step
def deployment_trigger(accuracy:float, config:DeploymentTriggerConfig):
    """Implements a simple model deployment trigger that looks at the input model accuracy and decided if it is good enough to deploy or not."""
    
    return accuracy >= config.min_accuracy



class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True
    

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,pipeline_step_name: str,running: bool = True,model_name: str = "model",)-> MLFlowDeploymentService:
    
    pass



@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(datapath: str,
                                   min_accuracy: float = 0.5, 
                                   workers: int = 1,
                                   timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT,):
    df_artifact = run(datapath)
    df_cleaned = clean_data(df_artifact)
    X,y = split_data(df_cleaned)
    X_train, X_test, y_train, y_test = train_and_test_split(X,y)

    model = train_model(X_train, y_train)
    accuracy_score = model_eval(X_test, y_test, model)
    
    deployment_decision = deployment_trigger(accuracy_score)
    mlflow_model_deployer_step(model=model, 
                               deploy_decision=deployment_decision, 
                               workers=workers, 
                               timeout=timeout,)
    
    