from zenml import pipeline
from steps.ingest_data import run
import pandas as pd
from zenml.artifact_stores.local_artifact_store import LocalArtifactStore

@pipeline
def training_pipeline(data_path: str):
    """
    Training pipeline.
    """
    df_artifact = run(data_path)
