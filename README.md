# MLOps Project for Titanic Dataset using ZenML and ML Flow

This repository contains the implementation of Titanic Survival Prediction using Scikit-learn. But the Catch here is I have written `Production Level Code`, which can be made live in your local server and see its stats using `ZenML` and `MLFlow`.

# Requirements:

- `Python 3.x`
- `Linux or Mac Environment (In windows, use WSL. In my case, I have used Linux to develop this)`
- `Decent specs to run python scripts, as the ML model is just SVM`

# Training Pipeline

This script, `training_pipeline.py`, is a ZenML pipeline that orchestrates the process of training a machine learning model. It includes the following steps:

1. **Data Ingestion**: The `run` function from `steps.ingest_data` module is used to ingest data from a given data path.

2. **Data Cleaning**: The `clean_data` function from `steps.clean_data` module is used to clean the ingested data.

3. **Data Splitting**: The `split_data` function from `steps.split_data` module is used to split the cleaned data into features (X) and target (y).

4. **Train and Test Split**: The `train_and_test_split` function from `steps.train_and_test_split` module is used to split the data into training and testing sets.

5. **Model Training**: The `train_model` function from `steps.model_train` module is used to train the model using the training data.

6. **Model Evaluation**: The `model_eval` function from `steps.model_eval` module is used to evaluate the trained model using the testing data.

## Usage

This pipeline is decorated with the `@pipeline` decorator from ZenML, and takes a single argument: `data_path`, which is the path to the data to be ingested.

```python
@pipeline
def training_pipeline(data_path: str):
    ...
```

To run this pipeline, you would typically import it in another script and call it with the path to your data:
```python
from training_pipeline import training_pipeline
training_pipeline('path/to/your/data.csv')
```

### Important Commands
`zenml up` - To turn up the server<br>
`zenml down` - To turn down the server of zenml<br>
`zenml disconnect` - to disconnect zenml server<br>
`zenml init` - To initialize the zenml folder<br>
`zenml stack describe`- To see the stack description<br>
`zenml stack list` - Lists down the stack names along with Stack ID and which stack is active

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

`zenml integration install mlflow -y`<br>
`zenml experiment-tracker register mlflow_tracker --flavor=mlflow`<br>
`zenml model-deployer register mlflow --flavor=mlflow`<br>
`zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set`<br>
`python run_deployment.py --config deploy` - To Deploy the Model<br>
`python run_deployment.py --config predict` - To Predict the results from the Model<br>
