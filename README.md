# MLOps_Project


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