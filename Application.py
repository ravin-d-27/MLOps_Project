
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment
import pandas as pd
import json
import numpy as np

st.title('Titanic Survival Prediction')



pclass = st.number_input("PClass: ")
age = st.number_input("Age: ")
sex = st.number_input("Sex: ")
sibsp = st.number_input("Sib Sp: ")
parch = st.number_input("Parch: ")
fare = st.number_input("Fare: ")
    
if st.button("Predict"):
        service = prediction_service_loader(pipeline_name="continuous_deployment_pipeline", pipeline_step_name="mlflow_model_deployer_step", running=False)
        if service is None:
            st.write("No service could be found. The pipeline will be run first to create a service.")
            
        df = pd.DataFrame(
            {
            "Pclass":[pclass],
            "Age":[age],
            "Sex":[sex],
            "SibSp":[sibsp],
            "Parch":[parch],
            "Fare":[fare],
            }
        )
        
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        
        if pred == 0:
            st.success("The Given Person did not survive :-{}".format(pred))
        else:
            st.success("The Given Person has survived :-{}".format(pred))
        
        

