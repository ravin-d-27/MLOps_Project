FROM python:3.8

# Install system dependencies
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /opt/app

# Copy and install Python dependencies
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r requirements.txt

# Copy the application files
COPY . /opt/app

# Specify the command to run on container start
RUN zenml init \
    && zenml integration install mlflow -y \
    && zenml experiment-tracker register mlflow_tracker --flavor=mlflow \
    && zenml model-deployer register mlflow --flavor=mlflow \
    && zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set \
    && python3 run_pipeline.py \
    && python3 run_deployment.py --config deploy \
    && python3 run_deployment.py --config predict \
    && streamlit run Application.py
