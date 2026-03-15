import mlflow
from sklearn.ensemble import IsolationForest
import pandas as pd

mlflow.set_experiment("vendor_pricing")

def train():

    data = pd.read_csv("training_data.csv")

    model = IsolationForest(contamination=0.05)

    model.fit(data)

    mlflow.sklearn.log_model(model,"pricing_model")

if __name__ == "__main__":
    train()