from fastapi import FastAPI
import pandas as pd
from sklearn.ensemble import IsolationForest

app = FastAPI()

model = IsolationForest(contamination=0.05)

@app.post("/detect")

def detect_anomalies(data: dict):

    df = pd.DataFrame(data)

    predictions = model.fit_predict(df[["price_ratio"]])

    anomalies = df[predictions == -1]

    return {
        "anomalies": anomalies.to_dict()
    }