from typing import List
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

LABELS = [
    'Verdante',
    'Rubresco',
    'Floralis'
]

class Features(BaseModel):
    features = List[float]

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: Features):
    prediction = model.predict([features])[0]
    prediction_label = LABELS[prediction]
    return {"prediction": LABELS[prediction]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)