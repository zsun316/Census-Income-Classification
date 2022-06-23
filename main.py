# Put the code for your API here.
import os
import numpy as np
import pandas as pd

from fastapi import FastAPI
from typing import Union, Optional
from pydantic import BaseModel, Field
from ml_pipeline.ml.data import process_data
from ml_pipeline.ml.model import load_model, inference


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class Data(BaseModel):
    age: Optional[Union[int, list]] = [39, 52]
    workclass: Optional[Union[str, list]] = ['State-gov', 'Self-emp-inc']
    fnlgt: Optional[Union[int, list]] = [77516, 287927]
    education: Optional[Union[str, list]] = ['Bachelors', 'HS-grad']
    education_num: Optional[Union[int, list]] = Field([13, 9], alias='education-num')
    marital_status: Optional[Union[str, list]] = Field(['Never-married', 'Married-civ-spouse'], alias='marital-status')
    occupation: Optional[Union[str, list]] = ['Adm-clerical', 'Exec-managerial']
    relationship: Optional[Union[str, list]] = ['Not-in-family', 'Wife']
    race: Optional[Union[str, list]] = ['White', 'White']
    sex: Optional[Union[str, list]] = ['Male', 'Female']
    capital_gain: Optional[Union[int, list]] = Field([2174, 15024], alias='capital-gain')
    capital_loss: Optional[Union[int, list]] = Field([0, 0], alias='capital-loss')
    hours_per_week: Optional[Union[int, list]] = Field([40, 40], alias='hours-per-week')
    native_country: Optional[Union[str, list]] = Field(['United-States', 'United-States'],
                                                       alias='native-country')

root_path = os.path.dirname(os.path.abspath(__file__))
model = load_model(root_path, 'model.pkl')
preprocessor = load_model(root_path, 'preprocessor.pkl')
lb = load_model(root_path, 'lb.pkl')

app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Welcome to lightGBM model deployment!"}


@app.post("/model_inference")
async def predict(data: Data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    inputData = data.dict(by_alias=True)
    for key, val in inputData.items():
        if not isinstance(val, list):
            inputData[key] = [val]

    df = pd.DataFrame(inputData)

    X, y, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        preprocessor=preprocessor,
        lb=lb
    )

    pred = list(inference(model, X))
    for key, val in enumerate(pred):
        if val == 0:
            pred[key] = '<=50K'
        else:
            pred[key] = '>50K'

    return {"prediction": pred}



