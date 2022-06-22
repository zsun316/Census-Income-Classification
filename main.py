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
    age: int = Field(..., example=30)
    workclass: str = Field(..., example='State-gov')
    fnlgt:  int = Field(..., example=100000)
    education: str = Field(..., example='Bachelors')
    education_num: int = Field(..., example=13, alias='education-num')
    marital_status: str = Field(..., example='Never-married', alias='marital-status')
    occupation: str = Field(..., example='Adm-clerical')
    relationship: str = Field(..., example='Wife')
    race: str = Field(..., example='White')
    sex: str = Field(..., example='Male')
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
async def model_inference(data: Data):
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

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label='salary',
        preprocessor=preprocessor,
        lb=lb,
        training=False
    )

    pred = list(inference(model, X))
    for key, val in enumerate(pred):
        if val == 0:
            pred[key] = '<=50K'
        else:
            pred[key] = '>50K'

    return {"prediction": pred}



