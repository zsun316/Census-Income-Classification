# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A lightGBM classfication model is used as a binary classification model for classifying salary into two classes in the Census data.

- Model type: Binary classifier
- Model date: Jun 21, 2022
- Parameters:
  - learning rate: 0.05
  - max number of leaves: 100
  - max depth: 7
  - alpha (L1 regularizaton parameter): 0.05
- Categorical features: OneHot encoding is used
- Continuous features: Standard scaler is used

## Intended Use
The model is used for classifying salary into two categories: '<=50K' and '>50K' per year.

## Training Data
- The dataset is downloaded in https://archive.ics.uci.edu/ml/datasets/census+income. 

- Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

- Prediction task is to determine whether a person makes over 50K a year.

## Evaluation Data
The whole dataset contains 32561 rows, and 20% of the data is used for testing.


## Metrics
- Precision: 0.793
- Recall: 0.671
- F-Score: 0.727

## Ethical Considerations
In this project, a classification model is developed on publicly available Census Bureau data. Unit tests are included to monitor the model performance on various slices of the data. The trained model was deployed using the FastAPI package with API tests. Both the slice-validation and the API tests are incorporated into a CI/CD framework using GitHub Actions.

## Caveats and Recommendations
Grid search was used to tune hyper parameters in the lgb model. K-fold cross validation, which is not considered in this project can be utilized to get more stable results.


