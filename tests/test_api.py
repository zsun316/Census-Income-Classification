from fastapi.testclient import TestClient
import sys
import os

# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# from main import app
# client = TestClient(app)


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to lightGBM model deployment!"}


def test_inference_exp1(client, test_api_example):
    data = test_api_example[0].to_dict()

    r = client.post('/inference', json=data)
    assert r.status_code == 200
    assert "prediction" in r.json()
    assert r.json()["prediction"] == ["<=50K"]


def test_inference_exp2(client, test_api_example):
    data = test_api_example[1].to_dict()

    r = client.post('/inference', json=data)
    assert r.status_code == 200
    assert "prediction" in r.json()
    assert r.json()["prediction"] == [">50K"]

