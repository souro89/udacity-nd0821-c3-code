"""
Test the API endpoints
Date : 28-09-2026
Author : Sourodeep Banerjee
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root_returns_welcome_message():
    """Check that GET / returns a welcome message."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Salary Prediction API"
    }


def test_post_predict_returns_under_or_equal_50k():
    """Check that inference get the correct label."""
    payload = {
        "age": 20,
        "workclass": "?",
        "fnlgt": 201490,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Never-married",
        "occupation": "?",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_post_predict_returns_over_50k():
    """Check that inference get the correct label."""
    payload = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 171150,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 99999,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
