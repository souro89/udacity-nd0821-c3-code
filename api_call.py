"""
Code for testing the GET and POSt Request from the deployed endpoint
Date : 29-09-2026
Author : Sourodeep Banerjee
"""

import requests


# GET Request
get_response = requests.get("https://udacity-nd0821-c3-code.onrender.com/")

# print("GET Status :", get_response.status_code)
# print("GET JSON :", get_response.json())

# POST Request
post_response = requests.post(
    "https://udacity-nd0821-c3-code.onrender.com/predict",
    json={
        "age": 37,
        "capital-gain": 99999,
        "capital-loss": 0,
        "education": "Bachelors",
        "education-num": 13,
        "fnlgt": 171150,
        "hours-per-week": 60,
        "marital-status": "Married-civ-spouse",
        "native-country": "United-States",
        "occupation": "Sales",
        "race": "White",
        "relationship": "Husband",
        "sex": "Male",
        "workclass": "Private"
        }
    )

print("POST Status :", post_response.status_code)
print("POST JSON :", post_response.json())
