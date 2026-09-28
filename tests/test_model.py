"""
Test the Model traning and output
Date : 28-09-2026
Author : Sourodeep Banerjee
"""
import numpy as np
import pandas as pd
import pytest

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


def test_compute_model_metrics():
    precision, recall, f1 = compute_model_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    )

    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(0.8)


def test_train_model_fits_and_predicts():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = train_model(X, y)

    assert len(model.predict(X)) == len(y)


def test_inference_returns_one_prediction_per_row():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    model = train_model(X, y)

    assert inference(model, X[:2]).shape == (2,)


def test_process_data_reuses_encoder_for_unknown_category():
    train = pd.DataFrame(
        {
            "age": [20, 40],
            "workclass": ["Private", "Public"],
            "salary": ["<=50K", ">50K"],
        }
    )
    X_train, y_train, encoder, lb = process_data(
        train, ["workclass"], label="salary", training=True
    )
    new_row = pd.DataFrame({"age": [30], "workclass": ["Other"]})
    X_new, y_new, same_encoder, same_lb = process_data(
        new_row,
        ["workclass"],
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert X_train.shape == (2, 3)
    assert y_train.tolist() == [0, 1]
    assert X_new[0, 1:].tolist() == [0.0, 0.0]
    assert y_new.size == 0
    assert same_encoder is encoder
    assert same_lb is lb
