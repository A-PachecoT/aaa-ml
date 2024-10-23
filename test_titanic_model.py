import pytest
import numpy as np
from titanic_model import TitanicModel

@pytest.fixture
def titanic_model():
    return TitanicModel()

def test_train_and_validate(titanic_model):
    # Arrange
    X_train, X_test, y_train, y_test = titanic_model.load_and_preprocess_data('titanic.csv')

    # Act
    cv_scores = titanic_model.train_and_validate(X_train, y_train)

    # Assert
    assert len(cv_scores) == 5  # 5-fold cross-validation
    assert np.mean(cv_scores) >= 0.8  # Minimum required accuracy

def test_evaluate(titanic_model):
    # Arrange
    X_train, X_test, y_train, y_test = titanic_model.load_and_preprocess_data('titanic.csv')
    titanic_model.train_and_validate(X_train, y_train)

    # Act
    accuracy, precision, recall, f1 = titanic_model.evaluate(X_test, y_test)

    # Assert
    assert accuracy >= 0.8
    assert precision >= 0.75
    assert recall >= 0.75
    assert f1 >= 0.75
