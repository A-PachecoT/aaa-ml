import pytest
import numpy as np
from titanic_model import TitanicModel

@pytest.fixture
def titanic_model():
    return TitanicModel()

def test_load_and_preprocess_data(titanic_model):
    # Arrange
    expected_feature_count = 7
    expected_train_size = 0.8
    expected_test_size = 0.2

    # Act
    X_train, X_test, y_train, y_test = titanic_model.load_and_preprocess_data('titanic.csv')

    # Assert
    assert X_train.shape[1] == expected_feature_count
    assert X_test.shape[1] == expected_feature_count
    assert len(y_train) / (len(y_train) + len(y_test)) == pytest.approx(expected_train_size, abs=0.01)
    assert len(y_test) / (len(y_train) + len(y_test)) == pytest.approx(expected_test_size, abs=0.01)

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

def test_save_and_load_model(titanic_model, tmp_path):
    # Arrange
    X_train, X_test, y_train, y_test = titanic_model.load_and_preprocess_data('titanic.csv')
    titanic_model.train_and_validate(X_train, y_train)
    model_path = tmp_path / "test_model.joblib"

    # Act
    titanic_model.save_model(model_path)
    loaded_model = TitanicModel()
    loaded_model.load_model(model_path)

    # Assert
    original_predictions = titanic_model.model.predict(X_test)
    loaded_predictions = loaded_model.model.predict(X_test)
    assert np.array_equal(original_predictions, loaded_predictions)
