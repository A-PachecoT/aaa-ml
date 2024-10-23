from behave import given, when, then
from titanic_model import TitanicModel
import numpy as np


@given('the trained model and the test dataset')
def step_trained_model_and_test_data(context):
    context.model = TitanicModel()
    context.X_train, context.X_test, context.y_train, context.y_test = context.model.load_and_preprocess_data('titanic.csv')
    context.model.train_and_validate(context.X_train, context.y_train)

@when('the model predicts survival on the test set')
def step_predict_on_test_set(context):
    context.accuracy, context.precision, context.recall, context.f1 = context.model.evaluate(context.X_test, context.y_test)

@then('the model\'s performance metrics should be greater than {threshold:f}')
def step_check_performance_metrics(context, threshold):
    assert context.accuracy > threshold, f"Accuracy {context.accuracy:.4f} is not greater than the threshold {threshold}"
    assert context.precision > threshold, f"Precision {context.precision:.4f} is not greater than the threshold {threshold}"
    assert context.recall > threshold, f"Recall {context.recall:.4f} is not greater than the threshold {threshold}"
    assert context.f1 > threshold, f"F1-score {context.f1:.4f} is not greater than the threshold {threshold}"
