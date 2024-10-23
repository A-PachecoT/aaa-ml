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

@then('the model\'s performance metrics should meet the acceptance criteria')
def step_check_performance_metrics(context):
    assert context.accuracy >= 0.8, f"Accuracy {context.accuracy:.4f} is below the acceptance criteria of 0.8"
    assert context.precision >= 0.75, f"Precision {context.precision:.4f} is below the acceptance criteria of 0.75"
    assert context.recall >= 0.75, f"Recall {context.recall:.4f} is below the acceptance criteria of 0.75"
    assert context.f1 >= 0.75, f"F1-score {context.f1:.4f} is below the acceptance criteria of 0.75"
