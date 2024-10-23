from behave import given, when, then
from titanic_model import TitanicModel
import numpy as np

@given('the Titanic dataset is loaded and preprocessed')
def step_load_and_preprocess_data(context):
    context.model = TitanicModel()
    context.X_train, context.X_test, context.y_train, context.y_test = context.model.load_and_preprocess_data('titanic.csv')

@when('the model is trained using cross-validation')
def step_train_and_validate_model(context):
    context.cv_scores = context.model.train_and_validate(context.X_train, context.y_train)

@then('the model\'s cross-validation scores should meet the acceptance criteria')
def step_check_cv_scores(context):
    mean_cv_score = np.mean(context.cv_scores)
    assert mean_cv_score >= 0.8, f"Mean CV score {mean_cv_score:.4f} is below the acceptance criteria of 0.8"

@given('the trained model and the test dataset')
def step_trained_model_and_test_data(context):
    # This step is already covered by the previous steps, so we can pass
    pass

@when('the model predicts survival on the test set')
def step_predict_on_test_set(context):
    context.accuracy, context.precision, context.recall, context.f1 = context.model.evaluate(context.X_test, context.y_test)

@then('the model\'s performance metrics should meet the acceptance criteria')
def step_check_performance_metrics(context):
    assert context.accuracy >= 0.8, f"Accuracy {context.accuracy:.4f} is below the acceptance criteria of 0.8"
    assert context.precision >= 0.75, f"Precision {context.precision:.4f} is below the acceptance criteria of 0.75"
    assert context.recall >= 0.75, f"Recall {context.recall:.4f} is below the acceptance criteria of 0.75"
    assert context.f1 >= 0.75, f"F1-score {context.f1:.4f} is below the acceptance criteria of 0.75"

@given('a trained model that meets the acceptance criteria')
def step_trained_model(context):
    # This step is already covered by the previous steps, so we can pass
    pass

@when('the model is saved to disk')
def step_save_model(context):
    context.model.save_model('titanic_model_test.joblib')

@when('the model is loaded from disk')
def step_load_model(context):
    context.loaded_model = TitanicModel()
    context.loaded_model.load_model('titanic_model_test.joblib')

@then('the loaded model should produce identical predictions to the original model')
def step_compare_predictions(context):
    original_predictions = context.model.model.predict(context.X_test)
    loaded_predictions = context.loaded_model.model.predict(context.X_test)
    assert np.array_equal(original_predictions, loaded_predictions), "Loaded model predictions do not match original model predictions"
