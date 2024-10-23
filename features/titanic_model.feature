Feature: Titanic Survival Prediction Model

  Scenario: Evaluate the model on the test set
    Given the trained model and the test dataset
    When the model predicts survival on the test set
    Then the model's performance metrics should be greater than 0.75

