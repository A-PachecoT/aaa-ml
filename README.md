# User Story

As a data scientist,
I want to develop a reliable machine learning model for classifying Titanic survival data,
So that I can accurately predict passenger survival and gain insights from the dataset.

## Acceptance Criteria

1. The model should achieve a minimum accuracy of 75% on the test set.
2. The model should have a minimum precision of 75% for both survival and non-survival classes.
3. The model should have a minimum recall of 75% for both survival and non-survival classes.
4. The model's F1-score should be at least 75% for both classes.
5. The model should use cross-validation during training to ensure robustness.
6. The entire process of data loading, preprocessing, model training, and evaluation should be automated.

## Gherkin Scenarios
```gherkin
Feature: Titanic Survival Prediction Model

  Scenario: Evaluate the model on the test set
    Given the trained model and the test dataset
    When the model predicts survival on the test set
    Then the model's performance metrics should be greater than 0.75
```