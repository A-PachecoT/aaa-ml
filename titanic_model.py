import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class TitanicModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self, filepath):
        data = pd.read_csv(filepath)
        
        # Basic preprocessing
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
        data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        X = data[features]
        y = data['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_and_validate(self, X_train, y_train):
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        self.model.fit(X_train, y_train)
        return cv_scores

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        return accuracy, precision, recall, f1

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        self.model = joblib.load(filepath)

if __name__ == "__main__":
    model = TitanicModel()
    X_train, X_test, y_train, y_test = model.load_and_preprocess_data('titanic.csv')
    
    cv_scores = model.train_and_validate(X_train, y_train)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {np.mean(cv_scores):.4f}")

    accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
    print(f"Test set performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    model.save_model('titanic_model.joblib')
