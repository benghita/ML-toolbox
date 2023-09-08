import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import pickle

# Regression
class RegressionModelEvaluator:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'MLP': MLPRegressor()
        }
        self.metrics = {
            'Mean Squared Error': make_scorer(mean_squared_error, greater_is_better=False),
            'Mean Absolute Error': make_scorer(mean_absolute_error, greater_is_better=False),
            'R2 Score': make_scorer(r2_score),
            'Explained Variance Score': make_scorer(explained_variance_score),
            'Negative Log Likelihood': make_scorer(lambda y_true, y_pred: -np.log(np.maximum(y_pred, 1e-15)).mean())
        }
        self.trained_models = {}

    def train_and_evaluate_model(self, model_name, X, y, cv=5):
        model = self.models.get(model_name)

        if model is None:
            raise ValueError(f"Model '{model_name}' not found in the available models.")

        model_scores = {}

        # Train the model
        model.fit(X, y)

        for metric_name, metric in self.metrics.items():
            score = np.mean(cross_val_score(model, X, y, cv=cv, scoring=metric))
            model_scores[metric_name] = score

        # Save the trained model
        self.trained_models[model_name] = model

        return model_scores

    def save_models(self):
        return self.trained_models

    def get_best_model(self, X, y, cv=5, scoring='neg_mean_squared_error'):
            if not self.trained_models:
                raise ValueError("No models have been trained yet.")

            best_model_name = max(self.trained_models, key=lambda model_name: np.mean(cross_val_score(self.trained_models[model_name], X, y, cv=cv, scoring=scoring)))
            best_model = self.trained_models[best_model_name]
            best_scores = cross_val_score(best_model, X, y, cv=cv, scoring=scoring)

            return best_model_name, best_model, best_scores

# Classification
class ClassificationModelEvaluator:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'MLP': MLPClassifier()
        }
        self.metrics = {
            'Accuracy': make_scorer(accuracy_score),
            'Precision': make_scorer(precision_score),
            'Recall': make_scorer(recall_score),
            'F1 Score': make_scorer(f1_score)
        }
        self.trained_models = {}

    def train_and_evaluate_model(self, model_name, X, y, cv=5):
        model = self.models.get(model_name)

        if model is None:
            raise ValueError(f"Model '{model_name}' not found in the available models.")

        model_scores = {}

        # Train the model
        model.fit(X, y)

        for metric_name, metric in self.metrics.items():
            score = np.mean(cross_val_score(model, X, y, cv=cv, scoring=metric))
            model_scores[metric_name] = score

        # Save the trained model
        self.trained_models[model_name] = model

        return model_scores

    def save_models(self):
        return self.trained_models

    def get_best_model(self, X, y, cv=5, scoring='accuracy'):
        if not self.trained_models:
            raise ValueError("No models have been trained yet.")

        best_model_name = max(self.trained_models, key=lambda model_name: np.mean(cross_val_score(self.trained_models[model_name], X, y, cv=cv, scoring=scoring)))
        best_model = self.trained_models[best_model_name]
        best_scores = cross_val_score(best_model, X, y, cv=cv, scoring=scoring)

        return best_model_name, best_model, best_scores