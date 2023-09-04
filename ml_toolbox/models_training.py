import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor



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
            'Negative Log Likelihood': make_scorer(lambda y_true, y_pred: -np.log(np.maximum(y_pred, 1e-15)).mean()),
            'Execution Time': make_scorer(lambda _, __, exec_time: -exec_time, greater_is_better=True)
        }

    def evaluate_models(self, X, y, cv=5):
        scores = {}

        for model_name, model in self.models.items():
            model_scores = {}
            for metric_name, metric in self.metrics.items():
                if metric_name == 'Execution Time':
                    start_time = time.time()
                    score = cross_val_score(model, X, y, cv=cv, scoring=metric, fit_params={'exec_time': time.time() - start_time})
                else:
                    score = np.mean(cross_val_score(model, X, y, cv=cv, scoring=metric))
                model_scores[metric_name] = score
            scores[model_name] = model_scores

        score_grid = pd.DataFrame(scores)
        best_model = score_grid.mean().idxmax()

        return best_model, score_grid


# Example usage
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
evaluator = RegressionModelEvaluator()
best_model, score_grid = evaluator.evaluate_models(X, y)
print("Best Model:", best_model)
print(score_grid)

 



# Classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

class ClassificationModelEvaluator:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'SVM': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
            'MLP': MLPClassifier()
        }
        self.metrics = {
            'Accuracy': accuracy_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'F1-Score': f1_score,
            'ROC AUC': roc_auc_score,
            'Execution Time': lambda _, __, exec_time: exec_time
        }

    def evaluate_models(self, X, y, cv=5):
        scores = {}

        for model_name, model in self.models.items():
            model_scores = {}
            for metric_name, metric in self.metrics.items():
                if metric_name == 'Execution Time':
                    start_time = time.time()
                    score = -np.mean(cross_val_score(model, X, y, cv=cv, scoring=make_scorer(metric), fit_params={'exec_time': time.time() - start_time}))
                else:
                    score = np.mean(cross_val_score(model, X, y, cv=cv, scoring=make_scorer(metric)))
                model_scores[metric_name] = score
            scores[model_name] = model_scores

        score_grid = pd.DataFrame(scores)
        best_model = score_grid.mean().idxmax()

        return best_model, score_grid


# Example usage
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

evaluator = ClassificationModelEvaluator()
best_model, score_grid = evaluator.evaluate_models(X_scaled, y)
print("Best Model:", best_model)
print(score_grid)
