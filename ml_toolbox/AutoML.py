
from . import data_preprocessing as dp
from . import models_training as mt
import pandas as pd
class autoML :


    def __init__(self, df, target):
        self.df = df
        self.target = target
        

    def handle_missing_and_types(self,
                                 num_imputation_type,
                                 categorical_imputation_type,
                                 numerical_imputation_value,
                                 categorical_imputation_value,
                                 numeric_features,
                                 categorical_features,
                                 ignore_features):
        
        self.df = dp.handle_missing_values(self.df, num_imputation_type, categorical_imputation_type,
                                   numerical_imputation_value, categorical_imputation_value)
        self.df = dp.handle_data_types(self.df, numeric_features, categorical_features, ignore_features)
        
        if True : self.feautures = self.df.drop(self.target, axis=1)

    def handle_encoding_and_normalization(self,
                                          columns_to_encode,
                                          max_encoding,
                                          #ordinal_features,
                                          normalization_method,
                                          features_to_normalize):
        self.feautures = dp.perform_one_hot_encoding(self.feautures, columns_to_encode, max_encoding)
        #self.feautures = dp.perform_ordinal_encoding(self.feautures, ordinal_features)
        self.feautures = dp.perform_normalization(self.feautures, normalization_method, features_to_normalize)
        self.df = pd.concat([self.feautures, self.df[self.target]], axis=1)

    def selection (self, feature_selection_method, threshold):

        dp.remove_low_variance_features(self.feautures, threshold)
        self.df = pd.concat([self.feautures, self.df[self.target]], axis=1)
        dp.perform_feature_selection(self.df, self.target, feature_selection_method)
    
    def define_task(self):

        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]

        column = self.df[self.target].dtype
        if pd.api.types.is_numeric_dtype(column):
            self.model_evaluator = mt.RegressionModelEvaluator()
            return "regression"
        elif column.dtype == 'object':
            self.model_evaluator = mt.ClassificationModelEvaluator()
            return "classification"
        else:
            return "other"

    def linear_regression(self):
        return self.model_evaluator.train_and_evaluate_model('Linear Regression', self.X, self.y)
    def ridge(self):
        return self.model_evaluator.train_and_evaluate_model('Ridge', self.X, self.y)
    def lasso(self):
        return self.model_evaluator.train_and_evaluate_model('Lasso', self.X, self.y)
    def decision_tree(self):
        return self.model_evaluator.train_and_evaluate_model('Decision Tree', self.X, self.y)
    def random_forest(self):
        return self.model_evaluator.train_and_evaluate_model('Random Forest', self.X, self.y)
    def gradient_boosting(self):
        return self.model_evaluator.train_and_evaluate_model('Gradient Boosting', self.X, self.y)
    def AdaBoost(self):
        return self.model_evaluator.train_and_evaluate_model('AdaBoost', self.X, self.y)
    def SVR(self):
        return self.model_evaluator.train_and_evaluate_model('SVR', self.X, self.y)
    def KNN(self):
        return self.model_evaluator.train_and_evaluate_model('KNN', self.X, self.y)
    def MLP(self):
        return self.model_evaluator.train_and_evaluate_model('MLP', self.X, self.y)
    def logistic_regression(self):
        return self.model_evaluator.train_and_evaluate_model('Logistic Regression', self.X, self.y)
    
    def best_model(self):
        return self.model_evaluator.get_best_model(self.X, self.y)
    def save_models(self):
        self.model_evaluator.save_models('trained_classification_models.pkl')

    