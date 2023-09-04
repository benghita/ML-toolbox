
from . import data_preprocessing as dp
#from models_training import *
#from model_evaluation import *
import pandas as pd
import plotly.express as px
import numpy as np
class autoML :
    
    
    def __init__(self, df):
        self.df = df

    def is_any_variable_not_empty(*args):
        for var in args:
            if var:  # Checks if the variable is not empty or evaluates to True
                return True  # At least one variable is not empty
        return False  # All variables are empty

    def handle_missing_and_types(self,
                                 num_imputation_type,
                                 categorical_imputation_type,
                                 numerical_imputation_value,
                                 categorical_imputation_value,
                                 numeric_features,
                                 categorical_features,
                                 date_features,
                                 ignore_features):
        
        self.df = dp.handle_missing_values(self.df, num_imputation_type, categorical_imputation_type,
                                   numerical_imputation_value, categorical_imputation_value)
        self.df = dp.handle_data_types(self.df, numeric_features, categorical_features, date_features, ignore_features)

    def handle_encoding_and_normalization(self,
                                          columns_to_encode,
                                          max_encoding,
                                          ordinal_features,
                                          normalization_method,
                                          features_to_normalize):
        self.df = dp.perform_one_hot_encoding(self.df, columns_to_encode, max_encoding)
        self.df = dp.perform_ordinal_encoding(self.df, ordinal_features)
        self.df = dp.perform_normalization(self.df, normalization_method, features_to_normalize)

    def visualize_data(self):

        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        # Visualize the correlation matrix
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        fig_1 = px.imshow(self.df[num_cols])

        # Visualize the distribution of each feature
        #for col in self.df.select_dtypes(include=np.number).columns.tolist():
         #   fig_2 = px.histogram(self.df, x=col)

        # Visualize the heatmap of each feature
        fig_3 = px.imshow(self.df[num_cols])

        return fig_1, fig_3
        


    def selection (self, target_column, feature_selection_method, threshold):
        self.df = self.df
        dp.perform_feature_selection(self.df, target_column, feature_selection_method)
        dp.remove_low_variance_features(self.df, threshold)

    