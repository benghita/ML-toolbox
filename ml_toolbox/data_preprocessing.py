import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold


# Data Preparation

def handle_missing_values(data, num_imputation_type='mean', categorical_imputation_type='mode',
                          numerical_imputation_value=None, categorical_imputation_value=None):
    """
    Handle missing values in a Pandas DataFrame.

    Parameters:
        - data (DataFrame): The input dataset.
        - num_imputation_type (str): Imputation strategy for missing values in numerical columns.
        - categorical_imputation_type (str): Imputation strategy for missing values in categorical columns.
        - numerical_imputation_value: The value to use for numerical imputation when 'num_imputation_type' is int or float.
        - categorical_imputation_value: The value to use for categorical imputation when 'categorical_imputation_type' is 'str'.

    Returns:
        DataFrame: The original DataFrame with missing values handled according to the specified strategy.
    """
    if num_imputation_type != 'drop':
        numerical_cols = data.select_dtypes(include=['number']).columns
        imputer = SimpleImputer(strategy=num_imputation_type, fill_value=numerical_imputation_value)
        data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

    if categorical_imputation_type != 'drop':
        categorical_cols = data.select_dtypes(include=['object']).columns
        imputer = SimpleImputer(strategy=categorical_imputation_type, fill_value=categorical_imputation_value)
        data[categorical_cols] = imputer.fit_transform(data[categorical_cols])

    return data


def handle_data_types(data, numeric_features=None, categorical_features=None, ignore_features=None):
    """
    Handle data types in a Pandas DataFrame.

    Parameters:
        - data (DataFrame): The input dataset.
        - numeric_features (list of str): List of column names to be treated as numeric features.
        - categorical_features (list of str): List of column names to be treated as categorical features.
        - date_features (list of str): List of column names to be treated as date features.
        - ignore_features (list of str): List of column names to be ignored.

    Returns:
        DataFrame: The DataFrame with updated data types based on the specified parameters.
    """
    if ignore_features:
        data = data.drop(columns=ignore_features)

    if numeric_features:
        data[numeric_features] = data[numeric_features].astype(float)

    if categorical_features:
        data[categorical_features] = data[categorical_features].astype(str)

    return data



def perform_one_hot_encoding(data, columns_to_encode, max_encoding=5):
    """
    Perform one-hot encoding on categorical features in a Pandas DataFrame.

    Parameters:
        - data (DataFrame): The input dataset.
        - columns_to_encode (list of str): List of column names to encode.
        - max_encoding (int): Categorical columns with max_encoding or fewer unique values are encoded using OneHotEncoding directly
          or the one_hot_encoding will be applied on the most max_encoding repeated values in the categorical column.

    Returns:
        DataFrame: The DataFrame with one-hot encoded categorical features.
    """
    encoded_data = data.copy()

    for column in columns_to_encode:
        if column not in encoded_data.columns:
            continue

        unique_values = encoded_data[column].value_counts()

        if len(unique_values) <= max_encoding:
            # If the number of unique values is less than or equal to max_encoding,
            # perform one-hot encoding directly
            encoded_data = pd.get_dummies(encoded_data, columns=[column])
        else:
            # Otherwise, encode the most frequent max_encoding values and replace the rest with 'other'
            top_values = unique_values.head(max_encoding).index
            encoded_data[column] = encoded_data[column].apply(lambda x: x if x in top_values else 'other')

            # Perform one-hot encoding on the modified column
            encoded_data = pd.get_dummies(encoded_data, columns=[column], prefix=[column])

    return encoded_data


def perform_ordinal_encoding(data, ordinal_features=None):
    """
    Perform ordinal encoding on specified columns in a Pandas DataFrame.

    Parameters:
        - data (DataFrame): The input dataset.
        - ordinal_features (dict): Dictionary of column names and their ordinal mappings.

    Returns:
        DataFrame: The DataFrame with ordinal encoded features.
    """

    if ordinal_features:
        encoder = OrdinalEncoder(mapping=ordinal_features)
        data = encoder.fit_transform(data)

    return data


def perform_normalization(data, normalization_method='minmax', features_to_normalize=None):
    """
    Perform normalization on specified numerical features in a Pandas DataFrame.

    Parameters:
        - data (DataFrame): The input dataset.
        - normalization_method (str): The normalization method to use ('zscore', 'minmax', 'robust').
        - features_to_normalize (list of str): List of numerical feature column names to normalize.

    Returns:
        DataFrame: The DataFrame with normalized numerical features.
    """
    if normalization_method == 'zscore':
        scaler = StandardScaler()
    elif normalization_method == 'minmax':
        scaler = MinMaxScaler()
    elif normalization_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid normalization method. Use 'zscore', 'minmax', or 'robust'.")

    if features_to_normalize:
        data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
    else:
        data[data.select_dtypes(include=['number']).columns] = scaler.fit_transform(data.select_dtypes(include=['number']))

    return data


# Feature Selection
def perform_feature_selection(data, target_column, feature_selection_method='classic'):
    """
    Perform feature selection on a Pandas DataFrame.

    Parameters:
        - data (DataFrame): The input dataset.
        - target_column (str): The target variable column name.
        - feature_selection_method (str): The feature selection method ('univariate', 'classic', 'sequential').

    Returns:
        None (Modifies the input DataFrame in-place).
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if feature_selection_method == 'univariate':
        selector = SelectKBest()
    elif feature_selection_method == 'classic':
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0))
    elif feature_selection_method == 'sequential':
        selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators=100, random_state=0))
    else:
        raise ValueError("Invalid feature selection method.")

    selected_data = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]

    data.drop(columns=X.columns.difference(selected_columns), inplace=True)


def remove_low_variance_features(data, threshold=0.01):
    """
    Remove low variance features from a Pandas DataFrame.

    Parameters:
        - data (DataFrame): The input dataset.
        - threshold (float): Variance threshold below which features will be removed.

    Returns:
        DataFrame: The DataFrame with low variance features removed.
    """
    selector = VarianceThreshold(threshold=threshold)
    high_variance_data = selector.fit_transform(data)

    selected_columns = data.columns[selector.get_support()]
    high_variance_df = pd.DataFrame(data=high_variance_data, columns=selected_columns)

    return high_variance_df
