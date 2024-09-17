import pandas as pd
import numpy as np

def train_val_split(df: pd.DataFrame, target):
    """
    Splits a DataFrame into training and validation (test) sets.

    This function shuffles the input DataFrame and then splits it into training and test sets
    based on a fixed test size ratio. It separates the features and targets variable for both sets.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the dataset to split.
        It includes both feature columns and the target variable column.

    target : str
        The name of the target column in the DataFrame that needs to be predicted.
        This column is separated from the feature set.
    
    Returns:
    --------
    X_train : pd.DataFrame
        The training set features, excluding the target columns.
    
    X_test : pd.DataFrame
        The test set features, excluding the target columns.
    
    y_train : pd.Series
        The training set target values.
    
    y_test : pd.Series
        The test set target values.
    
    Notes:
    ------
    - The function currently uses a fixed test size ratio of 0.2 (20% for test and 80% for training).
    - The input DataFrame is shufflled before splitting to ensure randomness in the selection of training and test data.
    """
    df = df.sample(frac=1).reset_index(drop=True)
    test_size = 0.2
    test_set_size = int(len(df) * test_size)
    train_set_size = len(df) - test_set_size
    df_train = df.iloc[:train_set_size]
    df_val = df.iloc[train_set_size:]
    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    X_val = df_val.drop(target, axis=1)
    y_val = df_val[target]
    return X_train, X_val, y_train, y_val