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


def cross_val(df: pd.DataFrame, target, folds):
    """
    Performs k-fold cross-validation on a DataFrame.

    This function splits the input DataFrame into folds number of folds, shuffling the data beforehand.
    For each fold, it separates the data into training and test sets, ensuring each data point is used once
    as a test sample and folds - 1 times as a training sample. It returns the training and test sets for each
    fold.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the dataset to be used for cross-validation.
        It should include both features columns and the target variable column.
    
    target : str
        The name of the target column in the DataFrame that needs to be predicted.
        This column is separated from the features set.
    
    folds : int
        The number of folds to create for cross-validation.
        This determines how many times the data will be split into training and test sets.
    
    Returns:
    --------
    training : list of tuples
        A list where each element is a tuple containing the training features (X_train) and
        training target (y_train) for a particular fold.
    
    testing : list of tuples
        A list where each element is a tuple containing the test features (X_test) and
        test target (y_test) for a particular fold.
    
    Notes:
    ------
    - The function shuffles the data before splitting to ensure randomness.
    - The DataFrame is split into approximately equal-sized folds, except possibly the last fold if
    the number of rows in the DataFrame isn't perfectly divisible by the number of folds.
    - This method is useful for assessing the performance of a model on different subsets of data and helps
    prevent overfitting by ensuring that each data point gets a chance to be in the training and test set.
    """
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    fold_size = len(df) // folds
    training = []
    validation = []
    for fold in range(folds):
        start, stop = fold * fold_size, (fold + 1) * fold_size
        val_idx = indices[start:stop]

        train_idx = np.concatenate([indices[:start], indices[stop:]])
        df_train = df.iloc[train_idx]
        df_test = df.iloc[val_idx]

        X_train, y_train = df_train.drop(columns=[target]), df_train[target]
        X_test, y_test = df_test.drop(columns=[target]), df_test[target]

        training.append((X_train, y_train))
        validation.append((X_test, y_test))

    return training, validation
