import pandas as pd
import numpy as np

def train_val_split(df: pd.DataFrame, target):
    df = df.sample(frac=1).reset_index(drop=True)
    test_size = 0.2
    test_set_size = int(len(df) * test_size)
    train_set_size = len(df) - test_set_size
    df_train = df.iloc[:train_set_size]
    df_test = df.iloc[train_set_size:]
    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]
    return X_train, X_test, y_train, y_test

def cross_val(df: pd.DataFrame, target, folds):
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    fold_size = len(df) // folds
    training = []
    testing = []
    for fold in range(folds):
        start, stop = fold * fold_size, (fold + 1) * fold_size
        test_idx = indices[start:stop]

        train_idx = np.concatenate([indices[:start], indices[stop:]])
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        X_train, y_train = df_train.drop(columns=[target]), df_train[target]
        X_test, y_test = df_test.drop(columns=[target]), df_test[target]

        training.append((X_train, y_train))
        testing.append((X_test, y_test))

    return training, testing



    
