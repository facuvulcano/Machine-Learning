import sys
import os
import pandas as pd

from datasets.preprocessing import processing, normalize
from datasets.train_validation import train_val_split


def load_and_prepare_data():
    
    # Cargar datos
    df = pd.read_csv('/home/facuvulcano/Machine-Learning/TP4/data/raw/toyota_dev.csv')
    test_df = pd.read_csv('/home/facuvulcano/Machine-Learning/TP4/data/raw/toyota_test.csv')
    
    # Procesar datos
    df_path = processing(df)
    test_df_path = processing(test_df)

    processed_df = pd.read_csv(df_path)
    processed_test_df = pd.read_csv(test_df_path)

    # Separar caracteristicas y variables objetivo
    X_test = processed_test_df.drop(columns=['Precio'])
    y_test = processed_test_df['Precio']

    # Dividir el conjunto de desarrollo en entrenamiento (80%) y validacion (20%)
    X_train, X_val, y_train, y_val = train_val_split(processed_df, 'Precio')

    # Normalizacion de todas las caracteristicas de entrada
    features_to_normalize = ['Año','Kilómetros']
    min_max_values = {feature : (X_train[feature].min(), X_train[feature].max()) for feature in features_to_normalize}


    for feature in features_to_normalize:
        min_val, max_val = min_max_values[feature]
        X_train[feature] = normalize(X_train, feature, min_val, max_val, [0, 1])
        X_val[feature] = normalize(X_val, feature, min_val, max_val, [0, 1])
        X_test[feature] = normalize(X_test, feature, min_val, max_val, [0, 1])

    y_min = y_train.min()
    y_max = y_train.max()

    y_train_norm = (y_train - y_min) / (y_max - y_min)
    y_val_norm = (y_val - y_min) / (y_max - y_min)
    y_test_norm = (y_test - y_min) / (y_max - y_min)

    y_train_norm = y_train_norm.values
    y_val_norm = y_val_norm.values
    y_test_norm = y_test_norm.values

    return X_train, X_val, y_train_norm, y_val_norm, X_test, y_test_norm, y_min, y_max