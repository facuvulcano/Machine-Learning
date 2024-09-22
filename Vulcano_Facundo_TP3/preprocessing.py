def normalize(X, feature, min_val, max_val, rango):
    min_max_scaler = rango[0] + ((X[feature] - min_val) / (max_val - min_val)) * (rango[1] - rango[0])
    return min_max_scaler