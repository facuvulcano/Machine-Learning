import numpy as np

def rmse(targets, target_prediction):
    sum_error = 0
    for i in range(len(targets)):
        real_y = targets[i]
        predicted_y = target_prediction[i]
        sum_error += (real_y - predicted_y)**2
    return np.sqrt(sum_error/len(targets))