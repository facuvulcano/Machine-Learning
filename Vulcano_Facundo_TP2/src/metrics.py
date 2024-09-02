import numpy as np

def rmse(targets, target_prediction):
    sum_error = 0
    for i in range(len(targets)):
        real_y = targets[i]
        predicted_y = target_prediction[i]
        sum_error += (real_y - predicted_y)**2
    return np.sqrt(sum_error/len(targets))


def mae(targets, target_prediction):
    sum_error = 0
    for i in range(len(targets)):
        real_y = targets[i]
        predicted_y = target_prediction[i]
        sum_error += abs(real_y - predicted_y)
    return sum_error / len(targets)

def r2(targets, target_prediction):
    residual_sum_of_squares = 0
    total_sum_of_squares = 0
    y_mean = sum(targets) / len(targets)
    for i in range(len(targets)):
        real_y = targets[i]
        predicted_y = target_prediction[i]
        residual_sum_of_squares += (real_y - predicted_y)**2
        total_sum_of_squares += (real_y - y_mean)**2
    return 1 - (residual_sum_of_squares / total_sum_of_squares)
