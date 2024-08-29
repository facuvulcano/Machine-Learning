import numpy as np

def rmse(N, targets, target_prediction):
    sum = 0
    for i in range(N):
        real_y = targets[i]
        predicted_y = target_prediction[i]
        sum += (real_y - predicted_y)**2
    return np.sqrt((1/N) * sum)

def mae(N, targets, target_prediction):
    sum = 0
    for i in range(N):
        real_y = targets[i]
        predicted_y = target_prediction[i]
        sum += abs(real_y - predicted_y)
    return (1/N) * sum   

def r2(N, targets, target_prediction):
    residual_sum_of_squares = 0
    total_sum_of_squares = 0
    y_mean = sum(targets) / N
    for i in range(N):
        real_y = targets[i]
        predicted_y = target_prediction[i]
        residual_sum_of_squares += (real_y - predicted_y)**2
        total_sum_of_squares += (real_y - y_mean)**2
    return 1 - (residual_sum_of_squares / total_sum_of_squares)
