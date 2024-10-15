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
