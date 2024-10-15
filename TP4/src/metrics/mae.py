def mae(targets, target_prediction):
    sum_error = 0
    for i in range(len(targets)):
        real_y = targets[i]
        predicted_y = target_prediction[i]
        sum_error += abs(real_y - predicted_y)
    return sum_error / len(targets)