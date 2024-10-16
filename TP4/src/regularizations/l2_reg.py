def l2_regularization(model, lambda_reg):
    l2_sum = 0.0
    for param in model.parameters():
        if hasattr(param, 'data'):
            l2_sum += param.data ** 2
    return lambda_reg * l2_sum