import sys
sys.path.append('/home/facuvulcano/Machine-Learning/TP4/src')
from regularizations.l2_reg import l2_regularization

def mse_loss(y_true, y_pred):
    return (y_pred - y_true)**2


def total_loss(y_true, y_pred, model, lambda_reg):
    return mse_loss(y_true, y_pred) + l2_regularization(model, lambda_reg)
