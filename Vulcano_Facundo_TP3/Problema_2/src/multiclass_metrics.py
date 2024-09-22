import numpy as np

class ClassificationMetrics():
    def __init__(self, y_test, y_pred) -> None:
        self.y_test = y_test
        self.y_pred =  y_pred

    def accuracy(self):
        return np.sum(self.y_test == self.y_pred) / len(self.y_test)
    
    def precision(self, class_):
        y_test_class_ = (self.y_test == class_).astype(int)
        y_pred_class_ = (self.y_pred == class_).astype(int)

        tp = np.sum((y_test_class_ == 1) & (y_pred_class_== 1))
        fp = np.sum((y_test_class_ == 0) & (y_pred_class_ == 1))

        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self, class_):
        y_test_class_ = (self.y_test == class_).astype(int)
        y_pred_class_ = (self.y_pred == class_).astype(int)

        tp = np.sum((y_test_class_ == 1) & (y_pred_class_ == 1))
        fn = np.sum((y_test_class_ == 1) & (y_pred_class_ == 0))

        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f_score(self):
        pass

    def auc_roc(self):
        pass
    
    def auc_pr(self):
        pass

    def plot_roc(self):
        pass

    def plot_pr(self):
        pass

