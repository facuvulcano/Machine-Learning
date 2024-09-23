import numpy as np
import matplotlib.pyplot as plt

class ClassificationMetrics():
    def __init__(self, y_test, y_pred, label, probas)-> None:
        self.y_test = y_test
        self.y_pred =  y_pred
        self.label = label
        self.y_test_class = (self.y_test == self.label).astype(int)
        self.y_pred_class = (self.y_pred == self.label).astype(int)
        self.y_pred_proba_class = probas[:, self.label]


    def accuracy(self):
        return np.sum(self.y_test == self.y_pred) / len(self.y_test)
    
    def precision(self):
        tp = np.sum((self.y_test_class == 1) & (self.y_pred_class== 1))
        fp = np.sum((self.y_test_class == 0) & (self.y_pred_class == 1))

        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self):
        tp = np.sum((self.y_test_class == 1) & (self.y_pred_class == 1))
        fn = np.sum((self.y_test_class == 1) & (self.y_pred_class == 0))

        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f_score(self):
        prec = self.precision()
        recc = self.recall()
        if (prec + recc) == 0:
            return 0.0
        return (2 * prec * recc) / (prec + recc)
    
    def confusion_matrix(self):
        tp = np.sum((self.y_test_class == 1) & (self.y_pred_class== 1))
        fp = np.sum((self.y_test_class == 0) & (self.y_pred_class == 1))
        fn = np.sum((self.y_test_class == 1) & (self.y_pred_class == 0))  
        tn = np.sum((self.y_test_class == 0) & (self.y_pred_class == 0))

        return np.array([[tn, fp],
                         [fn, tp]])
    
    def plot_confusion_matrix(self):
        cm = self.confusion_matrix()
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for class {self.label}')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
        plt.yticks(tick_marks, ['Negative', 'Positive'])

        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color='white' if cm[i, j] > thresh else "black")
            
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def auc_roc(self):
        thresholds = np.concatenate(([0], np.sort(np.unique(self.y_pred_proba_class)), [1]))
        tpr_list = []
        fpr_list = []

        for threshold in thresholds:
            y_pred_threshold = (self.y_pred_proba_class >= threshold).astype(int)
            tp = np.sum((self.y_test_class == 1) & (y_pred_threshold == 1))
            fn = np.sum((self.y_test_class == 1) & (y_pred_threshold == 0))
            fp = np.sum((self.y_test_class == 0) & (y_pred_threshold == 1))
            tn = np.sum((self.y_test_class == 0) & (y_pred_threshold == 0))

            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        fpr_list, tpr_list = zip(*sorted(zip(fpr_list, tpr_list)))

        return np.trapz(tpr_list, fpr_list), tpr_list, fpr_list

    def auc_pr(self):
        thresholds = np.concatenate(([0], np.sort(np.unique(self.y_pred_proba_class)), [1]))
        precision_list = []
        recall_list = []

        for threshold in thresholds:
            y_pred_threshold = (self.y_pred_proba_class >= threshold).astype(int)
            tp = np.sum((self.y_test_class == 1) & (y_pred_threshold == 1))
            fn = np.sum((self.y_test_class == 1) & (y_pred_threshold == 0))
            fp = np.sum((self.y_test_class == 0) & (y_pred_threshold == 1))

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            precision_list.append(precision)
            recall_list.append(recall)
    
        recall_list, precision_list = zip(*sorted(zip(recall_list, precision_list)))

        return np.trapz(precision_list, recall_list), precision_list, recall_list

    def plot_roc(self):

        auc, tpr_list, fpr_list = self.auc_roc()

        plt.figure()
        plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.02f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic for class {self.label}')
        plt.legend(loc="lower right")
        plt.show()

    def plot_pr(self):

        auc, precision_list, recall_list = self.auc_pr()

        plt.figure()
        plt.plot(precision_list, recall_list, color='darkorange', lw=2, label=f'PR curve (AUC = {auc:.02f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title(f'Precision Recall for class {self.label}')
        plt.legend(loc="lower right")
        plt.show()


