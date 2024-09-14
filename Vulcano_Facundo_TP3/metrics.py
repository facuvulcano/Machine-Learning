class ClassificationMetrics:
    """
    A class to compute and store classification metrics for binary classifcation problems
    """

    def __init__(self, real_target, predicted_target) -> None:
        self.real_target = real_target
        self.predicted_target = predicted_target
        self.TP, self.TN, self.FP, self.FN = self.count_values()

    def count_values(self):
        """
        Counts the number of true positives (TP), true negatives (TN), false positives (FP)
        and false negatives (FN) from the provided real and predicted target values.
        
        Paramters:
        ----------
        real_target: list or array-like
            The true class labels

        predicted_target: 
            The predicted class labels from the model

        Returns:
        --------
        TP : int
            Number of true positives
        
        TN : int
            Number of true negatives
        
        FP : int
            Number of false positives
        
        FN : int
            Number of false negatives

        Notes:
        ------
        The function iterates over the real and predicted targets and counts the ocurrences in
        each category
        
        """
        TP, TN, FP, FN = 0, 0, 0, 0
        for real, pred in zip(self.real_target, self.predicted_target):
            if real == 1:
                if pred == 1:
                    TP += 1
                elif pred == 0:
                    FN += 1
            elif real == 0:
                if pred == 0:
                    TN += 1
                elif pred == 1:
                    FP += 1
        return TP, TN, FP, FN


    def accuracy(self):
        """
        Calculates the accuracy of the classification model.

        Returns:
        --------
        accuracy_value : float
            The accuracy score
        
        Notes:
        ------
        - Accuracy is the proportion of true results among the total number of cases examined

        """
        total = self.TP +  self.TN + self.FP + self.FN
        if total == 0:
            return 0
        return (self.TP + self.TN) / (total)

    def precision(self):
        """
        Calculates the precision of the clasification model.

        Returns:
        --------
        precision_value : float
            The precision score
        
        Notes:
        ------
        - Precision is the proportion of positive identifications that were actually correct.
        """
        if self.TP + self.FP == 0:
            return 0
        return self.TP / (self.TP + self.FP)
    
    def recall(self):
        """
        Calculates the recall (sensitivity) of the classification model
    
        Returns:
        --------
        recall_value : float
            The recall score
        
        Notes:
        ------
        - Recall is the proportion of actual positives that were correclty ifentified.
        """
        if self.TP + self.FN == 0:
            return 0
        return self.TP / (self.TP + self.FN)

    def f_score(self):
        """
        Calculates the F1 score of the classification model
        
        Returns:
        --------
        f1 : float
            The F1 score.
        
        Notes:
        ------
        - F1 score is the harmonic mean of precision and recall
        """
        prec = self.precision()
        rec = self.recall()
        if prec + rec == 0:
            return 0
        f1 = (2 * prec + rec) / (prec + rec)
        return f1

    def confusion_matrix(self):
        """
        Constructs the confusion matrix for the classification model

        Returns:
        --------
        matrix : list of lists
            a 2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        return [[self.TN, self.FP],
                [self.FN, self.TP]]

    def plot_pr_curve(self):
        """
        Plots the precision-recall curve for the classification model

        Returns:
        --------
        precision_values : array
            Precicion values at various thresholds
        recall_values : array
            Recall valus at various thresholds.
        auc_pr_value : float
            The area under the precision-recall curvee
        
        Notes:
        ------
        - Requires predicted probabilites to plot the curve
        """
        pass

    def plot_roc_curve(self):
        """
        Plots the receiver operating characteristics (ROC) curve for the classification model

        Returns:
        --------
        fpr_values : array
            False positive rates at various thresholds
        tpr_values : array
            True positive rate at various thresholds
        auc_roc_value : float
            The area under the ROC curve.
        
        Notes:
        ------
        - Requires predicted probabilites to plot the curve.
        """


    def auc_pr():
        """
        Calculates the area un the precision-recall curve (AUC-PR)

        Returns:
        --------
        auc_pr_value : float
            The AUC-PR score.
        """
        pass

    def auc_roc():
        """
        Calculates the area under the receiver operating characteristics curve (AUC-ROC)

        Returns:
        --------
        auc_roc_value : float
            The AUC-ROC score
        """
        pass
