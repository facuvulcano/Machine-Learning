import numpy as np
import pandas as pd

class Node():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:

        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.left = None
        self.right = None


class DecisionTree():

    def __init__(self,
                 max_depth,
                 min_samples_leaf,
                 min_information_gain):
        self.max_depth  = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = self.min_information_gain
        

    def entropy(self, class_probabilties: list) -> list:
        pass

    def class_probabilites(self, labels: list) -> list:
        pass

    def datta_entropy(self, labels: list) -> float:
        pass

    def partition_entropy(self, suvsets:list) ->  float:
        pass

    def split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        pass

    def find_best_split(self, data: np.array) -> tuple:
        pass

    def find_label_probs(self, data: np.array) -> np.array:
        pass
    
    def create_tree(self, data: np.array, current_depth: int) -> Node:
        pass

    def predict_one_sample(self, X: np.array) -> np.array:
        pass

    def train(self, X_train: np.array, Y_train: np,array) -> None:
        pass

    def predict_probs(self, X_set: np.array) -> np.array:
        pass

    def predict(self, X_set: np.array) -> np.array:
        pass

    def plot_decision_tree(self):
        pass


class LogisticRegression:
    def __init__(self, threshold=0.5, max_iter=1000, learning_rate=0.01, lambda_penalty=0.01):
        """
        Logistic Regression without re balancing technique
        threshold: threshold value to classify as class 1 (default 0.5)
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        lambda_penalty: L2 regularization lambda penalty
        """
        self.threshold = threshold
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_penalty = lambda_penalty
        self.coef_ = None
        self.intercept_ = None
    
    def _sigmoid(self, z):
        """
        Sigmoid function to transform inputs into probabilities.
        z: scalar or numpy array
        """
        return 1 / (1 + np.exp(-z))
    
    def _add_intercept(self, X):
        """
        Adds column of 1s to X for the intercept (bias) term.
        X: input feature matrix
        """
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y):
        """
        Fits the logistic regression model to the data points 
        using gradient descent.
        X: design matrix (n_samples, n_features)
        y: labels vector (n_samples,)
        """
        X = np.array(X)
        X = self._add_intercept(X)
        y = np.array(y)
        
        # Initialize the coefficients
        self.coef_ = np.zeros(X.shape[1])
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Predict probability
            z = np.dot(X, self.coef_)
            y_hat = self._sigmoid(z)
            # NLL gradient
            gradient = np.dot(X.T, (y_hat - y)) / y.size
            regularization_term = (self.lambda_penalty / y.size) * self.coef_
            regularization_term[0] = 0
            # Update coefficients
            self.coef_ -= self.learning_rate * (gradient + regularization_term)
        
        self.intercept_ = self.coef_[0] # Intercept is the fist value of coef_
        self.coef_ = self.coef_[1:]
    
    def predict_proba(self, X):
        """
        Predicts probabilities for each class for inputs X.
        X: design matrix (n_samples, n_features)
        """
        X = self._add_intercept(X)
        prob_positive = self._sigmoid(np.dot(X, np.r_[self.intercept_, self.coef_]))
        prob_negative = 1 - prob_positive
        return np.vstack((prob_negative, prob_positive)).T
    
    def predict(self, X):
        """
        Predicts class (0 or 1) for the inputs X using a threshold.
        X: design matrix (n_samples, n_features)
        """
        probas = self.predict_proba(X)
        return (probas >= self.threshold).astype(int)
    

class LogisticRegressionUndersampling:
    def __init__(self, df: pd.DataFrame, threshold=0.5, max_iter=1000, learning_rate=0.01, lambda_penalty=0.01):
        """
        Logistic Regression using undersampling: randomly eliminates samples from majority class
        threshold: threshold value to classify as class 1 (default 0.5)
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        lambda_penalty: L2 regularization lambda penalty
        """
        self.df = df
        self.threshold = threshold
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_penalty = lambda_penalty
        self.coef_ = None
        self.intercept_ = None

    def _undersample(self, target_majority_class, target_minority_class):

        majority_class_idxs = np.array(self.df.index[self.df['target'] == target_majority_class]) 
        minority_class_idxs = np.array(self.df.index[self.df['target'] == target_minority_class])
        diff = len(majority_class_idxs) - len(minority_class_idxs)
        np.random.shuffle(majority_class_idxs)
        majority_class_idxs_to_drop = majority_class_idxs[:diff]
        self.df = self.df.drop(index=majority_class_idxs_to_drop)

    def undersampling(self):

        class0 = self.df[self.df['target'] == 0]['target']
        class1 = self.df[self.df['target'] == 1]['target']
        if len(class0) > len(class1):
            self._undersample(0, 1)
        elif len(class1) > len(class0):
            self._undersample(1, 0)
        else:
            raise ValueError("Classes are balanced, undersampling is not required.")
    
    def _sigmoid(self, z):
        """
        Sigmoid function to transform inputs into probabilities.
        z: scalar or numpy array
        """
        return 1 / (1 + np.exp(-z))
    
    def _add_intercept(self, X):
        """
        Adds column of 1s to X for the intercept (bias) term.
        X: input feature matrix
        """
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y):
        """
        Fits the logistic regression model to the data points 
        using gradient descent.
        X: design matrix (n_samples, n_features)
        y: labels vector (n_samples,)
        """
        X = np.array(X)
        X = self._add_intercept(X)
        y = np.array(y)
        
        # Initialize the coefficients
        self.coef_ = np.zeros(X.shape[1])
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Predict probability
            z = np.dot(X, self.coef_)
            y_hat = self._sigmoid(z)
            # NLL gradient
            gradient = np.dot(X.T, (y_hat - y)) / y.size
            regularization_term = (self.lambda_penalty / y.size) * self.coef_
            regularization_term[0] = 0
            # Update coefficients
            self.coef_ -= self.learning_rate * (gradient + regularization_term)
        
        self.intercept_ = self.coef_[0] # Intercept is the fist value of coef_
        self.coef_ = self.coef_[1:]
    
    def predict_proba(self, X):
        """
        Predicts probabilities for each class for inputs X.
        X: design matrix (n_samples, n_features)
        """
        X = self._add_intercept(X)
        prob_positive = self._sigmoid(np.dot(X, np.r_[self.intercept_, self.coef_]))
        prob_negative = 1 - prob_positive
        return np.vstack((prob_negative, prob_positive)).T
    
    def predict(self, X):
        """
        Predicts class (0 or 1) for the inputs X using a threshold.
        X: design matrix (n_samples, n_features)
        """
        probas = self.predict_proba(X)
        return (probas >= self.threshold).astype(int)
    
class LogisticRegressionOversampling:
    def __init__(self, threshold=0.5, max_iter=1000, learning_rate=0.01, lambda_penalty=0.01):
        """
        Logistic Regression with oversampling: randomly duplicated samples from minority class
        until both have the same proportion
        threshold: threshold value to classify as class 1 (default 0.5)
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        lambda_penalty: L2 regularization lambda penalty
        """
        self.threshold = threshold
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_penalty = lambda_penalty
        self.coef_ = None
        self.intercept_ = None
    
    def _oversample(self, target_minority_class, target_majority_class):
        majority_class_idxs = np.array(self.df.index[self.df['target'] == target_majority_class])
        minority_class_idxs = np.array(self.df.index[self.df['target'] == target_minority_class])
        diff = len(majority_class_idxs) - len(minority_class_idxs)
        np.random.shuffle(minority_class_idxs)
        while diff != 0:
            random_row = self.df[self.df['target'] == target_minority_class].sample(n=1)  
            random_index = random_row.index[0]
            self.df = pd.concat([self.df, random_row])
            diff -= 1 

    def oversampling(self):

        class0 = self.df[self.df['target'] == 0]['target']
        class1 = self.df[self.df['target'] == 1]['target']
        if len(class0) < len(class1):
            self._oversample(class0, 0, 1)
        elif len(class1) < len(class0):
            self._oversample(class1, 1, 0)
        else:
            raise ValueError("Classes are balanced, oversampling is not required.")

    def _sigmoid(self, z):
        """
        Sigmoid function to transform inputs into probabilities.
        z: scalar or numpy array
        """
        return 1 / (1 + np.exp(-z))
    
    def _add_intercept(self, X):
        """
        Adds column of 1s to X for the intercept (bias) term.
        X: input feature matrix
        """
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y):
        """
        Fits the logistic regression model to the data points 
        using gradient descent.
        X: design matrix (n_samples, n_features)
        y: labels vector (n_samples,)
        """
        X = np.array(X)
        X = self._add_intercept(X)
        y = np.array(y)
        
        # Initialize the coefficients
        self.coef_ = np.zeros(X.shape[1])
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Predict probability
            z = np.dot(X, self.coef_)
            y_hat = self._sigmoid(z)
            # NLL gradient
            gradient = np.dot(X.T, (y_hat - y)) / y.size
            regularization_term = (self.lambda_penalty / y.size) * self.coef_
            regularization_term[0] = 0
            # Update coefficients
            self.coef_ -= self.learning_rate * (gradient + regularization_term)
        
        self.intercept_ = self.coef_[0] # Intercept is the fist value of coef_
        self.coef_ = self.coef_[1:]
    
    def predict_proba(self, X):
        """
        Predicts probabilities for each class for inputs X.
        X: design matrix (n_samples, n_features)
        """
        X = self._add_intercept(X)
        prob_positive = self._sigmoid(np.dot(X, np.r_[self.intercept_, self.coef_]))
        prob_negative = 1 - prob_positive
        return np.vstack((prob_negative, prob_positive)).T
    
    def predict(self, X):
        """
        Predicts class (0 or 1) for the inputs X using a threshold.
        X: design matrix (n_samples, n_features)
        """
        probas = self.predict_proba(X)
        return (probas >= self.threshold).astype(int)