import numpy as np


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
