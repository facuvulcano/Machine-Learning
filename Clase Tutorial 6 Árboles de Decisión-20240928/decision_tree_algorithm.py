import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import patches

class Node():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.feature_importance = self.data.shape[0] * self.information_gain
        self.left = None
        self.right = None

class DecisionTree():
    def __init__(self, 
                 max_depth=4, 
                 min_samples_leaf=1, 
                 min_information_gain=0.0) -> None:
        """
        Constructor function for DecisionTree instance
        Inputs:
            max_depth (int): max depth of the tree
            min_samples_leaf (int): min number of samples required to be in a leaf 
                                    to make the splitting possible
            min_information_gain (float): min information gain required to make the 
                                          splitting possible                              
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain

    def entropy(self, class_probabilities: list) -> float: #esto podria ser indice de gini o chi cuadrado tambien
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def data_entropy(self, labels: list) -> float:
        return self.entropy(self.class_probabilities(labels))
    
    def partition_entropy(self, subsets: list) -> float:
        """
        Calculates the entropy of a partitioned dataset. 
        Inputs:
            - subsets (list): list of label lists 
            (Example: [[1,0,0], [1,1,1] represents two subsets 
            with labels [1,0,0] and [1,1,1] respectively.)

        Returns:
            - Entropy of the labels
        """
        # Total count of all labels across all subsets.
        total_count = sum([len(subset) for subset in subsets]) 
        # Calculates entropy of each subset and weights it by its proportion in the total dataset 
        return sum([self.data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    
    def split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        """
        Partitions the dataset into two groups based on a specified feature 
        and its corresponding threshold value.
        Inputs:
        - data (np.array): training dataset
        - feature_idx (int): feature used to split
        - feature_val (float): threshold value 
        """
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2
        
    def find_best_split(self, data: np.array) -> tuple:
        """
        Finds the optimal feature and value to split the dataset on 
        at each node of the tree (with the lowest entropy).
        Inputs:
            - data (np.array): numpy array with training data
        Returns:
            - 2 splitted groups (g1_min, g2_min) and split information 
            (min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy)
        """
        min_part_entropy = 1e9
        feature_idx =  list(range(data.shape[1]-1))

        for idx in feature_idx: # For each feature
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25)) # Calc 25th, 50th, and 75th percentiles
            for feature_val in feature_vals: # For each percentile value we partition in 2 groups
                g1, g2, = self.split(data, idx, feature_val)
                part_entropy = self.partition_entropy([g1[:, -1], g2[:, -1]]) # Calculate entropy of that partition
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2

        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def find_label_probs(self, data: np.array) -> np.array:
        """
        Computes the distribution of labels in the dataset.
        It returns the array label_probabilities, which contains 
        the probabilities of each label occurring in the dataset.

        Inputs:
            - data (np.array): numpy array with training data
        Returns:
            - label_probabilities (np.array): numpy array with the
            probabilities of each label in the dataset.
        """
        # Transform labels to ints (assume label in last column of data)
        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)
        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def create_tree(self, data: np.array, current_depth: int) -> Node:
        """
        Recursive, depth first tree creation algorithm.
        Inputs:
            - data (np.array): numpy array with training data
            - current_depth (int): current depth of the recursive tree
        Returns:
            - node (Node): current node, which contains references to its left and right child nodes.
        """
        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        # Find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self.find_best_split(data)
        # Find label probs for the node
        label_probabilities = self.find_label_probs(data)
        # Calculate information gain
        node_entropy = self.entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        # Create node
        node = Node(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)
        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node
        
        current_depth += 1
        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)
        
        return node
    
    def predict_one_sample(self, X: np.array) -> np.array:
        """
        Returns prediction for 1 dim array.
        """
        node = self.tree
        # Finds the leaf which X belongs to
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the model with given X and Y datasets.
        Inputs:
            - X_train (np.array): training features
            - Y_train (np.array): training labels
        """
        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)
        # Create tree
        self.tree = self.create_tree(data=train_data, current_depth=0)

    def predict_proba(self, X_set: np.array) -> np.array:
        """
        Returns the predicted probs for a given data set
        """
        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """
        Returns the predicted labels for a given data set
        """
        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds   


    # new
    def plot_decision_tree(self, feature_names, fig_size=(12, 6)):
        """
        Plots the decision tree.
        Inputs:
            - feature_names (list): list of feature names.
            - fig_size (tuple): size of the plot (default is (12, 6)).
        """
        def plot_node(node, depth, pos):
            if node is None:
                return
            
            # Plot the current node
            plt.text(pos[0], pos[1], f"{feature_names[node.feature_idx]}\n<{node.feature_val}\nIG: {node.information_gain:.4f}",
                    ha="center", va="center", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

            # Calculate positions of children nodes
            shift = 1.0 / (2**depth)
            if node.left:
                plt.plot([pos[0], pos[0] - shift], [pos[1], pos[1] - 1], 'k-')
                plot_node(node.left, depth + 1, (pos[0] - shift, pos[1] - 1))
            if node.right:
                plt.plot([pos[0], pos[0] + shift], [pos[1], pos[1] - 1], 'k-')
                plot_node(node.right, depth + 1, (pos[0] + shift, pos[1] - 1))

        # Create plot
        plt.figure(figsize=fig_size)
        plt.title("Decision Tree")

        # Plot the tree starting from the root
        plot_node(self.tree, depth=1, pos=(0.5, 1.0))

        plt.axis('off')
        plt.show() 
        return
    
    def print_tree(self, node=None, depth=0):
        """
        Prints the decision tree.
        """
        if node is None:
            node = self.tree

        # Print node information
        if node.feature_idx is not None:
            print(f"{'|   ' * depth}Node: {depth}, Feature: {node.feature_idx}, Threshold: {node.feature_val:.4f}, Information Gain: {node.information_gain:.4f}")
        else:
            print(f"{'|   ' * depth}Leaf: {node.prediction_probs}")

        # Recursively print left and right subtrees
        if node.left:
            self.print_tree(node.left, depth + 1)
        if node.right:
            self.print_tree(node.right, depth + 1)
        return
    
    def plot_feature_importance(self, feature_names):
        """
        Plots the feature importance based on the information gain across the nodes.
        Inputs:
            - feature_names (list): list of feature names.
        """
        def compute_feature_importance(node, importance_dict):
            if node is None:
                return
            
            # Aggregate feature importance
            if node.feature_idx is not None:
                if node.feature_idx in importance_dict:
                    importance_dict[node.feature_idx] += node.feature_importance
                else:
                    importance_dict[node.feature_idx] = node.feature_importance
            
            # Recursively compute for left and right children
            compute_feature_importance(node.left, importance_dict)
            compute_feature_importance(node.right, importance_dict)
        
        importance_dict = {}
        compute_feature_importance(self.tree, importance_dict)

        # Normalize importance
        total_importance = sum(importance_dict.values())
        feature_importances = {feature_names[idx]: importance / total_importance for idx, importance in importance_dict.items()}

        # Sort and plot
        sorted_importance = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
        labels, values = zip(*sorted_importance)
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()
        return


