
import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd


class LDA():
    """
    Linear Discriminant Analysis (LDA) for dimensionality reduction.
    This class imlements LDA, which aims to find a linear combination of features that characterizes
    or separates two or more classes.
    It is used for both classification adn dimensionality reduction.
    """ 
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.means_ = None
        self.global_mean_ = None
        self.S_W_ = None
        self.S_B_ = None
        self.eig_pairs_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.projection_matrix_ = None

    def _compute_class_means(self):
        features = self.df.columns.drop('Diabetes')
        class_means = {}
        for class_label in self.df['Diabetes'].unique():
            class_data = self.df[self.df['Diabetes'] == class_label][features]
            class_means[class_label] = class_data.mean().values

        self.class_means_ = class_means
        return class_means

    def compute_global_means(self):
        features = self.df.columns.drop('Diabetes')
        global_mean = self.df[features].mean().values
        self.global_mean_ = global_mean
        return global_mean


    def compute_within_class_scatter(self):
        features = self.df.columns.drop('Diabetes')
        class_means = self._compute_class_means()

        S_W = np.zeros((len(features), len(features)))

        for class_label, mean_vector in class_means.items():
            class_scatter = np.zeros((len(features), len(features)))
            class_data = self.df[self.df['Diabetes'] == class_label][features]
            mean_vector  = mean_vector.reshape(-1, 1)
            for _, row in class_data.iterrows():
                row_vector = row.values.reshape(-1, 1)
                difference = row_vector - mean_vector
                class_scatter += difference.dot(difference.T)
            S_W += class_scatter
        
        S_W += np.eye(S_W.shape[0]) * 1e-6
        self.S_W_ = S_W
        return S_W
    
    def compute_between_class_scatter(self):
        features = self.df.columns.drop('Diabetes')
        class_means = self._compute_class_means()
        global_mean = self.compute_global_means().reshape(-1, 1)

        S_B = np.zeros((len(features), len(features)))

        for class_label, mean_vector in class_means.items():
            mean_vector = mean_vector.reshape(-1, 1)
            N_k = self.df[self.df['Diabetes'] == class_label].shape[0]
            mean_diff = mean_vector - global_mean
            S_B += N_k * mean_diff.dot(mean_diff.T)

        self.S_B_ = S_B
        return S_B

    def compute_eigen_pairs(self):
        if self.S_W_ is None or self.S_B_ is None:
            raise ValueError("Las matrices S_W y S_B deben calcularse antes de obtener los autovalores y autovectores")

        eig_values, eig_vectors = np.linalg.eig(np.linalg.inv(self.S_W_).dot(self.S_B_))

        eig_values = np.real(eig_values)
        eig_vectors = np.real(eig_vectors)
    
        eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i]) for i in range(len(eig_values))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

        self.eig_pairs_ = eig_pairs
        return eig_pairs

    def select_top_eigenvectors(self, n_components):
        if not hasattr(self, 'eig_pairs_'):
            raise ValueError("Los pares de autovalores y autovectores deben calcularse antes de seleccionar los princiaples.")
        
        top_eigenvectors = [self.eig_pairs_[i][1] for i in range(n_components)]
        self.projection_matrix_ = np.stack(top_eigenvectors, axis=1)
        return self.projection_matrix_

    def transform(self, X):
        if not hasattr(self, 'projection_matrix_'):
            raise ValueError("La matriz de proyeccion debe ser calculada antes de transformar los datos")
        
        X_transformed = X.dot(self.projection_matrix_)
        return X_transformed

    def fit(self, X, y, n_components):
        self.df = X.copy()
        self.df['Diabetes'] = y

        self.compute_within_class_scatter()
        self.compute_between_class_scatter()
        self.compute_eigen_pairs()
        self.select_top_eigenvectors(n_components)

    def predict_proba(self, X):
        X_transformed = self.transform(X)
        probabilities = []
        for row in X_transformed:
            distances = []
            for class_label, mean_vector in self.class_means_.items():
                mean_vector_projected = mean_vector.reshape(1, -1).dot(self.projection_matrix_)
                distance = np.linalg.norm(row - mean_vector_projected)
                distances.append(distance)
            
            total_distance = sum(distances)
            probabilities.append([1 - (dist / total_distance) for dist in distances])
        
        return np.array(probabilities)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions


class LogisticRegressionMulticlass:
    """
    Multiclass Logistic Regression with gradient descent optimization.

    This class implements a multiclass logistic regression model using softmax for classification
    and gradient descent for optimization
    """
    def __init__(self, threshold=0.5, n_iter=10000):
        """
        Initialize the LogisticRegressionMulticlass model.

        Parameters:
        -----------
        threshold : float, optional (default=0.5)
            threshold value for classifying a sample.
        
        n_iter : int, optional (default=10000)
            Maximum number of iterations for gradient descent.
        """
        self.threshold = threshold
        self.n_iter = n_iter
    
    def _add_intercept(self, X):
        """

        """
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y, batch_size=64, lr=0.001, verbose=False):
        """
        Fit the logistic regression model to the data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target labels.

        batch_size : int, optional (default=64)
            Number of samples per gradient update.

        lr : float, optional (default=0.001)
            Learning rate for gradient descent.

        verbose : bool, optional (default=False)
            Whether to print progress during training.

        Returns:
        --------
        self : object
            Returns an instance of self.
        """
        self.classes = np.unique(y)
        self.class_labels = {c : i for i, c in enumerate(self.classes)}
        X = self._add_intercept(X)
        y = self.one_hot(y)
        self.loss = []
        self.coef_ = np.random.randn(len(self.classes), X.shape[1]) * 0.01
        self.fit_data(X, y, batch_size, lr, verbose)
        return self
    
    def fit_data(self, X, y, batch_size, lr, verbose):
        """
        Helper function to fit the data using gradient descent.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data with intercept term.

        y : array-like, shape (n_samples, n_classes)
            One-hot encoded target labels.

        batch_size : int
            Number of samples per gradient update.

        lr : float
            Learning rate for gradient descent.

        verbose : bool
            Whether to print progress during training.
        """
        previous_loss = np.inf
        for i in tqdm(range(self.n_iter), desc="Outer Iteration", unit="iteration"):
            loss_value = self.cross_entropy(y, self.predict_(X))
            self.loss.append(loss_value)
            if verbose and i % 1000 == 0:
                print(f'Iteration {i}: Loss = {loss_value}, Accuracy = {self.evaluate_(X, y)}')
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.predict_(X_batch)
            update = (lr * np.dot(error.T, X_batch))
            self.coef_ += update
            
            if np.linalg.norm(update) < self.threshold or np.abs(previous_loss - loss_value) < 1e-5:
                print(f'Convergence reached at iteration {i}')
                break
            
            previous_loss = loss_value

    def predict(self, X):
        """
        Predict probabilities for the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        predictions : ndarray
            Predicted probabilities for each class.
        """
        X = self._add_intercept(X)
        return self.predict_(X)    
    
    def predict_(self, X):
        """
        Internal method to compute predictions using softmax.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        probs : ndarray
            Probability distribution over classes for each sample.
        """
        pre_vals = np.dot(X, self.coef_.T).reshape(-1,len(self.classes))
        return self.softmax(pre_vals)

    def softmax(self, z):
        """
        Apply the softmax function to the input data.

        Parameters:
        -----------
        z : array-like, shape (n_samples, n_classes)
            Input data.

        Returns:
        --------
        softmaxed : ndarray
            The softmax-transformed probabilities.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def get_random_weights(self, row, col):
        """
        Generate random weights for initialization.

        Parameters:
        -----------
        row : int
            Number of rows for the weight matrix.

        col : int
            Number of columns for the weight matrix.

        Returns:
        --------
        weights : ndarray
            The generated random weights.
        """
        return np.zeros(shape=(row, col))

    def predict_classes(self, X):
        """
        Predict class labels for the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        predicted_classes : ndarray
            Predicted class labels.

        probs : ndarray
            Predicted probabilities for each class.
        """
        self.probs_ = self.predict(X)
        predicted_classes = np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
        return predicted_classes, self.probs_
    
    def score(self, X, y):
        return np.mean(self.predict_classes(X) == y)
    
    def evaluate_(self, X, y):
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))
    
    def cross_entropy(self, y, probs, epsilon=1e-12):
        """
        Compute the cross-entropy loss.

        Parameters:
        -----------
        y : array-like, shape (n_samples, n_classes)
            True labels in one-hot encoded format.

        probs : array-like, shape (n_samples, n_classes)
            Predicted probabilities for each class.

        epsilon : float, optional (default=1e-12)
            Small value to prevent division by zero in log function.

        Returns:
        --------
        loss : float
            The cross-entropy loss.
        """
        probs = np.clip(probs, epsilon, 1. - epsilon)
        return -np.mean(y * np.log(probs))
    
    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

class Node:
    """
    Node class for Decision Tree.

    This class represents a single node in a decision tree, which can either be a leaf node with a value or an internal node with a feature and threshold for splitting.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None) -> None:
        """
        Initialize a node.

        Parameters:
        -----------
        feature : int, optional
            The index of the feature used for splitting.

        threshold : float, optional
            The threshold value for splitting.

        left : Node, optional
            The left child node.

        right : Node, optional
            The right child node.

        value : int or float, optional
            The value for a leaf node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.

        Returns:
        --------
        is_leaf : bool
            True if the node is a leaf node, False otherwise.
        """
        return self.value is not None

class DecisionTree:
    """
    Decision Tree classifier.

    This class implements a decision tree classifier for binary or multiclass classification tasks.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None) -> None:
        """
        Initialize the Decision Tree classifier.

        Parameters:
        -----------
        min_samples_split : int, optional (default=2)
            The minimum number of samples required to split an internal node.

        max_depth : int, optional (default=100)
            The maximum depth of the tree.

        n_features : int, optional
            The number of features to consider when looking for the best split.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Fit the decision tree to the data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target labels.
        """
        self.n_features = X.shape[1] if not self.n_features else min()(X.shape[1], self.n_features)
        self.y_train = y
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the decision tree.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            Target labels.

        depth : int, optional (default=0)
            The current depth of the tree.

        Returns:
        --------
        node : Node
            The root node of the grown subtree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if n_samples == 0:
            return None
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        features_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_threshold = self._best_split(X, y, features_idxs)

        if best_feature is None or best_threshold is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feature_idxs):
        """
        Find the best feature and threshold for splitting the data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            Target labels.

        feature_idxs : array-like, shape (n_features,)
            The indices of the features to consider for splitting.

        Returns:
        --------
        split_idx : int
            The index of the best feature to split on.

        split_threshold : float
            The threshold value for the best split.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = thr

        if split_idx is None:
            return None, None
        
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """
        Compute the information gain from a potential split.

        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            Target labels.

        X_column : array-like, shape (n_samples,)
            The feature column to split on.

        threshold : float
            The threshold value for the split.

        Returns:
        --------
        gain : float
            The information gain from the split.
        """
        parent_entropy = self._entropy(y)
        
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    
    def _split(self, X_column, split_thresh):
        """
        Split the data into left and right subsets based on a threshold.

        Parameters:
        -----------
        X_column : array-like, shape (n_samples,)
            The feature column to split on.

        split_thresh : float
            The threshold value for the split.

        Returns:
        --------
        left_idxs : ndarray
            The indices of the samples that go to the left child.

        right_idxs : ndarray
            The indices of the samples that go to the right child.
        """
        left_idxs = np.argwhere(X_column<=split_thresh).flatten()
        right_idxs = np.argwhere(X_column>split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Compute the entropy of a set of labels.

        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            Target labels.

        Returns:
        --------
        entropy : float
            The entropy of the labels.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        Find the most common label in a set of labels.

        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            Target labels.

        Returns:
        --------
        most_common : int
            The most common label.
        """
        if len(y) == 0:
            return None
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        predictions : ndarray
            Predicted class labels for each sample.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.

        Parameters:
        -----------
        x : array-like, shape (n_features,)
            A single sample.

        node : Node
            The current node in the tree.

        Returns:
        --------
        prediction : int
            The predicted class label.
        """
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        probabilities : ndarray
            Predicted probabilities for each class.
        """
        probas = []
        for x in X:
            node = self.root
            while not node.is_leaf_node():
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            probas.append(self._get_leaf_proba(node))
        return np.array(probas)
    
    def _get_leaf_proba(self, node):
        """
        Compute the class probabilities for a leaf node.

        Parameters:
        -----------
        node : Node
            The leaf node.

        Returns:
        --------
        probas : ndarray
            The class probabilities for the leaf node.
        """
        if node.is_leaf_node():
            counts = np.bincount(self.y_train[self._get_node_indices(node)])
            probas = counts / counts.sum()
            return probas
        return np.zeros(self.n_classes)
    
    def _get_node_indices(self, node):
        """
        Get the indices of samples that reach a specific node.

        Parameters:
        -----------
        node : Node
            The node to find the sample indices for.

        Returns:
        --------
        indices : ndarray
            The indices of samples that reach the node.
        """
        if node.is_leaf_node():
            return np.arange(len(self.y_train))
        left_indices = self._get_node_indices(node.left)
        right_indices = self._get_node_indices(node.right)
        return np.concatenate([left_indices, right_indices])
    

class RandomForest():
    """
    Random Forest classifier.

    This class implements a random forest classifier using an ensemble of decision trees.
    """
    def __init__(self, n_trees=12, max_depth=5, min_samples_split=2, n_features=None) -> None:
        """
        Initialize the Random Forest classifier.

        Parameters:
        -----------
        n_trees : int, optional (default=12)
            The number of trees in the forest.

        max_depth : int, optional (default=5)
            The maximum depth of each tree.

        min_samples_split : int, optional (default=2)
            The minimum number of samples required to split an internal node.

        n_features : int, optional
            The number of features to consider when looking for the best split in each tree.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def fit(self, X, y):
        """
        Fit the random forest model to the data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target labels.
        """
        self.trees = []
        for _ in tqdm(range(self.n_trees), unit='tree'):
            tree = DecisionTree(max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
    def _bootstrap_samples(self, X, y):
        """
        Generate a bootstrap sample of the dataset.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            Target labels.

        Returns:
        --------
        X_sample : ndarray
            The bootstrap sample of the input data.

        y_sample : ndarray
            The bootstrap sample of the target labels.
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        """
        Find the most common label in a set of labels.

        Parameters:
        -----------
        y : array-like, shape (n_samples,)
            Target labels.

        Returns:
        --------
        most_common : int
            The most common label.
        """
        if len(y) == 0:
            return None
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        predictions : ndarray
            Predicted class labels for each sample.
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if predictions.ndim == 1:
            return predictions

        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        probabilities : ndarray
            Predicted probabilities for each class.
        """
        tree_probas =  np.array([tree.predict_proba(X) for tree in self.trees])
        avg_probas = np.mean(tree_probas, axis=0)
        return avg_probas
    
class DataSampler:
    def __init__(self, df, target_column='Diabetes'):
        self.df = df
        self.target_column = target_column

    def _oversample(self, minority_class_size, target_minority_class):
        num_samples_needed = minority_class_size
        minority_class_samples = self.df[self.df[self.target_column] == target_minority_class].sample(
            n=num_samples_needed, replace=True, random_state=42
        )
        self.df = pd.concat([self.df, minority_class_samples], axis=0)
        return self.df

    def oversampling(self):
        class_counts = self.df[self.target_column].value_counts()
        max_class_size = class_counts.max()
        for target_class in class_counts.index:
            if class_counts[target_class] < max_class_size:
                self.df = self._oversample(max_class_size - class_counts[target_class], target_class)

        return self.df