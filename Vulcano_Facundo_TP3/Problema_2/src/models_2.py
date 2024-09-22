
import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd


class LDA():
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
    def __init__(self, threshold=0.5, n_iter=10000):
        """
        Logistic Regression without re balancing technique
        threshold: threshold value to classify as class 1 (default 0.5)
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        lambda_penalty: L2 regularization lambda penalty
        """
        self.threshold = threshold
        self.n_iter = n_iter
    
    def _add_intercept(self, X):
        """
        Adds column of 1s to X for the intercept (bias) term.
        X: input feature matrix
        """
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X, y, batch_size=64, lr=0.001, verbose=False):
        """
        Fits the logistic regression model to the data points 
        using gradient descent.
        X: design matrix (n_samples, n_features)
        y: labels vector (n_samples,)
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
            
            if np.abs(update).max() < self.threshold or np.abs(previous_loss - loss_value) < 1e-5:
                print(f'Convergence reached at iteration {i}')
                break
            
            previous_loss = loss_value

    def predict(self, X):
        return self.predict_(self._add_intercept(X))    
    
    def predict_(self, X):
        pre_vals = np.dot(X, self.coef_.T).reshape(-1,len(self.classes))
        return self.softmax(pre_vals)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(np.exp(z), axis=1).reshape(-1,1)
    
    def get_random_weights(self, row, col):
        return np.zeros(shape=(row, col))

    def predict_classes(self, X):
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
    
    def score(self, X, y):
        return np.mean(self.predict_classes(X) == y)
    
    def evaluate_(self, X, y):
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))
    
    def cross_entropy(self, y, probs):
        return -1 * np.mean(y * np.log(probs))
    
    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min()(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if n_samples == 0:
            return None

        # check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        

        
        features_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # find the best split
        best_feature, best_threshold = self._best_split(X, y, features_idxs)

        if best_feature is None or best_threshold is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # create child nodes
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feature_idxs):
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
        #parent entropy
        parent_entropy = self._entropy(y)
        
        #create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        #calculate the weighted entropy avf of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        # calculcate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column<=split_thresh).flatten()
        right_idxs = np.argwhere(X_column>split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        if len(y) == 0:
            return None
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest():
    def __init__(self, n_trees=6, max_depth=4, min_samples_split=2, n_features=None) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        if len(y) == 0:
            return None
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if predictions.ndim == 1:
            return predictions

        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
