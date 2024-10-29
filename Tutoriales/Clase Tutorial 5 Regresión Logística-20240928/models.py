import numpy as np

class LinearRegression:
    def __init__(self, degree=1, basis_fn=None):
        self.degree = degree
        # Default to polynomial basis if none provided
        if basis_fn is None:
            self.basis_fn = lambda x, d: np.power(x, d).flatten()  # Ensure it returns a 1D array
        else:
            self.basis_fn = basis_fn
        self.coef_ = None
        self.intercept_ = None

    def _design_matrix(self, X):
        """
        Constructs the design matrix PHI using the basis function.
        X: input feature vector (1D or 2D)
        """
        X = X.flatten()  # Ensure X is a 1D array
        PHI = np.zeros((len(X), self.degree + 1))
        for i in range(self.degree + 1):
            PHI[:, i] = self.basis_fn(X, i)  # Basis function applied on 1D array
        return PHI

    def fit(self, X, Y):
        """
        Fits the model to the data using the normal equation.
        X: input feature vector (1D)
        Y: target vector
        """
        PHI = self._design_matrix(X)
        
        # Normal equation: w = (PHI^T * PHI)^-1 * PHI^T * Y
        PHI_T_PHI = np.dot(PHI.T, PHI)
        PHI_T_Y = np.dot(PHI.T, Y)
        PHI_T_PHI_inv = np.linalg.inv(PHI_T_PHI)
        
        # Get coefficients
        w = np.dot(PHI_T_PHI_inv, PHI_T_Y)
        
        self.coef_ = w[1:]  # Coefficients for X terms
        self.intercept_ = w[0]  # Intercept term
    
    def predict(self, X):
        """
        Predicts the target values for input X using the fitted model.
        X: input feature vector (1D)
        """
        PHI = self._design_matrix(X)
        return np.dot(PHI, np.concatenate(([self.intercept_], self.coef_)))

    def __str__(self):
        """
        Return a string representation showing the intercept and coefficients.
        """
        coef_str = ' + '.join([f'{round(c,3)} * x^{i+1}' for i, c in enumerate(self.coef_)])
        return f"ŷ(x) = {round(self.intercept_,3)} + {coef_str}"
    

class LogisticRegression:
    def __init__(self, threshold=0.5, max_iter=1000, learning_rate=0.01):
        """
        threshold: threshold value to classify as class 1 (default 0.5)
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        """
        self.threshold = threshold
        self.max_iter = max_iter
        self.learning_rate = learning_rate
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
            # Update coefficients
            self.coef_ -= self.learning_rate * gradient
        
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