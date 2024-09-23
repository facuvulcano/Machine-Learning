import numpy as np
from scipy.optimize import minimize

class RidgeRegression():
    """
    Ridge Regression model for non lineal regression with L2 regularizations.

    Ridge regression adresses multicollinearity by imposing a penalty on the size of coefficients.
    It minimizes the residual sum of squares plus a regularization term that penalizes large weights.
    This is a non linear regression on the features because of the add_polynomia_features function
    """
    def __init__(self, lambda_penalty, degree):
        """
        Initialize the RidgeRegression model.

        Parameters:
        -----------
        lambda_penalty : float
            Regularization strength. Must be a positive float. Larger values specify stronger regularization.
        """
        self.lambda_penalty = lambda_penalty
        self.degree = degree
        self.weights = None

    def _add_polynomial_features(self, X):
        """
        Generate polynomial features up to the specified degree.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data where n_samples is the number of samples and n_features is the number of features.
        
        Returns:
        --------
        X_poly : array-like, sahpe (n_samples, n_poly_features)
            The input data transformed to include polynomial features up to the specified degree.
        """
        X_poly = X.copy()
        n_samples, n_features = X.shape
        for d in range(2, self.degree + 1):
            for i in range(n_features):
                X_poly = np.c_[X_poly, X[:, i] ** d]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                X_poly = np.c_[X_poly, X[:, i] * X[:, j]]
        
        return X_poly

    def fit(self, X, y):
        """
        Fit the model

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.
        
        Returns:
        -------
        self : object
            Returns an instance of self.
        """

        num_features = X.iloc[:, :2].to_numpy()
        bin_features = X.iloc[:, 2:].to_numpy()
        num_features_poly = self._add_polynomial_features(num_features)
        X_poly = np.hstack((num_features_poly, bin_features))
        X_with_intercept = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        dimension = X_with_intercept.shape[1]
        A = np.identity(dimension)
        A[0, 0] = 0
        A_biased = self.lambda_penalty * A
        self.weights = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept) + A_biased).dot(X_with_intercept.T).dot(y)
        return self

    def predict(self, X):
        """
        Predict ussing the fitted model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data where n_samples is the number of samples and n_featues is the number of features.
        
        Returns:
        --------
        Predicitons : array, shape (n_samples,)
            Predicted target values.
        
        """
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call 'fit' with appropiate arguments before using this method.")
        
        num_features = X.iloc[:, :2].to_numpy()
        bin_features = X.iloc[:, 2:].to_numpy()
        num_features_poly = self._add_polynomial_features(num_features)
        X_poly = np.hstack((num_features_poly, bin_features))
        X_predictor = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
        predictions = X_predictor.dot(self.weights)
        return predictions

class NonLinearRegression():
    """
    Nonlinear regression model on the parameters.

    This class allows fitting regression models with a custom nonlinear model where the parameters are nonlinear (nonlinear parameters).
    """
    def __init__(self, model_func, initial_params, lambda_penalty):
        """
        Initialize the nonlinear regression model.

        Parameters:
        -----------
        model_func : callable
            A function representing the nonlinear model

        initial_params : array-like, optional
            Initial guess for the parameters
        
        lambda_penalty : float, optional, default=0.0
            Regularization strength. Must be a non-negative float. Larger values specify stronger regularization.
        
        """
        self.model_func = model_func
        self.initial_params = np.array(initial_params) if initial_params is not None else None
        self.lambda_penalty = lambda_penalty
        self.params = None

    def _loss_function(self, params, X, y):
        """
        Compute the loss function (mean squared error) for nonlinear parameters.

        Parameters:
        -----------
        params : array-like, shape (n_params,)
            Current guess of the model parameters.
        
        X : array-like, shape (n_samples, n_features)
            Input data where n_samples is the number of samples and n_features is the number of features.
        
        y : array-like, sahpe (n_samples,)
            Target values.
        
        Returns:
        --------
        Loss : float
            The mean squared error between the predicted values and the target values.
        
        """
        predictions = self.model_func(X, *params)
        loss = np.mean((y - predictions) ** 2)
        return loss
    
        
    def fit(self, X, y):
        """
        Fit the model

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.
        
        Returns:
        -------
        self : object
            Returns an instance of self.
        
        """
        
        result = minimize(self._loss_function, self.initial_params, args=(X, y), method='L-BFGS-B') #options={'disp':True}
        self.params = result.x
    
        return self

    def predict(self, X):
        """
        Predict ussing the fitted model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data where n_samples is the number of samples and n_featues is the number of features.
        
        Returns:
        --------
        Predicitons : array, shape (n_samples,)
            Predicted target values.
        
        """

    

        if self.params is None:
            raise ValueError("Model is not fitted yet. Call 'fit' with appropiate arguments before using this method.")      
        predictions = self.model_func(X, *self.params)
        return predictions
    

class LocallyWeightedRegression():
    def __init__(self, tau):
        """
        Initialize the locally weighted regression model with features, targets, and bandwith.
        :param features: list or np.array, independent variables (should include a constant term for intercept if needed)
        :param targets: list or np.array, dependent variable
        :param tau: float, bendwith parameter determiningthe weighting shceme, lower values gives more weight to points near the target point.
        """
        self.tau = tau
        self.X_train = None
        self.y_train = None

    def _compute_weights(self, x_query):
        
        distances = np.linalg.norm(x_query - self.X_train, axis=1)
        weights = np.exp(-distances ** 2 / (2 * self.tau ** 2))
        W = np.diag(weights)
        return W

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_query):
        """
        Predict the target value for a given query point using locally weighted regreassion.
        :param query_point: np.array, the input features at which the prediction is to be made
        :return: float, predicted value
        """

        y_pred = []
        for x_query in X_query:
            W = self._compute_weights(x_query)
            XTWX = self.X_train.T.dot(W).dot(self.X_train)
            XTWy = self.X_train.T.dot(W).dot(self.y_train)
            theta = np.linalg.pinv(XTWX).dot(XTWy)
            y_pred.append(x_query.dot(theta))

        return np.array(y_pred)
