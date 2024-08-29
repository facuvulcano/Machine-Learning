import numpy as np
from scipy.optimize import minimize

class RidgeRegression():
    """
    Ridge Regression model for lineal regression with L2 regularizations.

    Ridge regression adresses multicollinearity by imposing a penalty on the size of coefficients.
    It minimizes the residual sum of squares plus a regularization term that penalizes large weights.
    """
    def __init__(self, lambda_penalty):
        """
        Initialize the RidgeRegression model.

        Parameters:
        -----------
        lambda_penalty : float
            Regularization strength. Must be a positive float. Larger values specify stronger regularization.
        """
        self.lambda_penalty = lambda_penalty

    def fit(self, X, y):
        """
        Fit the Ridge Regression model to the training data.

        Parameters:
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data where n_samples is the number of samples and n_features is the number of features.
        y: array-like, shape (n_samples,)
            Target values.
        
        Returns:
        --------
        self : object
            Returns an instance of self.
        
        Notes:
        ------
        This method calculates the weights (coefficients) of the linear model using the closed-form solution
        for Ridge Regression:
        
            weights = (X^T * X + lambda * I)^(-1) * X^T * y
        
        where lambda is the regularization parameter and I is the identity matriz with the first element set to zero
        to exclude the bias term from the regularization.
        """
        X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
        self.X_intercept = X_with_intercept
        dimension = X_with_intercept.shape[1]
        A = np.identity(dimension)
        A[0,0] = 0
        A_biased = self.lambda_penalty * A
        self.weights = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept) + A_biased).dot(X_with_intercept.T).dot(y)
        return self

    def predict(self, X):
        """
        Predict target values using the fitted Ridge Regression model.

        Paramaters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data where n_samples is the number of samples and n_features is the number of features
        
        Returns:
        --------
        Predictions : array, shape (n_samples,)
            Predicted target values
        
        Notes:
        ------
        This method applies the learned weights to the input data to generate predictions. It adds an intercept
        term to the input data and computes the dot product with the weight vector.
        """
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call 'fit' with appropiate arguments before using this method.")
        X_predictor = np.c_[np.ones((X.shape[0], 1)), X]
        predictions = X_predictor.dot(self.weights)
        return predictions

class NonLinearRegression():
    """
    Flexible nonlinear regression model supporting both nonlinear features and nonlinear parameters.

    This class allows fitting regression models with either polynomial feature expansion (nonlinear features)
    or using a custom nonlinear model where the parameters are nonlinear (nonlinear parameters).
    """
    def __init__(self, model_func=None, initial_params=None, degree=None, lambda_penalty=0.0, mode='features'):
        """
        Initialize the flexible nonlinear regression model.

        Parameters:
        -----------
        model_func : callable, optional
            A function representing the nonlinear model, used only in 'parameters' mode.

        initial_params : array-like, optional
            Initial guess for the parameters, used only in 'parameters' mode.
        
        degree : int, optional
            Degree of polynomical features, used only in 'features' mode.
        
        lambda_penalty : float, optional, default=0.0
            Regularization strength. Must be a non-negative float. Larger values specify stronger regularization.
        
        mode : str, optional, default='features'
            Specifies the type of nonlinearity. Optons are:
            - 'features': Applies polynomial feature expansion (nonlinear features).
            - 'parameters': Fits a custom nonlinear model where the parameters are nonlinear.
        """
        self.model_func = model_func
        self.initial_params = np.array(initial_params) if initial_params is not None else None
        self.degree = degree
        self.lambda_penalty = lambda_penalty
        self.mode = mode
        self.params_ = None
        self.weights_ = None
    
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
        for d in range(2, self.degree + 1):
            X_poly = np.c_[X_poly, X**d]
        return X_poly
    
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
        
        Notes:
        ------
        This function is used in 'parameters' mode to evaluate the performance of the current set of parameters.
        """
        predictions = self.model_func(X, *params)
        loss = np.mean((y - predictions) ** 2)
        return loss
    
    def fit(self, X, y):
        """
        Fit the model using the chosen mode.

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
        
        Notes:
        ------
        Depending on the mode:
        - 'features': Fits a Ridge Regression model with polynomial features.
        - 'parameters': Uses optimization to fit a custom nonlinear model.
        """
        if self.mode == 'features':
            X_poly = self._add_polynomial_features(X)
            X_with_intercept = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
            dimension = X_with_intercept.shape[1]
            A = np.identity(dimension)
            A[0, 0] = 0
            A_biased = self.lambda_penalty * A
            self.weights = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept) + A_biased).dot(X_with_intercept.T).dot(y)
        
        elif self.mode == 'parameters':
            if self.model_func is None or self.initial_params is None:
                raise ValueError("model_func and initial_params must be provided for 'parameters' mode.")
            
            result = minimize(self._loss_function, self.initial_params, args=(X, y), method='L-BFGS-B')
            self.params_ = result.x
        
        else:
            raise ValueError("Invalid mode. Choose 'features' or 'parameters'.")
        
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
        
        Notes:
        ------
        Depending on the mode:
        - 'features': Uses the polynomial features and the fitted weights to predict.
        - 'parametes': Uses the fitted custom nonlinear model to predict.
        """
        if self.mode == 'features':
            if self.weights_ is None:
                raise ValueError("Model is not fitted yet. Call 'fit' with appropiate arguments before using this method.")
            
            X_poly = self._add_polynomial_features(X)
            X_predictor = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]
            predictions = X_predictor.dot(self.weights_)
        
        elif self.mode == 'parameters':
            if self.params_ is None:
                raise ValueError("Model is not fitted yet. Call 'fit' with appropiate arguments before using this method.")
            
            predictions = self.model_func(X, *self.params_)
        
        else:
            raise ValueError("Invalid mode. Choose 'features' or 'parameters'.")
        
        return predictions
    

class LocallyWeightedRegression():
    def __init__(self, features, targets, tau=1.0):
        """
        Initialize the locally weighted regression model with features, targets, and bandwith.
        :param features: list or np.array, independent variables (should include a constant term for intercept if needed)
        :param targets: list or np.array, dependent variable
        :param tau: float, bendwith parameter determiningthe weighting shceme, lower values gives more weight to points near the target point.
        """
        self.features = np.array(features)
        self.targets = np.array(targets)
        self.tau = tau

    def precit(self, query_points):
        """
        Predict the target value for a given query point using locally weighted regreassion.
        :param query_point: np.array, the input features at which the prediction is to be made
        :return: float, predicted value
        """

        weights = np.exp(-np.sum((self.features - query_points)**2, axis=1) / (2 * self.tau**2))
        W = np.diag(weights)
        X_transpose_W = self.features.T.dot(W)
        X_transpose_W_X = X_transpose_W.dot(self.features)

        if np.linalg.det(X_transpose_W_X) == 0:
            beta = np.linalg.pinv(X_transpose_W_X).dot(X_transpose_W).dot(self.targets)
        else:
            beta = np.linalg.inv(X_transpose_W_X).dot(X_transpose_W).dot(self.targets)

        return query_points.dot(beta)
        


# df_procesado = pd.read_csv('C:\\Users\\facuv\\Machine-Learning\\Vulcano_Facundo_TP2\\data\\processed\\dataset_procesado.csv')

# targets = df_procesado['Precio'].to_numpy()

# df_procesado = df_procesado.drop(columns=['Precio'])
# features = df_procesado.to_numpy()



# regresion = RidgeRegression(features, targets)
# weights = regresion.optimal_weights()
# target_prediction = regresion.target_prediciton(weights)
# print(target_prediction)
# print(targets)


#modelo lineal en los parametros busco el w* con la formula
#modelo lineal en los features busco el w* con algun metodo de optimizacion
