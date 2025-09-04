from sklearn.metrics import mean_absolute_error, mean_squared_error
from modules import models
from skopt import gp_minimize
from skopt.utils import use_named_args
import inspect

class OptFactory:
    """
    Create an abstract factory for optimization classes.
    """

    # Dictionary mapping strings to classes
    MODEL_CLASSES = {
                    'lgbm': models.LGBMTrainning,
                    'xgboost': models.XGBoostTraining,
                }

    # Dictionary mapping strings to metric functions
    METRIC_FUNCTIONS = {
                        'mae': mean_absolute_error,
                        'mse':mean_squared_error,
                    }
     
    def _select_model_class(self, model_class, X_train, X_test, y_train, y_test):
        """
        Selects and returns the correct model class instance.

        Rules:
        1. If model_class is a string, looks it up in MODEL_CLASSES and instantiates the corresponding class.
        2. If model_class is a class, instantiates it with the given args and kwargs.
        3. If model_class is already an instance, returns it directly.
        """
        # Case: string
        if isinstance(model_class, str):
            model_class = model_class.lower()
            cls = self.MODEL_CLASSES.get(model_class)
            if cls is None:
                raise ValueError(f"Model class '{model_class}' not found in MODEL_CLASSES.")
            return cls(X_train, X_test, y_train, y_test)

        # Case: class
        elif inspect.isclass(model_class):
            return model_class(X_train, X_test, y_train, y_test)

        # Case: instance
        else:
            return model_class

    def _select_metric(self, metric, y_test, y_pred):
        """
        Selects and applies the appropriate validation metric.

        Rules:
        1. If metric is a string, looks it up in METRIC_FUNCTIONS and applies the corresponding function.
        2. If metric is a callable (function), calls it directly with y_test, y_pred, and any additional arguments.
        3. Returns the computed metric value.
        """
        # Case: string
        if isinstance(metric, str):
            metric = metric.lower()
            func = self.METRIC_FUNCTIONS.get(metric)
            if func is None:
                raise ValueError(f"Metric '{metric}' not found in METRIC_FUNCTIONS.")
            return func(y_test, y_pred)

        # Case: callable
        elif callable(metric):
            return metric(y_test, y_pred)

        else:
            raise TypeError("Metric must be either a string key or a callable function.")

class BayesianOptimization(OptFactory):
    """
    Optimization class that selects and instantiates the appropriate model
    wrapper based on the provided 'model_type'.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize the Bayesian optimization with training and test data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    
    def _create_objective_func(self, model_class, space, fixed_params, metric, calibration_kwargs:dict=None):
        """
        Create the objective function for optimization.
        
        Args:
            model_class: Model class to optimize
            space: Search space for hyperparameters
            fixed_params: Fixed parameters for the model
            metric: Evaluation metric
            
        Returns:
            Objective function for gp_minimize
        """
        model_wrapper = self._select_model_class(
            model_class, 
            self.X_train, self.X_test, self.y_train, self.y_test,
        )

        @use_named_args(space)
        def objective(**params):
            all_params = {**params, **fixed_params}
            model = model_wrapper.train(params=all_params, **calibration_kwargs)
            y_pred = model_wrapper.predict(model)
            return self._select_metric(metric, self.y_test, y_pred)
        
        return objective
    
    def run(self, model_class, space, fixed_params, metric, calibration_kwargs: dict=None, **kwargs):
        """
        Run Bayesian optimization.
        
        Args:
            model_class: Model class to optimize
            space: Search space for hyperparameters
            fixed_params: Fixed parameters for the model
            metric: Evaluation metric
            **optimization_kwargs: Additional parameters for gp_minimize
            
        Returns:
            Optimization result from gp_minimize
        """
        objective_func = self._create_objective_func(
            model_class, space, fixed_params, metric, calibration_kwargs
        )
        
        return gp_minimize(
            func=objective_func,
            dimensions=space,
            **kwargs
        )

class GeneticOptimization(OptFactory):

    def __init__ (self, X_train, X_test, y_train, y_test):
        pass

    def _create_objective_func(self, model_class, space, fixed_params):
        pass

    def run(self, model_class, space, fixed_params,  **kwargs):
        pass