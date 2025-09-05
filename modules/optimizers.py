from sklearn.metrics import mean_absolute_error, mean_squared_error
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Dimension
from scipy.optimize import OptimizeResult
from scipy.optimize import OptimizeResult
from typing import Any, Callable, Union, List

class BayesianOptimization:
    """
    Performs Bayesian optimization for a given model instance.
    
    This class finds the best set of hyperparameters for a model object
    that knows how to train itself and make predictions. The process is separated
    into two main steps:
    1. `run()`: Executes the optimization search.
    2. `fit_best_model()`: Trains the final model using the best parameters found.
    """

    def __init__(self, model_instance):
        """
        Initializes the optimizer.

        Args:
            model_instance: An object that conforms to the required interface,
                            possessing .y_test, .train(), and .predict() methods.
        """
        self.model_instance = model_instance
        self.y_test = self.model_instance.y_test
        
        # Attributes to store optimization results and configuration
        self.optimizer_result_: OptimizeResult = None
        self.best_params_: dict = None
        self.best_score_: float = None
        self.best_model_ = None
        
        # Storing params used during the run for the final fit
        self._fixed_params: dict = None
        self._calibration_kwargs: dict = None


    def _create_objective_func(
        self,
        space: List[Dimension],
        fixed_params: dict[str, Any],
        metric: Callable[..., float],
        calibration_kwargs: Union[dict[str, Any], None] = None,
    ) -> Callable[..., float]:
        """Creates the objective function that will be minimized."""

        @use_named_args(space)
        def objective(**params: Any) -> float:
            """The internal function that the optimizer will call."""
            all_params = {**params, **fixed_params}
            model = self.model_instance.train(params=all_params, **(calibration_kwargs or {}))
            y_pred = self.model_instance.predict(model)
            score = metric(self.y_test, y_pred)
            return score

        return objective

    def run(
        self,
        space: List[Dimension],
        fixed_params: dict[str, Any],
        metric: Callable[..., float],
        calibration_kwargs: Union[dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> OptimizeResult:
        """
        Runs the Bayesian optimization search to find the best hyperparameters.

        This method only performs the search and stores the results. It does not
        train the final model.
        """
        # Store configuration for the final model training
        self._fixed_params = fixed_params
        self._calibration_kwargs = calibration_kwargs or {}

        objective_func = self._create_objective_func(
            space, self._fixed_params, metric, self._calibration_kwargs
        )

        result = gp_minimize(
            func=objective_func,
            dimensions=space,
            **kwargs,
        )

        self.optimizer_result_ = result
        self.best_score_ = result.fun
        self.best_params_ = {dim.name: val for dim, val in zip(space, result.x)}

        print("Optimization finished.")
        print(f"Best score ({metric.__name__}): {self.best_score_:.4f}")
        print(f"Best parameters: {self.best_params_}")
        print("\nTo train the final model with these parameters, call the '.fit_best_model()' method.")

        return result

    def get_best_model(self):
        """
        Trains and stores the final model using the best parameters found by `run()`.
        
        This method should be called only after `run()` has completed.
        
        Returns:
            The instance of the class itself, to allow for method chaining.
        """
        if self.best_params_ is None:
            raise RuntimeError(
                "You must run the optimization with .run() before fitting the best model."
            )

        # Combine the best dynamic parameters with the fixed ones
        final_params = {**self.best_params_, **self._fixed_params}
        
        print("Training the final model with the best parameters...")
        self.best_model_ = self.model_instance.train(
            params=final_params, **self._calibration_kwargs
        )
        print("Final model has been trained and is stored in the '.best_model_' attribute.")
        
        return self

class GeneticOptimization:

    def _create_objective_func(self, model_instance, space, fixed_params):
        pass

    def run(self, model_instance, space, fixed_params,  **kwargs):
        pass