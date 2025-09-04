from sklearn.metrics import mean_absolute_error, mean_squared_error
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Dimension
from scipy.optimize import OptimizeResult
from typing import Any, Callable, Type, Union


class OptFactory:
    pass

class BayesianOptimization(OptFactory):
    """
    Optimization class that selects and instantiates the appropriate model
    wrapper based on the provided 'model_type'.
    """

    def _create_objective_func(
        self,
        model_class: Union[str, Type[Any], Any],
        space: list[Dimension],
        fixed_params: dict[str, Any],
        metric: Union[str, Callable[..., float]],
        calibration_kwargs: Union[dict[str, Any], None] = None,
    ) -> Callable[..., float]:
        """
        Create the objective function for optimization.
        """
        
        @use_named_args(space)
        def objective(**params: Any) -> float:
            all_params = {**params, **fixed_params}
            model = model_class.train(params=all_params, **(calibration_kwargs or {}))
            y_pred = model_class.predict(model)
            return metric(y_pred)

        return objective

    def run(
        self,
        model_class: Union[str, Type[Any], Any],
        space: list[Dimension],
        fixed_params: dict[str, Any],
        metric: Union[str, Callable[..., float]],
        calibration_kwargs: Union[dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> OptimizeResult:
        """
        Run Bayesian optimization.
        """
        objective_func = self._create_objective_func(
            model_class, space, fixed_params, metric, calibration_kwargs
        )

        return gp_minimize(
            func=objective_func,
            dimensions=space,
            **kwargs,
        )


class GeneticOptimization(OptFactory):

    def _create_objective_func(self, model_class, space, fixed_params):
        pass

    def run(self, model_class, space, fixed_params,  **kwargs):
        pass