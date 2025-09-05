import inspect
import numpy as np
import pandas as pd
from skopt.space import Dimension
from modules.models import LGBMTrainning, XGBoostTraining
from modules.optimizers import BayesianOptimization
from sklearn.metrics import mean_absolute_error, mean_squared_error
from abc import ABC
from typing import Any, Callable, Type, Union, List, Tuple, Optional
from functools import partial

class AbstractFactory(ABC):
    """
    Creates an abstract factory for optimization classes, models, and metrics.
    """

    OPTIMIZATION_CLASSES: dict[str, Type[Any]] = {
        'gp_minimize': BayesianOptimization,
    }

    MODEL_CLASSES: dict[str, Type[Any]] = {
        'lgbm': LGBMTrainning,
        'xgboost': XGBoostTraining,
    }

    METRIC_FUNCTIONS: dict[str, Callable[..., float]] = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
    }

    def _get_component(
        self,
        component_var: Union[str, Type[Any], Callable, Any],
        dict_name: str,
        instantiate: bool = True,
        args: tuple = None,
        **kwargs,
    ) -> Any:
        """
        Selects, instantiates, or returns a component (class, function, etc.) from a dictionary.

        Args:
            component_var: The component identifier (string), the class/function itself,
                         or an already created instance.
            dict_name: The name of the attribute dictionary in the class where the search
                       will be performed (e.g., 'MODEL_CLASSES', 'METRIC_FUNCTIONS').
            instantiate: If True, attempts to instantiate the found component.
                         If False, returns the component as is (e.g., the class or function itself).
            *args: Positional arguments to be passed to the class constructor
                   if `instantiate` is True.
            **kwargs: Keyword arguments to be passed to the class constructor
                      if `instantiate` is True.

        Returns:
            An instance of the requested component (if instantiate=True) or the component
            itself (class/function) that was found. If `component_var` is already an
            instance, it returns the variable itself.

        Raises:
            ValueError: If the dictionary or component is not found.
            TypeError: If trying to instantiate something that is not a class.
        """
        # If component_var is already an instance, there's nothing to do, just return it.
        if not isinstance(component_var, (str, type)) and not callable(component_var):
            return component_var

        target_component: Union[Type[Any], Callable]

        if isinstance(component_var, str):
            component_key = component_var.lower()
            try:
                target_dict = getattr(self, dict_name)
            except AttributeError:
                raise ValueError(
                    f"The attribute dictionary '{dict_name}' does not exist in class {self.__class__.__name__}."
                )

            component = target_dict.get(component_key)
            if component is None:
                raise ValueError(
                    f"Component '{component_key}' was not found in the dictionary '{dict_name}'."
                )
            target_component = component
        else:  # If it's not a string, it's a class or a function (callable)
            target_component = component_var

        # If the instruction is to instantiate, create the instance.
        if instantiate:
            if not inspect.isclass(target_component):
                 raise TypeError(f"The component '{component_var}' is not a class and cannot be instantiated.")
            return target_component(*args, **kwargs)
        
        # Otherwise, return the component as it is (the class or the function).
        return target_component

class RunOptimization(AbstractFactory):

    def __init__(self):
        self.optimizer: List[Any] = []
        self.model: List[Any] = []

    def run_multiple_optimizations(
        self,
        opt_class: Union[str, Type[Any], Any],
        model_class: Union[str, Type[Any], Any],
        datasets: List[Tuple[
            Union[np.ndarray, pd.DataFrame],
            Union[np.ndarray, pd.DataFrame],
            Union[np.ndarray, pd.Series, list[float], list[int]],
            Union[np.ndarray, pd.Series, list[float], list[int]]
        ]],
        space: List[Dimension],
        fixed_params: dict[str, Any],
        metric: Union[str, Callable[..., float]],
        calibration_kwargs: Optional[dict[str, Any]] = None,
        optimization_kwargs: Optional[dict[str, Any]] = None
    ) -> List[Any]:
        
        """
        Run BayesianOptimization in loop for multiple datasets.
        """
        results: List[Any] = []
        if datasets and not isinstance(datasets[0], (list, tuple)):
            datasets = [datasets]

        for data in datasets:
            tmp_optimizer_class = self._get_component(opt_class,'OPTIMIZATION_CLASSES', False)
            tmp_model_instance = self._get_component(model_class,'MODEL_CLASSES', True, data)
            tmp_metric=self._get_component(metric, 'METRIC_FUNCTIONS', False)

            # 1. Create the optimizer instance and STORE IT in a new variable.
            optimizer_instance = tmp_optimizer_class(tmp_model_instance)

            # 2. Use this new instance to run the optimization.
            res = optimizer_instance.run(
                space=space,
                fixed_params=fixed_params,
                metric=tmp_metric,
                calibration_kwargs=calibration_kwargs or {},
                **(optimization_kwargs or {})
            )

            # 3. Append the INSTANCE (not the class) to your list.
            self.optimizer.append(optimizer_instance)
            results.append(res)

        return results


class RunModel(AbstractFactory):

    def run_multiple_lgbm(
        self,
        model_class: Union[str, Type[Any], Any],
        datasets: List[Tuple[
            Union[np.ndarray, pd.DataFrame],
            Union[np.ndarray, pd.DataFrame],
            Union[np.ndarray, pd.Series, list[float], list[int]],
            Union[np.ndarray, pd.Series, list[float], list[int]]
        ]],
        params: dict[str, Any],
        metric: str = "mae",
        num_boost_round: int = 100,
        early_stopping_rounds: int = 10
    ) -> List[Any]:
        """
        Run LGBMRunner in loop for multiple datasets.
        """
        results: List[Any] = []

        for X_train, X_test, y_train, y_test in datasets:
            model_class = self._get_component(model_class, X_train, X_test, y_train, y_test)

            res = model_class.run(
                params=params,
                metric=metric,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
            )

            results.append(res)

        return results
