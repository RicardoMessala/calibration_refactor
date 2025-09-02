import gbdt
import inspect
from abc import ABC
from modules.hyperparameters_opt.bayesian_opt import BayesianOptimization

class AbstractFactory(ABC):

    """
    Create an abstract factory for optimization classes.
    """

    # Dictionary mapping strings to classes
    MODEL_CLASSES = {
                    'lgbm': gbdt.LGBMTrainning,
                    'xgboost': gbdt.XGBoostTraining,
                }

    # Dictionary mapping strings to metric functions
    OPTIMIZATION_CLASSES = {
                        'gp_minimize': BayesianOptimization,
                    }
     
    def _select_model_class(self, model_class, X_train, y_train, X_test, y_test):
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
            return cls(X_train, y_train, X_test, y_test)

        # Case: class
        elif inspect.isclass(model_class):
            return model_class(X_train, y_train, X_test, y_test)

        # Case: instance
        else:
            return model_class

    def run():
        pass


class RunOptimization(AbstractFactory):

    def run_multiple_optimizations(self,
        datasets: list,
        opt_class,
        model_class,
        space,
        fixed_params,
        metric,
        objective_func_kwargs: dict = None,
        optimization_kwargs: dict = None
    ):
        """
        Run BayesianOptimization in loop for multiple datasets.
        
        Args:
            datasets (list): Lista de tuplas (X_train, y_train, X_test, y_test)
            model_class: Classe do modelo a ser otimizado
            space: Espaço de busca
            fixed_params (dict): Parâmetros fixos do modelo
            metric: Métrica de avaliação
            objective_func_kwargs (dict): Argumentos extras para função objetivo
            optimization_kwargs (dict): Argumentos extras para gp_minimize
        
        Returns:
            results (list): Lista de resultados de otimização
        """
        results = []

        for i, (X_train, y_train, X_test, y_test) in enumerate(datasets, start=1):
            print(f"\n[INFO] Running optimization {i}/{len(datasets)}...")
            
            optimizer = self._select_model_class(opt_class, X_train, y_train, X_test, y_test)
            
            res = optimizer.run(
                model_class=model_class,
                space=space,
                fixed_params=fixed_params,
                metric=metric,
                objective_func_kwargs=objective_func_kwargs or {},
                **(optimization_kwargs or {})
            )
            
            results.append(res)
        
        return results

class RunModel(AbstractFactory):

    def run_multiple_lgbm(self,
        model_class,
        datasets: list,
        params: dict,
        metric="mae",
        num_boost_round=100,
        early_stopping_rounds=10
    ):
        """
        Run LGBMRunner in loop for multiple datasets.

        Args:
            datasets (list): Lista de tuplas (X_train, y_train, X_test, y_test)
            params (dict): Parâmetros do LightGBM
            metric (str): Métrica de avaliação
            num_boost_round (int): Número de boosting rounds
            early_stopping_rounds (int): Critério de early stopping

        Returns:
            results (list): Lista de dicts com modelos e scores
        """
        results = []

        for i, (X_train, y_train, X_test, y_test) in enumerate(datasets, start=1):
            print(f"\n[INFO] Running LGBM {i}/{len(datasets)}...")

            runner = self._select_model_class(model_class, X_train, y_train, X_test, y_test)

            res = runner.run(
                params=params,
                metric=metric,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
            )

            results.append(res)

        return results
