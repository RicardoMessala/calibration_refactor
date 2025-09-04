from modules.models import LGBMTrainning, XGBoostTraining
import inspect
from abc import ABC
from modules.optimizers import BayesianOptimization

class AbstractFactory(ABC):

    """
    Create an abstract factory for optimization classes.
    """

        # Dictionary mapping strings to metric functions
    OPTIMIZATION_CLASSES = {
                        'gp_minimize': BayesianOptimization,
                    }
    
    # Dictionary mapping strings to classes
    MODEL_CLASSES = {
                    'lgbm': LGBMTrainning,
                    'xgboost': XGBoostTraining,
                }
     
    def _select_model_class(self, class_var, X_train, X_test, y_train, y_test):
        """
        Selects and returns the correct model class instance.

        Rules:
        1. If model_class is a string, looks it up in MODEL_CLASSES and instantiates the corresponding class.
        2. If model_class is a class, instantiates it with the given args and kwargs.
        3. If model_class is already an instance, returns it directly.
        """
        # Case: string
        if isinstance(class_var, str):
            class_var = class_var.lower()
            cls = self.MODEL_CLASSES.get(class_var)
            if cls is None:
                cls = self.OPTIMIZATION_CLASSES.get(class_var)
            if cls is None:
                raise ValueError(
                    f"A classe '{class_var}' não foi encontrada em MODEL_CLASSES "
                    f"nem em OPTIMIZATION_CLASSES."
                )
            
            return cls(X_train, X_test, y_train, y_test)

        # Case: class
        elif inspect.isclass(class_var):
            return class_var(X_train, X_test, y_train, y_test)

        # Case: instance
        else:
            return class_var

    def run():
        pass


class RunOptimization(AbstractFactory):

    def run_multiple_optimizations(self,
        opt_class,
        model_class,
        datasets: list,
        space,
        fixed_params,
        metric,
        calibration_kwargs: dict = None,
        optimization_kwargs: dict = None
    ):
        """
        Run BayesianOptimization in loop for multiple datasets.
        
        Args:
            datasets (list): Lista de tuplas (X_train, X_test, y_train, y_test)
            model_class: Classe do modelo a ser otimizado
            space: Espaço de busca
            fixed_params (dict): Parâmetros fixos do modelo
            metric: Métrica de avaliação
            calibration_kwargs (dict): Argumentos extras para função objetivo
            optimization_kwargs (dict): Argumentos extras para gp_minimize
        
        Returns:
            results (list): Lista de resultados de otimização
        """
        results = []
        if datasets and not isinstance(datasets[0], (list, tuple)):
            print("[INFO] Estrutura de dados 1D detectada. Adicionando uma dimensão para compatibilidade.")
            # Envolve a lista 'datasets' em outra lista para torná-la 2D.
            datasets = [datasets]

        for data  in datasets:
            print(f"\n[INFO] Running optimization", len(data))
            print(type(data))
            print(type(data[0]))
            print(type(data[1]))
            print(type(data[2]))
            print(type(data[3]))
            X_train, X_test, y_train, y_test = data
            optimizer = self._select_model_class(opt_class, X_train, X_test, y_train, y_test)
            
            res = optimizer.run(
                model_class=model_class,
                space=space,
                fixed_params=fixed_params,
                metric=metric,
                calibration_kwargs=calibration_kwargs or {},
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
            datasets (list): Lista de tuplas (X_train, X_test, y_train, y_test)
            params (dict): Parâmetros do LightGBM
            metric (str): Métrica de avaliação
            num_boost_round (int): Número de boosting rounds
            early_stopping_rounds (int): Critério de early stopping

        Returns:
            results (list): Lista de dicts com modelos e scores
        """
        results = []

        for i, (X_train, X_test, y_train, y_test) in enumerate(datasets, start=1):
            print(f"\n[INFO] Running LGBM {i}/{len(datasets)}...")

            runner = self._select_model_class(model_class, X_train, X_test, y_train, y_test)

            res = runner.run(
                params=params,
                metric=metric,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds
            )

            results.append(res)

        return results
