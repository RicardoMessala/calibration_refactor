from sklearn.metrics import mean_absolute_error, mean_squared_error
from modules.wrappers.core import LGBMWrapper, XGBoostWrapper
from skopt import gp_minimize
from skopt.utils import use_named_args
import numpy as np

class BayesianOptimization:
    """
    Classe de otimização que seleciona e instancia o wrapper de modelo
    apropriado com base no 'model_type' fornecido.
    """
    def __init__(self, model_type, X_train, y_train, X_test, y_test, space, fixed_params, metric=mean_absolute_error):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.space = space
        self.fixed_params = fixed_params
        self.metric = metric
        
        # O Padrão "Factory" acontece aqui!
        self.model_wrapper = self._get_wrapper(model_type)
    
    def _get_wrapper(self, model_type: str):
        """
        Método Fábrica: Mapeia uma string para a classe Wrapper correspondente
        e a instancia.
        """
        # Mapa de wrappers disponíveis
        available_wrappers = {
            "lightgbm": LGBMWrapper,
            "xgboost": XGBoostWrapper,
            # Adicione outros wrappers aqui no futuro (ex: "catboost": CatBoostWrapper)
        }
        
        wrapper_class = available_wrappers.get(model_type.lower())
        
        if wrapper_class is None:
            raise ValueError(f"Modelo '{model_type}' não suportado. "
                             f"Opções disponíveis: {list(available_wrappers.keys())}")
        
        # Instancia o wrapper selecionado, passando os dados de validação
        print(f"Wrapper para o modelo '{model_type}' selecionado.")
        return wrapper_class(self.X_train, self.y_train, self.X_test, self.y_test)
    
    def set_parameters():
        pass

    def _create_objective_func(self):
        @use_named_args(self.space)
        def objective(**params):
            all_params = {**params, **self.fixed_params}
            model = self.model_wrapper.train(**all_params)
            y_pred = self.model_wrapper.predict(model)
            score = mean_absolute_error(self.y_test, y_pred)
            return score
        return objective

    def run(self, **kwargs):
        print("\nIniciando a otimização Bayesiana com gp_minimize...")
        result_gp = gp_minimize(
            func=self._create_objective_func(),
            dimensions=self.space,
            **kwargs
        )
        return result_gp