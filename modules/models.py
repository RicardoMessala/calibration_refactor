import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Any, Union, List

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, List[float], List[int]]
Booster = lgb.Booster
class LGBMTraining:
    """
    Encapsulates the workflow for training, predicting, and saving a LightGBM model.

    This class stores the training and test datasets and manages the
    lifecycle of the internally trained model.

    Attributes:
        X_train (ArrayLike): Features for the training set.
        y_train (ArrayLike): Target for the training set.
        X_test (ArrayLike): Features for the test set.
        y_test (ArrayLike): Target for the test set.
        model (Booster | None): The internally trained LightGBM model.
        y_pred (ArrayLike | None): Predictions from the internal model on the test set.
    """
    def __init__(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_train: ArrayLike,
        y_test: ArrayLike,
    ) -> None:
        
        """Initializes the class with training and test data."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model: Booster | None = None
        self.y_pred: np.ndarray | None = None
        
    def train(self, params: dict[str, Any], **kwargs: Any) -> Booster:
        """
        Trains a LightGBM model with the provided data.

        Args:
            params (dict): Dictionary of parameters for LightGBM.
            **kwargs: Other keyword arguments for the `lgb.train` function
                      (e.g., num_boost_round, callbacks).

        Returns:
            lgb.Booster: The trained model.
        """
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_test = lgb.Dataset(self.X_test, self.y_test, reference=lgb_train)

        self.model = lgb.train(
            params=params,
            train_set=lgb_train,
            valid_sets=[lgb_train, lgb_test],
            **kwargs
        )
        return self.model

    def predict(self) -> np.ndarray:
        """
        Makes predictions on X_test using the class's INTERNAL trained model.

        This method updates the `self.y_pred` attribute with the result.

        Raises:
            RuntimeError: If the method is called before the model has been trained.

        Returns:
            np.ndarray: A numpy array with the predictions.
        """
        if self.model is None:
            raise RuntimeError("The internal model has not been trained yet. Call the 'train' method first.")

        self.y_pred = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
        return self.y_pred
    
    def predict_with(self, external_model:Booster) -> np.ndarray:
        """
        Makes predictions on X_test using a provided EXTERNAL model.

        This method is useful for comparing or validating models that were not
        trained by this class instance. It does NOT modify the internal state
        of the object (i.e., `self.y_pred` remains unchanged).

        Args:
            external_model (lgb.Booster): A trained LightGBM model.

        Returns:
            np.ndarray: A numpy array with the predictions from the external model.
        """
        if not isinstance(external_model, Booster):
            raise TypeError("The provided model is not a valid lgb.Booster object.")
        
        return external_model.predict(self.X_test, num_iteration=external_model.best_iteration)

    def save_model(self, path: str) -> None:
        """
        Saves the internally trained model to a file.

        Args:
            path (str): The file path where the model will be saved (e.g., 'model.txt').

        Raises:
            RuntimeError: If the method is called before the model has been trained.
        """
        if self.model is None:
            raise RuntimeError("The internal model has not been trained yet. Call the 'train' method first.")

        self.model.save_model(path)

    def plot_metric(self, **kwargs: Any):
        """
        Plots the metric results recorded during training.

        This method is a wrapper around `lightgbm.plot_metric`. It requires
        matplotlib to be installed. The user is responsible for calling
        `matplotlib.pyplot.show()` to display the plot.

        Args:
            **kwargs: Other keyword arguments to be passed to `lgb.plot_metric`
                      (e.g., ax, figsize, title, metric).

        Raises:
            RuntimeError: If the method is called before the model has been trained.
            ImportError: If matplotlib is not installed.

        Returns:
            matplotlib.axes.Axes: The axes object with the plot.
        """
        if self.model is None:
            raise RuntimeError("The model has not been trained yet. Call the 'train' method first.")
        
        try:
            import matplotlib
        except ImportError:
            raise ImportError("matplotlib is required to use plot_metric. Please install it with 'pip install matplotlib'.")

        return lgb.plot_metric(self.model, **kwargs)


class XGBoostTraining:
    def __init__(self, X_train, X_test, y_train, y_test): 
        # A fábrica manda 4 argumentos, nós recebemos 4 argumentos!
        
        self.model = None
        
        # Chama a função de processamento passando os 4 argumentos
        self._processar_dados(X_train, X_test, y_train, y_test)

    def _processar_dados(self, X_train, X_test, y_train, y_test):
        # Se os dados vierem como uma lista de pedaços (folds/gavetas), nós concatenamos.
        # Caso contrário, apenas salvamos direto.
        
        if isinstance(X_train, list):
            self.X_train = pd.concat(X_train, axis=0)
            self.X_test  = pd.concat(X_test, axis=0)
            self.y_train = pd.concat(y_train, axis=0)
            self.y_test  = pd.concat(y_test, axis=0)
        else:
            self.X_train = X_train
            self.X_test  = X_test
            self.y_train = y_train
            self.y_test  = y_test
            
    def train(self, params: Union[list, dict[str, Any]], **kwargs: Any ) -> Booster:
        # Converte os dados (Pandas/Numpy) para DMatrix.
        xgb_train = xgb.DMatrix(self.X_train, self.y_train)
        xgb_test = xgb.DMatrix(self.X_test, self.y_test)

        # Chama a função principal de treino da biblioteca
        self.model = xgb.train(
            params = params,      
            dtrain = xgb_train,   
            evals = [(xgb_train, 'train'), (xgb_test, 'test')],
            **kwargs 
        )
        
        return self.model

    def predict(self) -> np.ndarray:
        # Trava de segurança: Se tentar prever sem treinar, dá erro.
        if self.model is None:
            raise RuntimeError("The internal model has not been trained yet. Call the 'train' method first.")
        
        # Realiza a previsão usando a melhor iteração.
        self.y_pred = self.model.predict(xgb.DMatrix(self.X_test), iteration_range=(0, self.model.best_iteration + 1))
        
        return self.y_pred
        

class TransformersTrainning:
    def __init__(self, params, stopping_rounds=5):
        self.params = params
        self.early_stop = stopping_rounds
