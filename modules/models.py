import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import lightgbm as lgb

from typing import Any


class LGBMTrainning:
    def __init__(
        self,
        X_train: np.ndarray | pd.DataFrame,
        X_test: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series | list[float] | list[int],
        y_test: np.ndarray | pd.Series | list[float] | list[int],
    ) -> None:
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred: np.ndarray | pd.Series | list[float] | list[int] = None

    def train(
        self,
        params: dict[str, Any],
        **kwargs: Any
    ) -> lgb.Booster:
        
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_test = lgb.Dataset(self.X_test, self.y_test, reference=lgb_train)
        model = lgb.train(
            params=params,
            train_set=lgb_train,
            valid_sets=[lgb_test],
            **kwargs
        )
        return model

    def predict(self, model: lgb.Booster) -> np.ndarray:
        self.y_pred = model.predict(self.X_test, num_iteration=model.best_iteration)
        return self.y_pred

class XGBoostTraining: # Treinador para XGBoost
    def __init__(self, params, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.params = params
        self.stopping_rounds = 'stopping_rounds'

    def train(self, Xtr, ytr, Xte, yte):
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dval = xgb.DMatrix(Xte, label=yte)
        model = xgb.train(
            self.params, dtrain, evals=[(dval, 'eval')],
            early_stopping_rounds=self.stopping_rounds, verbose_eval=False
        )
        return model

    def predict(self, model, X_test):
        dtest = xgb.DMatrix(X_test)
        return model.predict(dtest, iteration_range=(0, model.best_iteration))

class TransformersTrainning:
    def __init__(self, params, stopping_rounds=5):
        self.params = params
        self.early_stop = stopping_rounds
