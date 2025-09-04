import lightgbm as lgb
import xgboost as xgb
from abc import ABC, abstractmethod

class CalibrationFactory:
    
    @abstractmethod
    def train(self, X_train, X_test, y_train, y_test):
        pass
        
    @abstractmethod    
    def predict(self, model):
        pass

class LGBMTrainning:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, params, **kwargs):
        # Criar os datasets
        lgb_train = lgb.Dataset(self.X_train, self.y_train)
        lgb_test = lgb.Dataset(self.X_test, self.y_test, reference=lgb_train)

        # Treinar o modelo sem normalização
        model = lgb.train(params=params, train_set=lgb_train,
                          valid_sets=[lgb_test],
                          **kwargs)
        return model
    
    def predict(self, model):
        predictions = model.predict(self.X_test)
        return predictions

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
