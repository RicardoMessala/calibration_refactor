import lightgbm as lgb
import xgboost as xgb

class GBDTTrainning:

    def __init__(self, params, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.params = params
        self.stopping_rounds = getattr(self, "stopping_rounds", 5)

    def train(self, XtrT, ytr, XteT, yte):
        # Criar os datasets
        lgb_train = lgb.Dataset(XtrT, ytr)
        lgb_test = lgb.Dataset(XteT, yte, reference=lgb_train)

        # Treinar o modelo sem normalização
        model = lgb.train(self.params, lgb_train, valid_sets=[lgb_test, lgb_train], callbacks=[lgb.early_stopping(stopping_rounds=self.stopping_rounds)])
        
        return model
    
    def predict(self, model, X_test):
        predictions = model.predict(X_test)
        return predictions

class XGBoostTraining: # Treinador para XGBoost
    def __init__(self, params, stopping_rounds=10):
        self.params = params
        self.stopping_rounds = stopping_rounds

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
