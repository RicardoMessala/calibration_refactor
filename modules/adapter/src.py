from modules.calibration_algorithm.gbdt.gbdt_trainning import GBDTTrainning, XGBoostTraining

class LGBMWrapper: # Wrapper para LightGBM
    def __init__(self, X_test, y_test):
        self.GBDTTraining = GBDTTrainning
        self.X_test = X_test
        self.y_test = y_test

    def train(self, X_train, y_train, **params):
        trainer = self.GBDTTraining(params=params)
        return trainer.train(X_train, y_train, self.X_test, self.y_test)

    def predict(self, model, X_test):
        predictor = self.GBDTTraining(params={})
        return predictor.predict(model, X_test)

class XGBoostWrapper: # Wrapper para XGBoost
    def __init__(self, X_test, y_test):
        self.XGBoostTraining = XGBoostTraining
        self.X_test = X_test
        self.y_test = y_test

    def train(self, X_train, y_train, **params):
        # XGBoost não gosta de parâmetros extras que não conhece
        params.pop('metric', None) # Remove 'metric' se existir, pois já está no 'eval_metric'
        trainer = self.XGBoostTraining(params=params)
        return trainer.train(X_train, y_train, self.X_test, self.y_test)

    def predict(self, model, X_test):
        predictor = self.XGBoostTraining(params={})
        return predictor.predict(model, X_test)