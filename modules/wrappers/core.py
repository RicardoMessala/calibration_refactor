from modules.calibration_algorithm.gbdt import GBDTTrainning, XGBoostTraining

class LGBMWrapper: # Wrapper para LightGBM
    def __init__(self, X_train, y_train, X_test, y_test):
        self.GBDTTraining = GBDTTrainning
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train

    def train(self, **params):
        trainer = self.GBDTTraining(params=params)
        return trainer.train(self.X_train, self.y_train, self.X_test, self.y_test)

    def predict(self, model):
        predictor = self.GBDTTraining(params={})
        return predictor.predict(model, self.X_test)

class XGBoostWrapper: # Wrapper para XGBoost
    def __init__(self, X_train, y_train, X_test, y_test):
        self.XGBoostTraining = XGBoostTraining
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train

    def train(self, **params):
        params.pop('metric', None) # Remove 'metric' se existir, pois já está no 'eval_metric'
        trainer = self.XGBoostTraining(params=params)
        return trainer.train(self.X_train, self.y_train, self.X_test, self.y_test)

    def predict(self, model):
        predictor = self.XGBoostTraining(params={})
        return predictor.predict(model, self.X_test)