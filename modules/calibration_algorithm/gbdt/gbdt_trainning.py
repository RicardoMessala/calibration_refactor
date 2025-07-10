import lightgbm as lgb

class GBDTTrainning:

    def __init__(self, params, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.params = params
        self.stopping_rounds = getattr(self, "stopping_rounds", 5)

    def StandartBDT(self, XtrT, ytr, XteT, yte):
        # Criar os datasets
        lgb_train = lgb.Dataset(XtrT, ytr)
        lgb_test = lgb.Dataset(XteT, yte, reference=lgb_train)

        # Treinar o modelo sem normalização
        model = lgb.train(self.params, lgb_train, valid_sets=[lgb_test, lgb_train], callbacks=[lgb.early_stopping(stopping_rounds=self.stopping_rounds)])
        
        return model

class TransformersTrainning:
    def __init__(self, params, stopping_rounds=5):
        self.params = params
        self.early_stop = stopping_rounds
