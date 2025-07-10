import sklearn
import pandas as pd
import sklearn.model_selection
from database.src import dataset_preprocessing
from database.connection import sql_connection

# External caching of data
_relevant_data_cache = sql_connection.get_relevant_data()
_quarterrings_data_cache = sql_connection.get_quarter_rings_data()
_stdrings_data_cache = sql_connection.get_standard_rings_data()
_quarterrings_relevant_cache = sql_connection.set_quarter_rings_data()
_stdrings_relevant_cache = sql_connection.set_standard_rings_data()

class CreateInput:

    def __init__(self,
                    dataframe_relevant_data=None,
                    quarterrings_dataframe=None,
                    stdrings_dataframe=None,
                    quarterrings_relevant_dataframe=None,
                    stdrings_relevant_dataframe=None):

        # Initialize data attributes
        self.dataframe_relevant_data = dataframe_relevant_data or _relevant_data_cache
        self.quarterrings_dataframe = quarterrings_dataframe or _quarterrings_data_cache
        self.stdrings_dataframe = stdrings_dataframe or _stdrings_data_cache
        self.quarterrings_relevant_dataframe = quarterrings_relevant_dataframe or _quarterrings_relevant_cache
        self.stdrings_relevant_dataframe = stdrings_relevant_dataframe or _stdrings_relevant_cache

        # initialize attributes columns columns
        self.clEt = self.dataframe_relevant_data["cluster_et"]
        self.clEta = self.dataframe_relevant_data["cluster_eta"]
        self.clPhi = self.dataframe_relevant_data["cluster_phi"]
        self.alpha = self.dataframe_relevant_data["alpha"]

        # Placeholder attributes (model inputs)
        self.XtrT = self.XteT = self.ytr = self.yte = None
        self.Reta = self.E_EM = self.E_E1E2 = self.E_E0E1 = self.RHad = None
        self.rings = self.qrings = self.userings = None

    def set_raw_data(self):
        self.Reta, self.E_E1E2, self.E_EM, self.E_E0E1, self.RHad = dataset_preprocessing.calc_sv(
            self.dataframe_relevant_data["cluster_e237"],
            self.dataframe_relevant_data["cluster_e277"],
            self.stdrings_dataframe.iloc[:, 0:99]
        )

        return pd.concat([ self.clEt, 
            self.clEta, 
            self.Reta, 
            self.E_EM, 
            self.E_E1E2, 
            self.E_E0E1, 
            self.RHad
            ], axis=1)

    def set_std_rings_data(self):

        return dataset_preprocessing.CalcRings(self.stdrings_dataframe.iloc[:, 0:99])

    def set_quarter_rings_data(self, mode='standard'):
            
            if mode in ('standard'):
                
                return dataset_preprocessing.calc_asym_weights(
                    self.quarterrings_dataframe.iloc[:, 0:378], 
                    self.stdrings_dataframe.iloc[:, 0:99]
                )


            elif mode == 'delta':

                return dataset_preprocessing.calc_asym_weights_delta(
                    qr=self.quarterrings_dataframe.iloc[:, 0:378],
                    rings=self.stdrings_dataframe.iloc[:, 0:99],
                    cluster_eta=self.dataframe_relevant_data["cluster_eta"],
                    cluster_phi=self.dataframe_relevant_data["cluster_phi"],
                    delta_eta_calib=self.dataframe_relevant_data["delta_eta_calib"],
                    delta_phi_calib=self.dataframe_relevant_data["delta_phi_calib"],
                    hotCellEta=self.dataframe_relevant_data["hotCellEta"],
                    hotCellPhi=self.dataframe_relevant_data["hotCellPhi"],
                    mc_et=self.dataframe_relevant_data["mc_et"]
                )
        
            else:
                raise ValueError('Invalide mode for qrings', mode)
            
    def get_raw_input(self, train_size, random_state): #ajustar parametros editaveis que servirão como input do usuario, random_state e train_size, por exemplo
        self.XtrT, self.XteT, self.ytr, self.yte = sklearn.model_selection.train_test_split(self.set_raw_data(), self.alpha, train_size=train_size, random_state=random_state)
        
        return self.XtrT, self.XteT, self.ytr, self.yte
    
    def get_std_rings_input(self, train_size, random_state):
        self.XtrT, self.XteT, self.ytr, self.yte = sklearn.model_selection.train_test_split(self.set_std_rings_data(), self.alpha, train_size=train_size, random_state=random_state)
        
        return self.XtrT, self.XteT, self.ytr, self.yte
    
    def get_quarter_rings_input(self, train_size, random_state, mode):
        self.XtrT, self.XteT, self.ytr, self.yte = sklearn.model_selection.train_test_split(self.set_quarter_rings_data(mode=mode), self.alpha, train_size=train_size, random_state=random_state)
        
        return self.XtrT, self.XteT, self.ytr, self.yte

    @classmethod
    def data_builder(self):
        pass
        # # TREINA OS MODELOS PARA CADA TIPO DE RING E SALVA OS ARQUIVOS
        # for userings in range(1,3):
        #     [Reta, E_E1E2, E_EM, E_E0E1, RHad, rings, qrings] = CreateInput.set_data(
        #     self.dataframe_relevant_data, self.stdrings_dataframe, self.quarterrings_dataframe, userings, delta)
            
        #     # Não alterar definição global dps x e y
        #     XtrT, XteT, ytr, yte = CreateInput.create_input(self.dataframe_relevant_data["cluster_et"],
        #                                     self.dataframe_relevant_data["cluster_eta"],
        #                                     self.dataframe_relevant_data["cluster_phi"],
        #                                     Reta,
        #                                     E_EM,
        #                                     E_E1E2,
        #                                     E_E0E1,
        #                                     RHad,
        #                                     rings,
        #                                     qrings,
        #                                     self.dataframe_relevant_data["alpha"],
        #                                     random_state,
        #                                     userings)