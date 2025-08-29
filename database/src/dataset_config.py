import sklearn
import pandas as pd
import numpy as np
import sklearn.model_selection
from database.src import dataset_preprocessing
from database.connection import sql_connection

# # External caching of data
# _relevant_data_cache = sql_connection.get_relevant_data()
# _quarterrings_data_cache = sql_connection.get_quarter_rings_data()
# _stdrings_data_cache = sql_connection.get_standard_rings_data()
# _quarterrings_relevant_cache = sql_connection.set_quarter_rings_data()
# _stdrings_relevant_cache = sql_connection.set_standard_rings_data()

class CreateInput:

    def __init__(self,
                    dataframe_relevant_data=None,
                    quarterrings_dataframe=None,
                    stdrings_dataframe=None,
                    quarterrings_relevant_dataframe=None,
                    stdrings_relevant_dataframe=None):

        # Initialize data attributes
        self.dataframe_relevant_data = dataframe_relevant_data #or _relevant_data_cache
        self.quarterrings_dataframe = quarterrings_dataframe #or _quarterrings_data_cache
        self.stdrings_dataframe = stdrings_dataframe #or _stdrings_data_cache
        self.quarterrings_relevant_dataframe = quarterrings_relevant_dataframe #or _quarterrings_relevant_cache
        self.stdrings_relevant_dataframe = stdrings_relevant_dataframe #or _stdrings_relevant_cache

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

def split_dataframe(data, params):
    
    if params is None:
        return [data.reset_index(drop=True)]

    result_list = []
    data_remaining = data.copy()

    for group_conditions in params:
        masks = []
        print('group_conditions: ', group_conditions)
        for condition_dict in group_conditions:
            print('condition_dict: ', condition_dict)
            for col_name, intervals in condition_dict.items():
                print('col_name: ', col_name)
                print('intervals: ', intervals)
                print ('intervals[0]: ', intervals[0], 'intervals[1]: ', intervals[1])
                print('TESTANDO')
                masks.append(make_mask(data_remaining[col_name], intervals))
                print('masks: ', masks)

        # Usa NumPy para combinar as máscaras (mais rápido que reduce com pandas)
        if masks:
            combined_mask = np.logical_and.reduce(masks)
        else:
            combined_mask = np.zeros(len(data_remaining), dtype=bool)

        # Filtra e adiciona à lista de resultados
        filtered = data_remaining.loc[combined_mask]
        result_list.append(filtered.reset_index(drop=True))

        # Remove linhas já usadas
        data_remaining = data_remaining.loc[~combined_mask]

    # Adiciona o que sobrou
    result_list.append(data_remaining.reset_index(drop=True))

    return result_list
    
def make_mask(column, values):
    print('values no metodo make_mask', values)
    

    """
    Create a boolean mask from a pandas Series based on either discrete values
    or an interval [min, max].

    Parameters
    ----------
    column : pandas.Series
        Column from which the mask will be created.
    values : list, tuple, set, pandas.Series, numpy.ndarray
        - If it contains multiple discrete values, rows matching those values are selected.
        - If it has exactly two elements [min, max]:
            * Both defined   -> select values >= min and < max
            * Only min       -> select values >= min
            * Only max       -> select values < max
            * Both None      -> select all values

    Returns
    -------
    numpy.ndarray
        Boolean mask array.
    """
    # Ensure values is iterable
    if not isinstance(values, (list, tuple, set, pd.Series, np.ndarray)):
        print('values no metodo make_mask', values)
        values = [values]

    # Case: interval [min, max]
    if len(values) == 2 and all(isinstance(v, (int, float, type(None))) for v in values):
        min_val, max_val = values
        print('min val: ', min_val)
        print('max val: ', max_val)

        if min_val is not None and max_val is not None:
            mask = (column >= min_val) & (column < max_val)
        elif min_val is not None:  # only min defined
            mask = column >= min_val
        elif max_val is not None:  # only max defined
            mask = column < max_val
        else:  # both None -> all True
            pass
    else:
        # Case: list of discrete values
        mask = column.isin(values)

    return mask.to_numpy()