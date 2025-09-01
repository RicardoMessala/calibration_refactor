import sklearn
import pandas as pd
import numpy as np
import sklearn.model_selection
from database.src import dataset_preprocessing
from sklearn.model_selection import train_test_split


def set_alpha(dataframe_relevant_data:pd):
    
    return dataframe_relevant_data["alpha"]

def set_raw_data(dataframe_relevant_data: pd, stdrings_dataframe: pd):
    Reta, E_E1E2, E_EM, E_E0E1, RHad = dataset_preprocessing.calc_sv(
        dataframe_relevant_data["cluster_e237"],
        dataframe_relevant_data["cluster_e277"],
        stdrings_dataframe.iloc[:, 0:99]
    )
    clEt = dataframe_relevant_data["cluster_et"]
    clEta = dataframe_relevant_data["cluster_eta"]

    return pd.concat([ clEt, 
            clEta, 
            Reta, 
            E_EM, 
            E_E1E2, 
            E_E0E1, 
            RHad
            ], 
            axis=1
        )

def set_std_rings_data(stdrings_dataframe: pd):

    return dataset_preprocessing.CalcRings(stdrings_dataframe.iloc[:, 0:99])

def set_quarter_rings_data(stdrings_dataframe:pd, quarterrings_dataframe:pd, dataframe_relevant_data:pd ,mode='standard'):
        
        if mode in ('standard'):
            
            return dataset_preprocessing.calc_asym_weights(
                quarterrings_dataframe.iloc[:, 0:378], 
                stdrings_dataframe.iloc[:, 0:99]
            )


        elif mode == 'delta':

            return dataset_preprocessing.calc_asym_weights_delta(
                qr=quarterrings_dataframe.iloc[:, 0:378],
                rings=stdrings_dataframe.iloc[:, 0:99],
                cluster_eta=dataframe_relevant_data["cluster_eta"],
                cluster_phi=dataframe_relevant_data["cluster_phi"],
                delta_eta_calib=dataframe_relevant_data["delta_eta_calib"],
                delta_phi_calib=dataframe_relevant_data["delta_phi_calib"],
                hotCellEta=dataframe_relevant_data["hotCellEta"],
                hotCellPhi=dataframe_relevant_data["hotCellPhi"],
                mc_et=dataframe_relevant_data["mc_et"]
            )
    
        else:
            raise ValueError('Invalide mode for qrings', mode)
        
def get_raw_input(dataframe_relevant_data, train_size, random_state): #ajustar parametros editaveis que servirão como input do usuario, random_state e train_size, por exemplo
    alpha = set_alpha(dataframe_relevant_data)
    XtrT, XteT, ytr, yte = sklearn.model_selection.train_test_split(set_raw_data(), 
                                                                    alpha, 
                                                                    train_size, 
                                                                    random_state
                                                                    )
    
    return XtrT, XteT, ytr, yte

def get_std_rings_input(dataframe_relevant_data, stdrings_dataframe, train_size, random_state):
    stdrings_data=set_std_rings_data(stdrings_dataframe)
    alpha = set_alpha(dataframe_relevant_data)
    XtrT, XteT, ytr, yte = sklearn.model_selection.train_test_split(stdrings_data,
                                                                    alpha,
                                                                    train_size,
                                                                    random_state
                                                                    )
    
    return XtrT, XteT, ytr, yte

def get_quarter_rings_input(dataframe_relevant_data, stdrings_dataframe, quarterrings_dataframe, train_size, random_state, mode):
    alpha = set_alpha(dataframe_relevant_data)
    qrings_data = set_quarter_rings_data(dataframe_relevant_data,
                                        stdrings_dataframe,
                                        quarterrings_dataframe,
                                        mode
                                        )
    XtrT, XteT, ytr, yte = sklearn.model_selection.train_test_split(qrings_data,
                                                                    alpha,
                                                                    train_size,
                                                                    random_state
                                                                    )
    
    return XtrT, XteT, ytr, yte


def prepare_and_split(dataframe_relevant_data, 
                      data_mode="raw", 
                      train_size=0.8, 
                      random_state=42, 
                      **kwargs):
    """
    General function to prepare data and split into train/test.
    
    Parameters
    ----------
    dataframe_relevant_data : pd.DataFrame
        Base dataframe used to calculate alpha.
    data_mode : str, default="raw"
        Options: "raw", "std_rings", "quarter_rings".
    train_size : float, default=0.8
        Proportion of data used for training.
    random_state : int, default=42
        Random seed.
    **kwargs : dict
        Extra parameters required depending on data_mode.
        - std_rings → requires `stdrings_dataframe`
        - quarter_rings → requires `stdrings_dataframe`, `quarterrings_dataframe`, and maybe `mode`
    """
    
    alpha = set_alpha(dataframe_relevant_data)
    
    if data_mode == "raw":
        data = set_raw_data()
    
    elif data_mode == "std_rings":
        if "stdrings_dataframe" not in kwargs:
            raise ValueError("stdrings_dataframe is required for std_rings mode.")
        data = set_std_rings_data(kwargs["stdrings_dataframe"])
    
    elif data_mode == "quarter_rings":
        if not all(k in kwargs for k in ["stdrings_dataframe", "quarterrings_dataframe"]):
            raise ValueError("stdrings_dataframe and quarterrings_dataframe are required for quarter_rings mode.")
        data = set_quarter_rings_data(dataframe_relevant_data,
                                      kwargs["stdrings_dataframe"],
                                      kwargs["quarterrings_dataframe"],
                                      kwargs.get("mode"))
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")
    
    return train_test_split(data, alpha, train_size=train_size, random_state=random_state)

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
    print('NEW COLUMNS')
    """
    Cria uma máscara booleana a partir de uma pandas Series com base em valores
    discretos ou em um intervalo [min, max].

    Parameters
    ----------
    column : pandas.Series
        Coluna a partir da qual a máscara será criada.
    values : list, tuple, set, pandas.Series, numpy.ndarray
        - Se contiver múltiplos valores discretos, seleciona as linhas que
          correspondem a esses valores.
        - Se tiver exatamente dois elementos [min, max]:
            * Ambos definidos   -> seleciona valores >= min e < max
            * Apenas min        -> seleciona valores >= min
            * Apenas max        -> seleciona valores < max
            * Ambos None        -> seleciona todos os valores

    Returns
    -------
    numpy.ndarray
        Array de máscara booleana.
    """
    # Garante que 'values' seja um tipo de lista para verificação de tamanho
    values = list(values)

    # Caso: intervalo [min, max]
    if len(values) == 2 and all(isinstance(v, (int, float, type(None))) for v in values):
        min_val, max_val = values

        # --- INÍCIO DA MODIFICAÇÃO SOLICITADA ---
        # Define a coluna a ser usada para comparação.
        # Se o nome da coluna for 'cluster_eta', usa seu valor absoluto.
        if column.name == 'cluster_eta':
            print(f"INFO: Aplicando comparação com módulo para a coluna '{column.name}'.")
            target_column = column.abs()
        else:
            target_column = column
        # --- FIM DA MODIFICAÇÃO SOLICITADA ---

        if min_val is not None and max_val is not None:
            mask = (target_column >= min_val) & (target_column < max_val)
        elif min_val is not None:  # apenas min definido
            mask = target_column >= min_val
        elif max_val is not None:  # apenas max definido
            mask = target_column < max_val
        else:  # ambos None -> seleciona tudo
            mask = pd.Series(True, index=column.index)
    
    # Caso: valores discretos
    else:
        mask = column.isin(values)

    return mask.to_numpy()

def setup_params(params):
    
    
    return params