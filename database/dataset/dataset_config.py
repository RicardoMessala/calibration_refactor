import pandas as pd
import numpy as np
import sklearn.model_selection
from database.dataset import dataset_preprocessing
from typing import Tuple, Optional, Literal, List, Union


def set_alpha(stdrings_cluster_dataframe:pd):
    return stdrings_cluster_dataframe["alpha"]

def set_raw_data(stdrings_cluster_dataframe: pd):
    Reta, E_E1E2, E_EM, E_E0E1, RHad = dataset_preprocessing.calc_sv(
        stdrings_cluster_dataframe["cluster_e237"],
        stdrings_cluster_dataframe["cluster_e277"],
        stdrings_cluster_dataframe.iloc[:, 0:99]
    )
    clEt = stdrings_cluster_dataframe["cluster_et"]
    clEta = stdrings_cluster_dataframe["cluster_eta"]

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

def set_std_rings_data(stdrings_cluster_dataframe: pd):

    return dataset_preprocessing.CalcRings(stdrings_cluster_dataframe.iloc[:, 0:99])

def set_quarter_rings_data(stdrings_cluster_dataframe:pd, qrings_cluster_dataframe:pd ,mode:str='delta'):
        
        if mode in ('default'):
            
            return dataset_preprocessing.calc_asym_weights(
                qrings_cluster_dataframe.iloc[:, 0:378], 
                stdrings_cluster_dataframe.iloc[:, 0:99]
            )


        elif mode == 'delta':

            return dataset_preprocessing.calc_asym_weights_delta(
                qr=qrings_cluster_dataframe.iloc[:, 0:378],
                rings=stdrings_cluster_dataframe.iloc[:, 0:99],
                cluster_eta=stdrings_cluster_dataframe["cluster_eta"],
                cluster_phi=stdrings_cluster_dataframe["cluster_phi"],
                delta_eta_calib=stdrings_cluster_dataframe["delta_eta_calib"],
                delta_phi_calib=stdrings_cluster_dataframe["delta_phi_calib"],
                hotCellEta=stdrings_cluster_dataframe["hotCellEta"],
                hotCellPhi=stdrings_cluster_dataframe["hotCellPhi"],
                mc_et=stdrings_cluster_dataframe["mc_et"]
            )
    
        else:
            raise ValueError('Invalide mode for qrings', mode)

def prepare_and_split_data(
    input_type: Literal['raw', 'std_rings', 'quarter_rings'],
    stdrings_df: Union[pd.DataFrame, List[pd.DataFrame]],
    qrings_df: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    mode: Optional[str] = None,
    train_size: float = 0.8,
    random_state: Optional[int] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Prepares data from one or more DataFrames and splits it into training and testing sets.

    Can accept a single DataFrame or a list of DataFrames for batch processing.

    Args:
        input_type (str): The data preparation type. 
                          Can be 'raw', 'std_rings', or 'quarter_rings'.
        stdrings_df (Union[pd.DataFrame, List[pd.DataFrame]]): A single DataFrame or a list
                                                               of DataFrames to be processed.
        qrings_df (Optional[Union[pd.DataFrame, List[pd.DataFrame]]]): A single DataFrame or list
                                                                         of DataFrames, required
                                                                         only for 'quarter_rings'.
        mode (Optional[str]): The operation mode required only for 'quarter_rings'.
        train_size (float): The proportion of the dataset to allocate to the training set.
        random_state (Optional[int]): Seed for the random number generator.

    Returns:
        List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]: 
            A list of tuples, where each tuple contains (X_train, X_test, y_train, y_test)
            for each DataFrame processed.
    """
    # --- Input Normalization ---
    # Check if the inputs are single DataFrames and convert them to lists if so.
    if not isinstance(stdrings_df, list):
        stdrings_df = [stdrings_df]
    
    if qrings_df is not None and not isinstance(qrings_df, list):
        qrings_df = [qrings_df]

    # --- Consistency Validation for 'quarter_rings' ---
    if input_type == 'quarter_rings':
        if qrings_df is None:
            raise ValueError("For 'quarter_rings', the 'qrings_df' argument is required.")
        if len(stdrings_df) != len(qrings_df):
            raise ValueError(
                "For 'quarter_rings', 'stdrings_df' and 'qrings_df' must be lists of the same length."
            )

    results = []
    for i in range(len(stdrings_df)):
        # Get the current DataFrame for this iteration
        current_std_df = stdrings_df[i]
        
        # 1. Define the target (y) for the current DataFrame
        alpha = set_alpha(current_std_df)

        # 2. Define the dispatch table for the current iteration
        data_setters = {
            'raw': lambda: set_raw_data(current_std_df),
            'std_rings': lambda: set_std_rings_data(current_std_df),
            'quarter_rings': lambda: set_quarter_rings_data(current_std_df, qrings_df[i], mode)
        }

        # 3. Get the correct data preparation action
        setter_action = data_setters.get(input_type)
        if not setter_action:
            raise ValueError(
                f"Unknown input type: '{input_type}'. "
                f"Valid values are {list(data_setters.keys())}."
            )

        # Execute the action to get the feature data
        data = setter_action()

        # 4. Perform the train-test split
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            data,
            alpha,
            train_size=train_size,
            random_state=random_state
        )
        results.append((X_train, X_test, y_train, y_test))

    return results

def split_dataframe(data, params):
    
    if params is None:
        return [data.reset_index(drop=True)]

    result_list = []
    data_remaining = data.copy()

    for group_conditions in params:
        masks = []
        for condition_dict in group_conditions:
            for col_name, intervals in condition_dict.items():
                masks.append(_make_mask(data_remaining[col_name], intervals))

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
    
def _make_mask(column, values):
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