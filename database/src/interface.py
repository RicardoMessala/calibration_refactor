import sklearn
import pandas as pd
import numpy as np
import sklearn.model_selection
from modules import i
from database.src import dataset_preprocessing
from database.connection import sql_connection
from dataset_config import CreateInput

# External caching of data
_relevant_data_cache = sql_connection.get_relevant_data()
_quarterrings_data_cache = sql_connection.get_quarter_rings_data()
_stdrings_data_cache = sql_connection.get_standard_rings_data()
_quarterrings_relevant_cache = sql_connection.set_quarter_rings_data()
_stdrings_relevant_cache = sql_connection.set_standard_rings_data()

class SetupInputs:
    
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

        #Initialize LoopBuilder Class
        self.LoopBuilder = LoopBuilder
        self.CreateInput = CreateInput

    def get_inputs(self, params, rigns_type: str):
        wrappers = {
            "quarterrings": self.quarterrings_relevant_dataframe,
            "stdrings": self.stdrings_relevant_dataframe,
            # Adicione outros wrappers aqui no futuro (ex: "catboost": CatBoostWrapper)
        }
        
        wrapper_data= wrappers.get(rigns_type.lower())

        return self.LoopBuilder(wrapper_data, params).split_dataframe()


    def set_inputs(self, ):
        

        # meter o laço do for chamando o create inputs aqui
        
        pass
    def input_builder(self):

        #Fazer o construtor
        pass
    

class LoopBuilder:

    def __init__(self, data, params):
        self.data = data
        self.params = params

    def check_repetition(self, vector):

        pairs_per_row = []
        
        # generate pairs for each row
        for i, (elements, numbers) in enumerate(vector):
            pairs = [(el, num) for el in elements for num in numbers]
            pairs_per_row.append((i, pairs))
        
        # dictionary to track where each pair appears
        occurrences = {}
        for i, pairs in pairs_per_row:
            for pair in pairs:
                if pair not in occurrences:
                    occurrences[pair] = []
                occurrences[pair].append(i)
        
        # check repetitions
        repeated = {pair: rows for pair, rows in occurrences.items() if len(rows) > 1}
        
        if repeated:
            msg = "⚠️ Duplicate pairs detected:\n"
            for pair, rows in repeated.items():
                msg += f"Pair {pair} appears in rows {rows}\n"
            raise ValueError(msg.strip())
        else:
            print("No repeated pairs found.")

def split_dataframe_by_groups(df, params=None):
    """
    Divide o DataFrame em grupos com base em uma lista de filtros complexos.

    A função remove as linhas a cada iteração para evitar sobreposição.
    O último DataFrame na lista de retorno é o 'remainder', com as linhas não filtradas.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame original a ser dividido.
    params : list of lists of dicts, optional
        Define os grupos de filtros. A estrutura é:
        [
            [ {'COL_A': [VAL1, VAL2]}, {'COL_B': [VAL3]} ],  # Grupo 1 (COL_A E COL_B)
            [ {'COL_C': [VAL4, VAL5]} ],                      # Grupo 2
            ...
        ]
        Dentro de cada grupo (lista interna), as condições (dicionários) 
        são combinadas com um operador LÓGICO E (AND).
        Se None, retorna apenas [df].

    Returns
    -------
    list
        Uma lista de DataFrames, onde cada elemento corresponde a um grupo de filtros
        e o último elemento é o DataFrame com os dados restantes ('remainder').
    """
    if params is None:
        return [df.reset_index(drop=True)]

    result_list = []
    df_remaining = df.copy()

    for group_conditions in params:
        masks = []

        for condition_dict in group_conditions:
            for col_name, values in condition_dict.items():
                # Garante que values seja sempre iterável
                if not isinstance(values, (list, tuple, set, pd.Series, np.ndarray)):
                    values = [values]
                masks.append(df_remaining[col_name].isin(values).to_numpy())

        # Usa NumPy para combinar as máscaras (mais rápido que reduce com pandas)
        if masks:
            combined_mask = np.logical_and.reduce(masks)
        else:
            combined_mask = np.zeros(len(df_remaining), dtype=bool)

        # Filtra e adiciona à lista de resultados
        filtered = df_remaining.loc[combined_mask]
        result_list.append(filtered.reset_index(drop=True))

        # Remove linhas já usadas
        df_remaining = df_remaining.loc[~combined_mask]

    # Adiciona o que sobrou
    result_list.append(df_remaining.reset_index(drop=True))

    return result_list
