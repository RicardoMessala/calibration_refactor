import pandas as pd
import numpy as np
import sklearn.model_selection
from typing import List, Tuple, Optional, Union, Literal, Dict, Type, Callable
from database.dataset import new_processing
from abc import ABC, abstractmethod

# --- 1. Decorator para o Dispatcher de Métodos ---
def data_handler(model_name: str):
    """
    Decorador para marcar um método de instância como um handler.
    Ex: @data_handler("default") registra o método para ser chamado quando model="default".
    """
    def decorator(func: Callable) -> Callable:
        func._handler_model_name = model_name
        return func
    return decorator


class DataPreparationStrategy():
    """
    Classe Híbrida:
    1. Atua como Factory de subclasses (baseado em 'name').
    2. Atua como Dispatcher de métodos de instância (baseado em @data_handler).
    """

    # --- Lógica de Factory (Classe/Estática) ---
    
    # Registro de subclasses conhecidas (Topologias)
    _strategies: Dict[str, Type['DataPreparationStrategy']] = {}
    
    # Nome identificador da subclasse (deve ser sobrescrito pelas subclasses)
    name: Optional[str] = None 

    def __init_subclass__(cls, **kwargs):
        """
        Auto-registro: Chamado automaticamente quando uma classe herda desta.
        Registra a classe no dicionário _strategies se ela tiver um 'name'.
        """
        super().__init_subclass__(**kwargs)
        
        if hasattr(cls, 'name') and cls.name:
            cls._strategies[cls.name] = cls

    @classmethod
    def create_topology(cls, topology: str) -> 'DataPreparationStrategy':
        """
        Factory Method: Retorna uma INSTÂNCIA da estratégia correspondente ao nome da topologia.
        """
        strategy_class = cls._strategies.get(topology)
        
        if not strategy_class:
            valid_keys = list(cls._strategies.keys())
            raise ValueError(
                f"Topologia '{topology}' não encontrada. "
                f"Disponíveis: {valid_keys}"
            )
        
        return strategy_class()

    @classmethod
    def get_available_topologies(cls) -> List[str]:
        """Retorna lista de topologias registradas."""
        return list(cls._strategies.keys())


    # --- Lógica de Dispatcher (Instância) ---

    def __init__(self):
        """
        Ao instanciar, varremos os próprios métodos para encontrar
        aqueles marcados com @data_handler.
        """
        self._model_handlers: Dict[str, Callable[..., pd.DataFrame]] = {}
        self._register_handlers_automatically()

    def _register_handlers_automatically(self) -> None:
        """Introspecção para registrar métodos decorados."""
        for attribute_name in dir(self):
            method = getattr(self, attribute_name)
            
            # Verifica se é um método e se tem o metadado do decorator
            if callable(method) and hasattr(method, "_handler_model_name"):
                model_key = method._handler_model_name
                self._model_handlers[model_key] = method

    def register_handler(self, model_name: str, handler: Callable[..., pd.DataFrame]) -> None:
        """
        Permite registrar um novo handler dinamicamente após a instanciação (Runtime).
        Útil para injetar lógica customizada sem precisar criar uma nova subclasse.
        """
        if not callable(handler):
            raise TypeError(f"O handler para '{model_name}' deve ser chamável.")
        self._model_handlers[model_name] = handler

    def feature_modeling(self, df: pd.DataFrame, model: str = 'default', **kwargs) -> pd.DataFrame:
        """
        Executa o handler específico registrado na instância.
        """
        handler = self._model_handlers.get(model)
        
        if not handler:
            valid_models = list(self._model_handlers.keys())
            raise ValueError(
                f"Modelo '{model}' não suportado pela estratégia '{self.name}'. "
                f"Modelos disponíveis nesta estratégia: {valid_models}"
            )
            
        return handler(df, **kwargs)


# --- Implementações Concretas (Topologias) ---

class RawDataPreparation(DataPreparationStrategy):
    """Estratégia para dados 'raw'."""
    name = 'raw'  # Chave para o Factory .create('raw')
    
    @data_handler("default") # Chave para o Dispatcher .feature_modeling(..., model='default')
    def process_raw_head(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"[{self.name}] Executando lógica Raw Default...")
        return new_processing._get_rings_default(dataframe)


class StdRingsDataPreparation(DataPreparationStrategy):
    """Estratégia para dados 'std_rings'."""
    name = 'std_rings'

    @data_handler("default")
    def standard_logic(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"[{self.name}] Executando lógica Standard Rings...")
        return dataframe.iloc[:, 0:10].head()


class QuarterRingsDataPreparation(DataPreparationStrategy):
    """Estratégia para dados 'quarter_rings'."""
    name = 'quarter_rings'

    @data_handler("default")
    def quarter_logic_main(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"[{self.name}] Executando Quarter Rings (default)...")
        dataframe=new_processing.normalize_qrings(dataframe, new_processing.layers)
        dataframe=new_processing.diff_qrings_energy(dataframe, new_processing.layers)
        dataframe=new_processing.vectorize_rings_energy(dataframe, new_processing.layers)
        dataframe=new_processing.vectorize_layer_energy(dataframe, new_processing.layers)
        print((dataframe.shape))
        return dataframe
    
    @data_handler("delta")
    def quarter_logic_delta(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        print(f"[{self.name}] Executando Quarter Rings (Delta)...")
        return dataframe.iloc[:, 0:5].head()


# --- Step 3: Create a Unified Builder/Facade Class ---
class DataBuilder:
    """
    Facade class that manages the state and orchestrates the entire process
    of data preparation and splitting.
    """
    def __init__(
            self, dataframe: pd.DataFrame, 
            bins_et: Dict[str, List], 
            bins_eta: Dict[str, List]):
        
        self._topology_factory = DataPreparationStrategy()
        self.dataframe = dataframe
        self.bins_eta = bins_eta
        self.bins_et = bins_et

        # State attributes, explicitly initialized
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def _set_alpha(self, alpha: Union[list, pd.Series, None]) -> pd.Series:
        """Private method to validate and format the target variable."""
        if alpha is None:
            if "alpha" not in self.dataframe.columns:
                raise ValueError("If 'alpha' is None, an 'alpha' column must exist in the dataframe.")
            return self.dataframe["alpha"]
        if isinstance(alpha, pd.Series):
            return alpha
        if isinstance(alpha, list):
            if len(alpha) != len(self.dataframe):
                raise ValueError(
                    f"The length of the alpha list ({len(alpha)}) must be equal to "
                    f"the length of the dataframe ({len(self.dataframe)})."
                )
            return pd.Series(alpha, index=self.dataframe.index, name='alpha')
        raise TypeError(f"Unsupported type for alpha: {type(alpha).__name__}")

    def _get_features(self, dataframe:pd.DataFrame, topology:str='std_rings', model:str='default', **kwargs) -> pd.DataFrame:
        """Private method to generate features using the factory and a strategy."""
        topology_class = self._topology_factory.create_topology(topology)
        # Ensures 'alpha' is not passed to the feature preparation step
        features_df = dataframe.drop(columns=['alpha'], errors='ignore')
        return topology_class.feature_modeling(features_df, model, **kwargs)

    def _split_data(self, dataframe: pd.DataFrame, features: pd.DataFrame, train_size: float, random_state: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Private method for splitting the data."""
        # Ensures that the index of y (alpha) is aligned with X (features)
        aligned_target = dataframe['alpha'].loc[features.index]
        # print('aligned', aligned_target)
        # print('out aligned')
        return sklearn.model_selection.train_test_split(
            features,
            aligned_target,
            train_size=train_size,
            random_state=random_state
        )
    
    def _split_by_bins(self, bins_size=None):
        """
        Splits a DataFrame into multiple DataFrames based on a list of conditions.
        """
        if bins_size is None:
            self.dataframe = [self.dataframe]

        else:
            result_list = []
            data_remaining = self.dataframe.copy()

            for group_conditions in bins_size:
                masks = []
                for condition_dict in group_conditions:
                    for col_name, intervals in condition_dict.items():
                        masks.append(self._make_mask(data_remaining[col_name], intervals))

                if masks:
                    combined_mask = np.logical_and.reduce(masks)
                else:
                    combined_mask = np.zeros(len(data_remaining), dtype=bool)

                filtered = data_remaining.loc[combined_mask]
                result_list.append(filtered)
                data_remaining = data_remaining.loc[~combined_mask]

                result_list.append(data_remaining)
            self.dataframe = result_list



    def _make_mask(self, column, values):
        """
        Creates a boolean mask from a pandas Series.
        """
        values = list(values)
        if len(values) == 2 and all(isinstance(v, (int, float, type(None))) for v in values):
            min_val, max_val = values
            target_column = column.abs() if column.name == 'cluster_eta' else column
            
            if min_val is not None and max_val is not None:
                mask = (target_column >= min_val) & (target_column < max_val)
            elif min_val is not None:
                mask = target_column >= min_val
            elif max_val is not None:
                mask = target_column < max_val
            else:
                mask = pd.Series(True, index=column.index)
        else:
            mask = column.isin(values)
        return mask.to_numpy()
    
    def _get_bins_alias(self, bins_key: str) -> List[List[Dict[str, List[float]]]]:
        """
        Selects the binning generation method based on the provided key.
        
        Using a mapping of function references (lazy evaluation) ensures that 
        we only execute the required logic, saving processing time.
        """
        # Map keys to the method references (without calling them yet)
        bins_map: Dict[str, Callable[[], List]] = {
            'all': self._generate_bins_combinations,
            # 'signal': self._generate_signal_bins,  # Example for future expansion
        }

        if bins_key not in bins_map:
            valid_keys = list(bins_map.keys())
            raise ValueError(f"Key '{bins_key}' not recognized. Available options: {valid_keys}")

        # Execute the selected method
        return bins_map[bins_key]()

    def _generate_bins_combinations(self):
        """
        Generates a list of interval combinations based on the provided bin edges.
        
        Args:
            bins_et (list): List of edge values for transverse energy (Et).
            bins_eta (list): List of edge values for pseudorapidity (Eta).
            
        Returns:
            list: A list of lists, where each sub-list contains dictionaries 
                with the Eta and Et intervals.
        """
        params = []
        et_values = self.bins_et['cluster_et']
        eta_values = self.bins_eta['cluster_eta']
        # Iterate over each eta interval (e.g., [0, 0.6], [0.6, 0.8], ...)
        # The range goes up to len - 1 because we always take the current index and the next one
        for i in range(len(eta_values) - 1):
            eta_interval = [eta_values[i], eta_values[i+1]]
            
            # For each eta interval, iterate over each et interval
            for j in range(len(et_values) - 1):
                et_interval = [et_values[j], et_values[j+1]]
                
                # Create the dictionary structure for the current combination
                param_combination = [
                    {'cluster_eta': eta_interval},
                    {'cluster_et': et_interval}
                ]
                
                # Add the combination to the final list
                params.append(param_combination)
                
        return params

    def run(
        self,
        topology: Literal['raw', 'std_rings', 'quarter_rings'],
        train_size: float = 0.7,
        random_state: Optional[int] = None,
        model: Optional[str] = 'default',
        bins_size = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Orchestrates feature generation and data splitting, updating the builder's state.
        """

        "check se dataframe é array se não for transformar. executar run com um for aqui dentreo que serra append em result"
        result=[]

        if isinstance(bins_size, str):
            bins_size=self._get_bins_alias(bins_size)

        self._split_by_bins(bins_size)
        for dataframe in self.dataframe:
            # 1. Generate features
            # print('dentro do loop',dataframe)
            # print('linha seguinte')
            features = self._get_features(dataframe=dataframe,topology=topology, model=model, **kwargs)

            # 2. Split data
            self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(
                dataframe=dataframe,
                features=features,
                train_size=train_size,
                random_state=random_state
            )

            # 3. Update the builder's state
            result.append([self.X_train, self.X_test, self.y_train, self.y_test])
        return result


# --- Utility functions ---

def _set_data_to_plot(
    df1: pd.DataFrame = None, # X_test
    s: pd.Series = None,      # y_test
    arr: np.ndarray= None,   # y_pred
    cols: list[str] = ['cluster_eta', 'cluster_et']
) -> pd.DataFrame:
    """
    Merge selected columns from a DataFrame with a Series and a NumPy ndarray 
    into a single DataFrame (side by side concatenation).

    Parameters
    ----------
    df1 : pd.DataFrame
        Input DataFrame containing the columns of interest.
    s : pd.Series
        Series to be added as a new column in the final DataFrame.
    arr : np.ndarray
        NumPy array to be added as one or multiple columns in the final DataFrame.
    cols : list[str], optional
        List of column names to be selected from df1. 
        Default is ['cluster_eta', 'cluster_et'].

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the selected columns from df1, the Series, 
        and the ndarray as additional columns.
    """

    # Select only the required columns from the first DataFrame
    df_sel: pd.DataFrame = df1[cols]

    # Convert Series to DataFrame (keep name if available)
    s_df: pd.DataFrame = s.to_frame(name=s.name if s.name else "series_col")

    # Convert ndarray to DataFrame
    if arr.ndim == 1:
        # If 1D, single column
        arr_df: pd.DataFrame = pd.DataFrame(arr, columns=["y_pred"])
    else:
        # If 2D, generate numbered column names
        arr_df: pd.DataFrame = pd.DataFrame(
            arr, 
            columns=[f"y_pred_{i}" for i in range(arr.shape[1])]
        )

    # Concatenate all DataFrames side by side
    final_df: pd.DataFrame = pd.concat(
        [df_sel.reset_index(drop=True),
         s_df.reset_index(drop=True),
         arr_df.reset_index(drop=True)],
        axis=1
    )

    return final_df

def _get_test_data_with_bins(dataframe:pd.DataFrame, y_test:pd.Series):
    # retun X_test values with all columns, once methode _get_features remove some features
    return dataframe.loc[y_test.index]