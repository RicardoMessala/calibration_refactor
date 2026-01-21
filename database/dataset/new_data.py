import pandas as pd
import numpy as np
import sklearn.model_selection
from typing import List, Tuple, Optional, Union, Literal, Dict, Type, Callable
from database.dataset import new_processing
from abc import ABC, abstractmethod

# Assuming 'dataset_preprocessing' is an imported module
# import dataset_preprocessing

# --- Step 1: Define Strategies (Concrete Products) ---
class DataPreparationStrategy(ABC):
    """Interface for all data preparation strategies."""
    @abstractmethod
    def feature_modeling(self, df: pd.DataFrame, model:str = 'default', **kwargs) -> pd.DataFrame:
        """Prepares the data according to a specific strategy."""
        pass

class RawDataPreparation(DataPreparationStrategy):
    """Strategy for preparing data in the 'raw' format."""
    def feature_modeling(self, df: pd.DataFrame, model:str = 'default', **kwargs) -> pd.DataFrame:
        print("Processing raw data...")
        # Original logic would go here
        return df[['cluster_et', 'cluster_eta']].head() # Mock implementation

class StdRingsDataPreparation(DataPreparationStrategy):
    """Strategy for preparing data in the 'std_rings' format."""
    def __init__(self):
        self._model_handlers: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
            'default': self._default_model,

            # More models can be registered here in the future
        }

    def feature_modeling(self, df: pd.DataFrame, model:str = 'default', **kwargs) -> pd.DataFrame:
        """Prepares the data by dispatching to the correct model handler."""
        handler = self._model_handlers.get(model)
        if not handler:
            raise ValueError(f"Unknown model '{model}' for StdRingsDataPreparation. "
                             f"Valid models are {list(self._model_handlers.keys())}.")
        return handler(df, **kwargs)

    def _default_model(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Handles the default preparation logic for std_rings."""
        # Original logic would go here
        return df.iloc[:, 0:10].head() # Mock implementation

class QuarterRingsDataPreparation(DataPreparationStrategy):
    """Strategy for preparing data in the 'quarter_rings' format."""
    def __init__(self):
        self._model_handlers: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
            'default': self._default_model,
            'delta': self._delta_model,

            # More models can be registered here in the future
        }

    def feature_modeling(self, df: pd.DataFrame,  model:str = 'default', **kwargs) -> pd.DataFrame:
        """Prepares the data by dispatching to the correct model handler."""
        handler = self._model_handlers.get(model)
        if not handler:
            raise ValueError(f"Unknown model '{model}' for QuarterRingsDataPreparation. "
                             f"Valid models are {list(self._model_handlers.keys())}.")
        return handler(df, **kwargs)

    def _default_model(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Handles the default preparation logic for std_rings."""
        # Original logic would go here
        return df.iloc[:, 0:10].head() # Mock implementation
    
    def _delta_model(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Handles the 'delta' preparation logic for quarter_rings."""
        # Original logic would go here
        return df.iloc[:, 0:10].head() # Mock implementation

# --- Step 2: Create a Simplified Factory ---
class DataPreparationFactory:
    """
    A Simple Factory that creates the correct data preparation strategy.
    It uses a dictionary to be easily extensible (following the Open/Closed Principle).
    """
    def __init__(self):
        self._topologies: Dict[str, Type[DataPreparationStrategy]] = {
            'raw': RawDataPreparation,
            'std_rings': StdRingsDataPreparation,
            'quarter_rings': QuarterRingsDataPreparation,
        }

    def register_topology(self, topology: str, topology_class: Type[DataPreparationStrategy]):
        """Allows for dynamically registering new strategies."""
        self._topologies[topology] = topology_class

    def set_topology(self, topology: str) -> DataPreparationStrategy:
        """
        Creates an instance of the requested strategy.
        Additional arguments (kwargs) are passed to the strategy's constructor.
        """
        topology_class = self._topologies.get(topology)
        if not topology_class:
            raise ValueError(
                f"Unknown input type: '{topology}'. "
                f"Valid values are {list(self._topologies.keys())}."
            )
        # Passes arguments (e.g., model) to the strategy's constructor
        return topology_class()

# --- Step 3: Create a Unified Builder/Facade Class ---
class DataBuilder:
    """
    Facade class that manages the state and orchestrates the entire process
    of data preparation and splitting.
    """
    def __init__(self, dataframe: pd.DataFrame, alpha: Union[list, pd.Series, None] = None):
        self.dataframe = dataframe
        self.alpha = self._set_alpha(alpha) # Defines the target only once
        self._topology_factory = DataPreparationFactory()

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

    def _get_features(self, topology: str, model:str='default', **kwargs) -> pd.DataFrame:
        """Private method to generate features using the factory and a strategy."""
        topology_class = self._topology_factory.set_topology(topology)
        # Ensures 'alpha' is not passed to the feature preparation step
        features_df = self.dataframe.drop(columns=['alpha'], errors='ignore')
        return topology_class.feature_modeling(features_df, model, **kwargs)

    def _split_data(self, features: pd.DataFrame, train_size: float, random_state: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Private method for splitting the data."""
        # Ensures that the index of y (alpha) is aligned with X (features)
        aligned_target = self.alpha.loc[features.index]
        return sklearn.model_selection.train_test_split(
            features,
            aligned_target,
            train_size=train_size,
            random_state=random_state
        )

    def run(
        self,
        topology: Literal['raw', 'std_rings', 'quarter_rings',],
        train_size: float = 0.7,
        random_state: Optional[int] = None,
        model: Optional[str] = 'default',
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Orchestrates feature generation and data splitting, updating the builder's state.
        """
        # 1. Generate features
        features = self._get_features(topology=topology, model=model, **kwargs)

        # 2. Split data
        result = self._split_data(
            features=features,
            train_size=train_size,
            random_state=random_state
        )

        # 3. Update the builder's state
        self.X_train, self.X_test, self.y_train, self.y_test = result
        return result


# --- Utility functions ---

def _split_by_bins(data, params=None):
    """
    Splits a DataFrame into multiple DataFrames based on a list of conditions.
    """
    if params is None:
        return [data]
    
    result_list = []
    data_remaining = data.copy()

    for group_conditions in params:
        masks = []
        for condition_dict in group_conditions:
            for col_name, intervals in condition_dict.items():
                masks.append(_make_mask(data_remaining[col_name], intervals))

        if masks:
            combined_mask = np.logical_and.reduce(masks)
        else:
            combined_mask = np.zeros(len(data_remaining), dtype=bool)

        filtered = data_remaining.loc[combined_mask]
        result_list.append(filtered)
        data_remaining = data_remaining.loc[~combined_mask]

    result_list.append(data_remaining)
    return result_list


def _make_mask(column, values):
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