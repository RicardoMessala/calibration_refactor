import pandas as pd
import numpy as np
import sklearn.model_selection
from typing import List, Tuple, Optional, Union, Literal, Dict, Type, Callable
from abc import ABC, abstractmethod

# Assuming 'dataset_preprocessing' is an imported module
# import dataset_preprocessing

# --- Step 1: Define Strategies (Concrete Products) ---
class DataPreparationStrategy(ABC):
    """Interface for all data preparation strategies."""
    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the data according to a specific strategy."""
        pass

class RawDataPreparation(DataPreparationStrategy):
    """Strategy for preparing data in the 'raw' format."""
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Processing raw data...")
        # Original logic would go here
        return df[['cluster_et', 'cluster_eta']].head() # Mock implementation

class StdRingsDataPreparation(DataPreparationStrategy):
    """Strategy for preparing data in the 'std_rings' format."""
    def __init__(self, mode: str = 'default'):
        self.mode = mode
        self._mode_handlers: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
            'default': self._prepare_default,
            # More modes can be registered here in the future
        }

    def _prepare_default(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles the default preparation logic for std_rings."""
        print(f"Processing standard rings data with mode: '{self.mode}'...")
        # Original logic would go here
        return df.iloc[:, 0:10].head() # Mock implementation

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the data by dispatching to the correct mode handler."""
        handler = self._mode_handlers.get(self.mode)
        if not handler:
            raise ValueError(f"Unknown mode '{self.mode}' for StdRingsDataPreparation. "
                             f"Valid modes are {list(self._mode_handlers.keys())}.")
        return handler(df)

class QuarterRingsDataPreparation(DataPreparationStrategy):
    """Strategy for preparing data in the 'quarter_rings' format."""
    def __init__(self, mode: str = 'delta'):
        if not mode:
            raise ValueError("Mode is required for QuarterRingsDataPreparation.")
        self.mode = mode
        self._mode_handlers: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
            'delta': self._prepare_delta,
            # More modes can be registered here in the future
        }

    def _prepare_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles the 'delta' preparation logic for quarter_rings."""
        print(f"Processing quarter rings data with mode: '{self.mode}'")
        # Original logic would go here
        return df.iloc[:, 0:10].head() # Mock implementation

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares the data by dispatching to the correct mode handler."""
        handler = self._mode_handlers.get(self.mode)
        if not handler:
            raise ValueError(f"Unknown mode '{self.mode}' for QuarterRingsDataPreparation. "
                             f"Valid modes are {list(self._mode_handlers.keys())}.")
        return handler(df)

# --- Step 2: Create a Simplified Factory ---
class DataPreparationFactory:
    """
    A Simple Factory that creates the correct data preparation strategy.
    It uses a dictionary to be easily extensible (following the Open/Closed Principle).
    """
    def __init__(self):
        self._strategies: Dict[str, Type[DataPreparationStrategy]] = {
            'raw': RawDataPreparation,
            'std_rings': StdRingsDataPreparation,
            'quarter_rings': QuarterRingsDataPreparation,
        }

    def register_strategy(self, name: str, strategy_class: Type[DataPreparationStrategy]):
        """Allows for dynamically registering new strategies."""
        self._strategies[name] = strategy_class

    def create_preparer(self, input_type: str, **kwargs) -> DataPreparationStrategy:
        """
        Creates an instance of the requested strategy.
        Additional arguments (kwargs) are passed to the strategy's constructor.
        """
        strategy_class = self._strategies.get(input_type)
        if not strategy_class:
            raise ValueError(
                f"Unknown input type: '{input_type}'. "
                f"Valid values are {list(self._strategies.keys())}."
            )
        # Passes arguments (e.g., mode) to the strategy's constructor
        return strategy_class(**kwargs)

# --- Step 3: Create a Unified Builder/Facade Class ---
class DataBuilder:
    """
    Facade class that manages the state and orchestrates the entire process
    of data preparation and splitting.
    """
    def __init__(self, dataframe: pd.DataFrame, alpha: Union[list, pd.Series, None] = None):
        self.dataframe = dataframe
        self.alpha = self._set_alpha(alpha) # Defines the target only once
        self._factory = DataPreparationFactory()

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

    def _get_features(self, input_type: str, **kwargs) -> pd.DataFrame:
        """Private method to generate features using the factory and a strategy."""
        preparer = self._factory.create_preparer(input_type, **kwargs)
        # Ensures 'alpha' is not passed to the feature preparation step
        features_df = self.dataframe.drop(columns=['alpha'], errors='ignore')
        return preparer.prepare(features_df)

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
        input_type: Literal['raw', 'std_rings', 'quarter_rings'],
        train_size: float = 0.7,
        random_state: Optional[int] = None,
        mode: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Orchestrates feature generation and data splitting, updating the builder's state.
        """
        # Collects strategy-specific arguments (kwargs)
        strategy_kwargs = {}
        # This is now generic and works for any strategy that accepts a 'mode' argument.
        if mode:
            strategy_kwargs['mode'] = mode

        # 1. Generate features
        features = self._get_features(input_type=input_type, **strategy_kwargs)

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
        return [data.reset_index(drop=True)]
    
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
        result_list.append(filtered.reset_index(drop=True))
        data_remaining = data_remaining.loc[~combined_mask]

    result_list.append(data_remaining.reset_index(drop=True))
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