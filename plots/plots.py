import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict, Callable, Optional, TypedDict, NotRequired
from scipy import stats
from matplotlib.axes import Axes

def merge_dataframes(
    df1: pd.DataFrame, # X_test
    s: pd.Series,      # y_test
    arr: np.ndarray,   # y_pred
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

def parameters_filter(
    data: pd.DataFrame,
    bins: dict,
    cols: Union[str, List[str], None] = None
) -> Union[List[pd.DataFrame], List[pd.Series]]:
    """
    Filters a DataFrame into bins based on specified column intervals.
    Each filtering step removes the selected rows from the dataset
    to avoid duplicates across intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to filter.
    bins : dict
        Dictionary where keys are column names and values are lists of interval edges.
    cols : str, list[str], or None
        - If None: all columns are returned.
        - If list[str]: only those columns are kept in the resulting DataFrames.
        - If str: only that column is returned, as Series.

    Returns
    -------
    list[pd.DataFrame] or list[pd.Series]
        List of filtered objects according to the bins.
    """
    # Lista que armazenará os resultados
    result: List[Union[pd.DataFrame, pd.Series]] = []

    # Cópia do DataFrame para remoção progressiva das linhas filtradas
    data_copy: pd.DataFrame = data.copy()

    # Itera pelas colunas e intervalos definidos no dicionário bins
    for col, intervals in bins.items():
        filtered_df: pd.DataFrame
        mask: pd.Series

        for i in range(1, len(intervals)):
            # Criação da máscara de filtragem
            if col == "cluster_eta":
                mask = (abs(data_copy[col]) <= intervals[i]) & (abs(data_copy[col]) > intervals[i - 1])
            else:
                mask = (data_copy[col] <= intervals[i]) & (data_copy[col] > intervals[i - 1])

            # Aplica a máscara e reseta os índices
            filtered_df = data_copy[mask].dropna().reset_index(drop=True)

            if not filtered_df.empty:
                # Seleciona apenas as colunas desejadas
                if isinstance(cols, str):
                    result.append(filtered_df[cols])
                elif isinstance(cols, list):
                    result.append(filtered_df[cols])
                else:
                    result.append(filtered_df)

                # Remove os registros filtrados do DataFrame copiado
                data_copy = data_copy.drop(filtered_df.index).reset_index(drop=True)

    return result


def calculate_iqr(df: pd.DataFrame, y_test_col: str, y_pred_col: str) -> float:
    """
    Calculates the IQR of the test column or the ratio between test and prediction.

    Args:
        df (pd.DataFrame): The input DataFrame.
        y_test_col (str): The name of the ground truth column.
        y_pred_col (str): The name of the prediction column.

    Returns:
        float: The calculated Interquartile Range.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The provided item is not a DataFrame.")
    if y_test_col not in df.columns:
        raise ValueError(f"Column '{y_test_col}' not found.")

    if y_pred_col not in df.columns:
        # Logic for when y_pred is not present
        target_data = df[y_test_col]
    else:
        # Logic for the ratio of y_test / y_pred
        ratio = df[y_test_col] / df[y_pred_col]
        target_data = ratio.replace([np.inf, -np.inf], np.nan)
        
    return stats.iqr(target_data, axis=None, rng=(25, 75), scale=1.0, nan_policy='omit', interpolation='linear')

def calculate_rmse(df: pd.DataFrame, y_test_col: str, y_pred_col: str) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE).

    Args:
        df (pd.DataFrame): The input DataFrame.
        y_test_col (str): The name of the ground truth column.
        y_pred_col (str): The name of the prediction column.

    Returns:
        float: The calculated RMSE value.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The provided item is not a DataFrame.")
    if y_test_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError(f"For RMSE, both columns '{y_test_col}' and '{y_pred_col}' are required.")
    
    squared_errors = (df[y_test_col] - df[y_pred_col]) ** 2
    return np.sqrt(squared_errors.mean())

def calculate_mae(df: pd.DataFrame, y_test_col: str, y_pred_col: str) -> float:
    """
    Calculates the Mean Absolute Error (MAE).

    Args:
        df (pd.DataFrame): The input DataFrame.
        y_test_col (str): The name of the ground truth column.
        y_pred_col (str): The name of the prediction column.

    Returns:
        float: The calculated MAE value.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The provided item is not a DataFrame.")
    if y_test_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError(f"For MAE, both columns '{y_test_col}' and '{y_pred_col}' are required.")
        
    absolute_errors = (df[y_test_col] - df[y_pred_col]).abs()
    return absolute_errors.mean()

# Define the type for our metric functions for clarity
MetricCallable = Callable[[pd.DataFrame, str, str], float]

# This dictionary acts as a registry to select the desired metric function.
METRIC_FUNCTIONS: Dict[str, MetricCallable] = {
    'iqr': calculate_iqr,
    'rmse': calculate_rmse,
    'mae': calculate_mae,
}

# --- Step 3: Create the Main Orchestrator Function ---

def evaluate_metrics(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    metric: str = 'iqr',
    y_test_col: str = 'y_test',
    y_pred_col: str = 'y_pred'
) -> List[Optional[float]]:
    """
    Calculates a specific metric for one or more DataFrames.

    Args:
        data (Union[pd.DataFrame, List[pd.DataFrame]]): A single DataFrame or a list
            of DataFrames to process.
        metric (str, optional): The name of the metric to calculate.
            Defaults to 'iqr'. Available metrics are keys in METRIC_FUNCTIONS.
        y_test_col (str, optional): The name of the ground truth column.
            Defaults to 'y_test'.
        y_pred_col (str, optional): The name of the prediction column.
            Defaults to 'y_pred'.

    Returns:
        List[Optional[float]]: A list containing the metric results for each DataFrame.
                               Returns None for any DataFrame that causes an error.
    """
    # 1. Select the metric function from the registry
    if metric not in METRIC_FUNCTIONS:
        available_metrics = ", ".join(METRIC_FUNCTIONS.keys())
        raise ValueError(f"Metric '{metric}' not recognized. Available options are: {available_metrics}")
    
    metric_func = METRIC_FUNCTIONS[metric]

    # 2. Standardize the input to always be a list for consistent processing
    if isinstance(data, pd.DataFrame):
        data_list = [data]
    elif isinstance(data, list):
        data_list = data
    else:
        raise TypeError("The 'data' parameter must be a DataFrame or a list of DataFrames.")

    # 3. Iterate and apply the selected metric function to each DataFrame
    results: List[Optional[float]] = []
    for i, df in enumerate(data_list):
        try:
            # The call is now dynamic, using the function we selected from the dictionary!
            value = metric_func(df, y_test_col=y_test_col, y_pred_col=y_pred_col)
            results.append(value)
        except (ValueError, TypeError) as e:
            print(f"WARNING: Error calculating '{metric}' for DataFrame at index {i}: {e}")
            results.append(None)
            
    return results

# A type alias to handle both lists and dictionaries for bin edges
BinsType = Union[List[Union[float, int]], Dict[str, List[Union[float, int]]]]

# Define a specific type for the plot configuration dictionary for clarity
class PlotConfig(TypedDict):
    """A dictionary defining a plot series with required and optional keys."""
    y_data: Union[list, np.ndarray]  # The y-values for the plot (Required)
    label: NotRequired[str]             # The legend label for the series (Optional)
    fmt: NotRequired[str]               # The marker format string (Optional)
    color: NotRequired[str]             # The color of the series (Optional)

def calculate_bin_centers(bins: BinsType) -> List[float]:
    """
    Calculates the center point for each interval defined by bin edges.

    Args:
        bins (BinsType): A list of bin edges (e.g., [0, 5, 10, 15]) or a
                         dictionary containing such a list as a value.

    Returns:
        List[float]: A list of the calculated center points for each bin.
    """
    if isinstance(bins, dict):
        if not bins:
            raise ValueError("The 'bins' dictionary cannot be empty.")
        # Use the first list of edges found in the dictionary's values
        bin_edges = next(iter(bins.values()))
    elif isinstance(bins, list):
        bin_edges = bins
    else:
        raise TypeError("The 'bins' parameter must be a list of edges or a dictionary.")
    
    return [(bin_edges[i] + bin_edges[i-1]) / 2 for i in range(1, len(bin_edges))]

def calculate_bin_half_widths(bins: BinsType) -> List[float]:
    """
    Calculates half the width of each bin from a list of bin edges.

    Args:
        bins (BinsType): A list of bin edges (e.g., [0, 5, 10, 15]) or a
                         dictionary containing such a list as a value.

    Returns:
        List[float]: A list of the calculated half-widths for each bin.
    """
    if isinstance(bins, dict):
        if not bins:
            raise ValueError("The 'bins' dictionary cannot be empty.")
        # Use the first list of edges found in the dictionary's values
        bin_edges = next(iter(bins.values()))
    elif isinstance(bins, list):
        bin_edges = bins
    else:
        raise TypeError("The 'bins' parameter must be a list of edges or a dictionary.")

    return [(bin_edges[i] - bin_edges[i-1]) / 2 for i in range(1, len(bin_edges))]

def plot_errorbars(
    plot_configs: List[PlotConfig],
    bins: BinsType,
    title: str = "Figure Title",
    y_label: str = "Y-axis Value",
    x_label: str = "X-axis Value",
    legend_fontsize: int = 8,
    xscale: str = 'linear',
    yscale: str = 'linear',
) -> None:
    """
    Creates a single-panel plot with error bars.

    Args:
        plot_configs (List[PlotConfig]): Configuration for each series to plot.
        bins (BinsType): A list of bin edges or a dictionary containing one.
        title (str, optional): The title for the plot.
        y_label (str, optional): The y-axis label for the plot.
        x_label (str, optional): The x-axis label.
        legend_fontsize (int, optional): The font size for the legend.
        xscale (str, optional): The scale for the x-axis (e.g., 'linear', 'log'). Defaults to 'linear'.
        yscale (str, optional): The scale for the y-axis (e.g., 'linear', 'log'). Defaults to 'linear'.
    """
    # 0. Calculate x-axis data and errors from the list of bin edges
    x_data = calculate_bin_centers(bins)
    x_err = calculate_bin_half_widths(bins)

    # 1. Set up the figure with a single subplot
    fig, ax = plt.subplots()
    ax: Axes = ax
    
    # 2. Loop through the configurations and plot each series
    for config in plot_configs:
        y_data = np.asarray(config['y_data'])
        if len(y_data) != len(x_data):
            raise ValueError(f"Length of y_data ({len(y_data)}) for label '{config.get('label')}' does not match number of bins ({len(x_data)}).")
            
        label = config.get('label', 'Series')
        fmt = config.get('fmt', 'o')
        color = config.get('color', None)
        
        plot_kwargs = {'fmt': fmt, 'label': label}
        if color:
            plot_kwargs['color'] = color
        
        ax.errorbar(x_data, y_data, xerr=x_err, **plot_kwargs)

    # 3. Set general plot aesthetics
    ax.legend(loc="best", fontsize=legend_fontsize)
    ax.grid(True, which="both", linestyle='--')
    ax.set_title(title)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xlabel(x_label, fontsize=16)
    
    # Set axis scales
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    fig.tight_layout(pad=1.0)
    plt.show()