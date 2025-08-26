import gbdt
#import transformers
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import modules.calibration_algorithm.gbdt
from modules.calibration_algorithm.gbdt import GBDTTrainning, TransformersTrainning
from database.src.dataset_config import CreateInput

class AlgorithmTrainingFactory:
    def __init__(self, algorithm ,params, kwarg):
        self.algorithm=algorithm
        self.params=params
        self.kwargs=kwarg

    def set_algorithm(self):

        if self.algorithm == 'gbt': 
            return GBDTTrainning(self.params, **self.kwargs)
        
        elif self.algorithm == 'transformers':
            return TransformersTrainning(self.params,self.kwargs)
        
        else:
            raise ValueError(f"Unknown algorithm name'{self.algorithm}'.")
        
class AlgorithmPredictionFactory:
    def __init__(self, algorithm ,params, kwarg):
        self.algorithm=algorithm
        self.params=params
        self.kwargs=kwarg

    def set_algorithm(self):

        if self.algorithm == 'gbt': 
            return GBDTTrainning(self.params, **self.kwargs)
        
        elif self.algorithm == 'transformers':
            return TransformersTrainning(self.params,self.kwargs)
        
        else:
            raise ValueError(f"Unknown algorithm name'{self.algorithm}'.")

class LoopBuilder:
    def __init__(self, data, params):
        self.data = data
        self.params = params

    def check_repetitiond(self, vector):

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

    def split_dataframe(df, col1, col2, params):
        """
        Split a DataFrame into multiple DataFrames according to filters in params.
        Rows are removed after each filter to avoid overlaps.
        Remaining rows form the 'remainder' DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The original DataFrame.
        col1 : str
            First column name to filter.
        col2 : str
            Second column name to filter.
        params : list of [list, list]
            Each element is [values_for_col1, values_for_col2].

        Returns
        -------
        list
            List of filtered DataFrames (one for each parameter).
        pandas.DataFrame
            Remainder DataFrame containing rows not captured by any filter.
        """
        df_remaining = df.copy()
        groups = []

        for vals1, vals2 in params:
            mask = df_remaining[col1].isin(vals1) & df_remaining[col2].isin(vals2)
            filtered = df_remaining[mask].reset_index(drop=True)
            groups.append(filtered)
            df_remaining = df_remaining[~mask]

        remainder = df_remaining.reset_index(drop=True)
        return groups, remainder