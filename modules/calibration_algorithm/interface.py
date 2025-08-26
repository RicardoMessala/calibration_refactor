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
