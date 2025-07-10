from modules.calibration_algorithm.algorithm_interface import AlgorithmFactory
import numpy as np

vetor_de_matrizes = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
])

class CalibrationInput:

    def __init__(self):
        pass
    
    def set_dataset_variables(self):
        pass

    def set_input_variables(self):
        # Loop through each matrix
        for i, matrix in enumerate(vetor_de_matrizes):
            print(f"Matrix {i}:")
            print(matrix)
            print("---")


class PlotSetUp:
    pass

class TudoJunto:
    pass
#Juntar otimização + algotirmo + plot - arquivo chamado pela main