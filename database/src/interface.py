import sklearn
import pandas as pd
import numpy as np
import sklearn.model_selection
from database.src import dataset_preprocessing
from database.connection import sql_connection

def set_data(data):
    # Se não for lista, transforma em lista
    if not isinstance(data, list):
        data = [data]
    
    for item in data:
       #usar as funções de split data
       #algumacoisa.append()
       pass
    #return algumacoisa

def executar_algoritmos(data, params):
    if not isinstance(data, list):
        data = [data]

    for item in data:  
        #executar o algoritmo com os parametros - Criar a classe de seleção para os algoritmos
        #append reseultados do algoritmo
        pass
    #return resultados com append

#criar classe seletora de algortimos

class MasterFactory:
    def __init__(self):
        self._optimazation = {
                "cpu": CPU,
                "gpu": GPU,
            }

        self.calibration = {
                "os": OperatingSystem,
                "antivirus": Antivirus,
            }

    def create_product(self, optimization_name, calibration_name, **kwargs):

        optimization_name = optimization_name.lower()
        calibration_name = calibration_name.lower()

        # Step 1: Find the sub-factory (the optimization's dictionary)
        tmp_optimization = self._optimazation.get(optimization_name)
        if not tmp_optimization:
            raise ValueError(f"Unknown product optimization: '{optimization_name}'")

        # Step 2: Find the class within the sub-factory
        tmp_calibration = self.calibration.get(calibration_name)
        if not tmp_calibration:
            raise ValueError(f"Unknown product calibration '{calibration_name}' in optimization '{optimization_name}'")

        # Step 3: Create the instance with the kwargs
        print(f"\nFactory: Creating '{calibration_name}' from the '{optimization_name}' optimization...")
        return tmp_optimization(tmp_calibration, **kwargs)