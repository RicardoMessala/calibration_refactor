import sklearn
import pandas as pd
import numpy as np
import sklearn.model_selection
from database.src import dataset_preprocessing
from database.connection import sql_connection
from database.src.dataset_config import CreateInput

# # External caching of data
# _relevant_data_cache = sql_connection.get_relevant_data()
# _quarterrings_data_cache = sql_connection.get_quarter_rings_data()
# _stdrings_data_cache = sql_connection.get_standard_rings_data()
# _quarterrings_relevant_cache = sql_connection.set_quarter_rings_data()
# _stdrings_relevant_cache = sql_connection.set_standard_rings_data()

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

        return self.LoopBuilder(wrapper_data, params).split_dataframe_by_groups()


    def set_inputs(self, ):
        

        # meter o laço do for chamando o create inputs aqui
        
        pass
    def input_builder(self):

        #Fazer o construtor
        pass