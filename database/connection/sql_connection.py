import sqlite3
import pandas as pd

def extrair_dados_sqlite(file_path, table_name):
    # Conectando ao arquivo SQLite

    conn = sqlite3.connect(file_path)
        
    # Definindo a consulta SQL que deseja executar
    sql_query = f"SELECT * FROM {table_name}"
        
    # Executando a consulta SQL e carregando os resultados em um DataFrame
    dataframe = pd.read_sql_query(sql_query, conn)
        
    # Fechando a conexão com o banco de dados
    conn.close()
        
    return dataframe


def get_standard_rings_data():
        return extrair_dados_sqlite(r'D:\Doutorado\calibration\calibration_refactor\calibration_refactor_main\database\data\StdRings.db', 'events')
           
def get_quarter_rings_data():
        return extrair_dados_sqlite(r'D:\Doutorado\calibration\calibration_refactor\calibration_refactor_main\database\data\QRings.db', 'events')

def get_relevant_data():
        return extrair_dados_sqlite(r'D:\Doutorado\calibration\calibration_refactor\calibration_refactor_main\database\data\ClusterData.db', 'events')

# Carregando banco de dados - Alterar APENAS o PATH para o arquivo
def set_standard_rings_data():
    return pd.concat([get_standard_rings_data(), get_relevant_data()], axis=1)

def set_quarter_rings_data():
    return pd.concat([get_quarter_rings_data(), get_relevant_data()], axis=1)