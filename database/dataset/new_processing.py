import pandas as pd
import numpy as np
from typing import List, Optional

# Metodos para atributos dos rings - Apenas criar novas def
# Function to compute the relevant variables

def calc_sv(e237, e277, rings):
	Reta = e237/e277
	E_E1E2 = rings.iloc[:,8:72].sum(axis=1)/rings.iloc[:,72:80].sum(axis=1)
	E_EM = rings.iloc[:,8:88].sum(axis=1)
	E_E0E1 = rings.iloc[:,0:8].sum(axis=1)/rings.iloc[:,8:72].sum(axis=1)
	RHad = rings.iloc[:,88:92].sum(axis=1)/E_EM #E_H1E

	return Reta, E_E1E2, E_EM, E_E0E1, RHad

# Novo def CalcRings(rings, cluster_eta, cluster_phi, delta_eta_calib, delta_phi_calib, hotCellEta, hotCellPhi,):
# será usado tanto pra o defaults do std rings quanto do quarter rings

def _get_rings_default(dataframe: pd.DataFrame, columns: Optional[List[List[str]]] = None) -> pd.DataFrame:
    """
    Processa o DataFrame 'data' selecionando colunas com base em uma lista
    de nomes de colunas. Se 'rings_columns' for None ou uma lista vazia, 
    retorna o DataFrame original.

    Args:
        data (pd.DataFrame): O DataFrame de entrada.
        rings_columns (Optional[List[List[str]]]): Uma lista de listas, onde cada
                                                     sublista contém os nomes das
                                                     colunas a serem selecionadas.
                                                     O padrão é None.

    Returns:
        pd.DataFrame: Um novo DataFrame com as colunas selecionadas, ou o
                      DataFrame original se rings_columns não for fornecido.
    """
    # Verifica se rings_columns é None ou uma lista vazia.
    # Se for o caso, retorna o DataFrame original sem modificações.
    if not columns:
        print('new_processing_print',list(dataframe.columns))
        return dataframe

    # Achata a lista de listas em uma única lista com todos os nomes das colunas.
    columns_to_select = [name for group in columns for name in group]
    
    # Seleciona todas as colunas de uma vez usando seus nomes.
    # Isso é mais eficiente do que selecionar e concatenar partes separadas.
    return dataframe[columns_to_select]
   
def calc_asym_weights_delta(
    quarter_rings_original: pd.DataFrame,
    std_rings_original: pd.DataFrame,
    rings_columns: List[List[str]] = None,
    clusters_columns: List[List[str]] = None
) -> pd.DataFrame:
    
    # 1. Filtragem dos dados (esta parte está correta)
    filtered_qr = _get_rings_default(dataframe=quarter_rings_original, columns=rings_columns)
    filtered_std = _get_rings_default(dataframe=std_rings_original, columns=rings_columns)
    clusters = _get_rings_default(dataframe=quarter_rings_original, columns=clusters_columns)

    # 2. Lógica de cálculo (correta, mas dependente da ordem das colunas)
    phi_p_eta_p = filtered_qr.iloc[:, 0::4]
    phi_p_eta_m = filtered_qr.iloc[:, 1::4]
    phi_m_eta_p = filtered_qr.iloc[:, 2::4]
    phi_m_eta_m = filtered_qr.iloc[:, 3::4]

    # --- CORREÇÃO APLICADA AQUI ---
    # Preserva o índice original e atribui nomes de colunas significativos
    
    # Nomes base para as novas colunas
    base_names = [col.rsplit('_', 2)[0] for col in phi_p_eta_p.columns]
    
    delta_eta_phi_p = pd.DataFrame(
        phi_p_eta_p.to_numpy() - phi_p_eta_m.to_numpy(),
        index=filtered_qr.index,
        columns=[f'{name}_delta_eta_phi_p' for name in base_names]
    )
    delta_eta_phi_m = pd.DataFrame(
        phi_m_eta_p.to_numpy() - phi_m_eta_m.to_numpy(),
        index=filtered_qr.index,
        columns=[f'{name}_delta_eta_phi_m' for name in base_names]
    )
    delta_phi_eta_p = pd.DataFrame(
        phi_p_eta_p.to_numpy() - phi_m_eta_p.to_numpy(),
        index=filtered_qr.index,
        columns=[f'{name}_delta_phi_eta_p' for name in base_names]
    )
    delta_phi_eta_m = pd.DataFrame(
        phi_p_eta_m.to_numpy() - phi_m_eta_m.to_numpy(),
        index=filtered_qr.index,
        columns=[f'{name}_delta_phi_eta_m' for name in base_names]
    )

    # 3. Concatenação final (agora funcionará corretamente)
    final_rings = pd.concat([
        delta_eta_phi_p, delta_eta_phi_m,
        delta_phi_eta_p, delta_phi_eta_m, 
        filtered_std,
        clusters
    ], axis=1)

    return final_rings

# # --- COMO CHAMAR A FUNÇÃO ---

# # 1. Crie um DataFrame de exemplo (similar ao seu 'rings')
# #    Vamos supor que as colunas são nomeadas 'ring_0', 'ring_1', etc.
# num_cols = 100
# col_names = [f'ring_{i}' for i in range(num_cols)]
# sample_data = pd.DataFrame([[i for i in range(num_cols)]], columns=col_names)

# # 2. Defina os grupos de colunas pelos NOMES, replicando a lógica do seu código antigo
# #    Antigo: RingsPS  = rings.iloc[:,0:4]
# rings_ps_cols = [f'ring_{i}' for i in range(0, 4)]

# #    Antigo: RingsEM1 = rings.iloc[:,8:41]
# rings_em1_cols = [f'ring_{i}' for i in range(8, 41)]

# #    Antigo: RingsEM2 = rings.iloc[:,72:76]
# rings_em2_cols = [f'ring_{i}' for i in range(72, 76)]

# #    Antigo: RingsEM3 = rings.iloc[:,80:84]
# rings_em3_cols = [f'ring_{i}' for i in range(80, 84)]

# #    Antigo: RingsHD  = rings.iloc[:,88:94]
# rings_hd_cols = [f'ring_{i}' for i in range(88, 94)]

# # Junte todos os grupos em uma única lista de listas
# all_column_groups = [
#     rings_ps_cols,
#     rings_em1_cols,
#     rings_em2_cols,
#     rings_em3_cols,
#     rings_hd_cols
# ]

# # 3. Chame a nova função com o DataFrame e a lista de nomes de colunas
# processed_rings = process_rings_default(sample_data, all_column_groups)