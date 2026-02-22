import pandas as pd
import numpy as np
from typing import List, Optional

# Metodos para atributos dos rings - Apenas criar novas def
# Function to compute the relevant variables

layers = {
    "PS":  {"rings": list(range(0, 8)),   "qrings": list(range(0, 29))},
    "EM1": {"rings": list(range(8, 72)),  "qrings": list(range(29, 282))},
    "EM2": {"rings": list(range(72, 80)), "qrings": list(range(282, 311))},
    "EM3": {"rings": list(range(80, 88)), "qrings": list(range(311, 340))},
    "H1":  {"rings": list(range(88, 92)), "qrings": list(range(340, 353))},
    "H2":  {"rings": list(range(92, 96)), "qrings": list(range(353, 366))},
    "H3":  {"rings": list(range(96, 100)),"qrings": list(range(366, 379))},
}

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

def normalize_qrings(df,layers):
    for layer in layers.keys():
        
        rings = layers[layer]["rings"]
        qrings = layers[layer]["qrings"]

        for i in range(1, len(rings)):

            std = df[f"StdRings_{rings[i]}"]
            qr_cols = [f"QuarterRings_{j}" for j in range(qrings[4*i-3], qrings[4*i]+1)]
            df.loc[:, qr_cols] = (
                df[qr_cols]
                .div(std.abs(), axis=0)
                .where(std.ne(0), 0)
            )

    return df

def diff_qrings_energy(df, layers):
    # 1. Criamos um dicionário para armazenar todas as novas colunas
    new_cols = {}

    for layer in layers.keys():
        qrings = layers[layer]["qrings"]
        rings = layers[layer]["rings"]

        for i in range(1, len(rings)):
            # Definindo os nomes das colunas de origem
            q1 = f"QuarterRings_{qrings[4*i-3]}"
            q2 = f"QuarterRings_{qrings[4*i-2]}"
            q3 = f"QuarterRings_{qrings[4*i-1]}"
            q4 = f"QuarterRings_{qrings[4*i]}"

            # 2. Em vez de df[novo] = ..., guardamos no dicionário
            name_1_3 = f"diff_QuarterRings_{qrings[4*i-3]}_{qrings[4*i-1]}"
            name_2_4 = f"diff_QuarterRings_{qrings[4*i-2]}_{qrings[4*i]}"
            
            new_cols[name_1_3] = df[q1] - df[q3]
            new_cols[name_2_4] = df[q2] - df[q4]

    # 3. Concatenamos todas as novas colunas de uma vez só ao final
    # axis=1 significa concatenar colunas (horizontalmente)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df

def vectorize_rings_energy(df, layers):
    new_cols = {}
    
    for layer in layers.keys():
        rings = layers[layer]["rings"]
        qrings = layers[layer]["qrings"]

        for i in range(1, len(rings)):
            # Nomes das colunas de diferença (calculadas no passo anterior)
            col_dx = f"diff_QuarterRings_{qrings[4*i-2]}_{qrings[4*i]}"
            col_dy = f"diff_QuarterRings_{qrings[4*i-3]}_{qrings[4*i-1]}"
            
            dx = df[col_dx]
            dy = df[col_dy]

            # Módulo do vetor usando a hipotenusa
            mod_name = f"vec_mod_{rings[i]}"
            new_cols[mod_name] = np.hypot(dx, dy)

            # Ângulo em radianos ou graus
            ang_name = f"vec_ang_{rings[i]}"
            # np.arctan2 já lida com a divisão por zero e quadrantes corretamente
            new_cols[ang_name] = np.arctan2(dy, dx)

            # Flag booleana convertida para int (se o vetor é nulo)
            zero_name = f"vec_is_zero_{rings[i]}"
            new_cols[zero_name] = (new_cols[mod_name] == 0).astype(int)

    # Concatena todas as métricas vetoriais de uma vez
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    return df

def vectorize_layer_energy(df, layers):
    new_cols = {}

    for layer in layers.keys():
        rings = layers[layer]["rings"]
        qrings = layers[layer]["qrings"]
        
        # Inicializamos com zeros no formato da Series do DataFrame
        dx_sum = 0.0
        dy_sum = 0.0

        for i in range(1, len(rings)):
            # Somando as diferenças de todos os anéis daquela camada
            dx_sum += df[f"diff_QuarterRings_{qrings[4*i-2]}_{qrings[4*i]}"]
            dy_sum += df[f"diff_QuarterRings_{qrings[4*i-3]}_{qrings[4*i-1]}"]

        # Calculamos as métricas da camada inteira
        mod_name = f"vec_mod_layer_{layer}"
        ang_name = f"vec_ang_layer_{layer}"
        zero_name = f"vec_is_zero_layer_{layer}"

        new_cols[mod_name] = np.hypot(dx_sum, dy_sum)
        new_cols[ang_name] = np.arctan2(dy_sum, dx_sum)
        new_cols[zero_name] = (new_cols[mod_name] == 0).astype(int)

    # Inserção única para evitar fragmentação
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df