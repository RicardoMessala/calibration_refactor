import pandas as pd
import numpy as np
from typing import List, Optional, Union

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

def _get_columns(dataframe: pd.DataFrame, columns: Optional[List[str]] = None) -> Union[pd.DataFrame,np.ndarray]:
    """
    Processes the DataFrame by selecting columns based on a list
    of column names. If 'columns' is None or an empty list, 
    returns the original DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        columns (Optional[List[str]]): A list containing the names of the
                                       columns to be selected.
                                       Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame with the selected columns, or the
                      original DataFrame if 'columns' is not provided or is empty.
    """
    # Checks if columns is None or an empty list.
    # If so, returns the original DataFrame without modifications.
    print('colunas do _get_columns', columns)
    if not columns:
        print('new_processing_print', list(dataframe.columns))
        return dataframe

    # Selects all columns at once using the list of names.
    return dataframe[columns].values.astype(np.float32)

def _remove_columns(dataframe: pd.DataFrame, columns: Optional[List[str]] = None) -> Union[pd.DataFrame, np.ndarray]:

    # Checks if columns is None or an empty list.
    # If so, returns the original DataFrame without modifications.
    print('colunas do _remove_columns', columns)
    if not columns:
        print('new_processing_print', list(dataframe.columns))
        return dataframe

    # Selects all columns at once using the list of names.
    return dataframe.drop(columns=columns)
   
def _columns_selector(dataframe: pd.DataFrame, columns: Optional[List[str]] = None, selector:str=None) -> Union[pd.DataFrame, np.ndarray]:
    print('saida selector ', selector)
    if selector=='get':
        return _get_columns(dataframe=dataframe, columns=columns)
    
    elif selector=='remove':
        return _remove_columns(dataframe=dataframe, columns=columns)
    
    else:
        return dataframe


def calc_asym_weights_delta(
    quarter_rings_original: pd.DataFrame,
    std_rings_original: pd.DataFrame,
    rings_columns: List[List[str]] = None,
    clusters_columns: List[List[str]] = None
) -> Union[pd.DataFrame, np.ndarray]:
    
    # 1. Filtragem dos dados (esta parte está correta)
    filtered_qr = _get_columns(dataframe=quarter_rings_original, columns=rings_columns)
    filtered_std = _get_columns(dataframe=std_rings_original, columns=rings_columns)
    clusters = _get_columns(dataframe=quarter_rings_original, columns=clusters_columns)

    # 2. Lógica de cálculo (correta, mas dependente da ordem das colunas)
    phi_p_eta_p = filtered_qr[:, 0::4]
    phi_p_eta_m = filtered_qr[:, 1::4]
    phi_m_eta_p = filtered_qr[:, 2::4]
    phi_m_eta_m = filtered_qr[:, 3::4]

    # --- CORREÇÃO APLICADA AQUI ---
    # Preserva o índice original e atribui nomes de colunas significativos
    
    # Nomes base para as novas colunas
    base_names = [col.rsplit('_', 2)[0] for col in rings_columns[0::4]]
    idx = quarter_rings_original.index

    delta_eta_phi_p = pd.DataFrame(
        phi_p_eta_p - phi_p_eta_m,
        index=quarter_rings_original.index,
        columns=[f'{name}_delta_eta_phi_p' for name in base_names]
    )
    delta_eta_phi_m = pd.DataFrame(
        phi_m_eta_p - phi_m_eta_m,
        index=quarter_rings_original.index,
        columns=[f'{name}_delta_eta_phi_m' for name in base_names]
    )
    delta_phi_eta_p = pd.DataFrame(
        phi_p_eta_p - phi_m_eta_p,
        index=quarter_rings_original.index,
        columns=[f'{name}_delta_phi_eta_p' for name in base_names]
    )
    delta_phi_eta_m = pd.DataFrame(
        phi_p_eta_m - phi_m_eta_m,
        index=quarter_rings_original.index,
        columns=[f'{name}_delta_phi_eta_m' for name in base_names]
    )

    df_std=pd.DataFrame(filtered_std, index =idx, columns=rings_columns)
    df_cluster = pd.DataFrame(clusters, index=idx, columns=clusters_columns)


    # 3. Concatenação final (agora funcionará corretamente)
    final_rings = pd.concat([
        delta_eta_phi_p, delta_eta_phi_m,
        delta_phi_eta_p, delta_phi_eta_m, 
        df_std,
        df_cluster
    ], axis=1)

    return final_rings

def normalize_qrings(df,layers):
    for layer in layers.keys():
        
        rings = layers[layer]["rings"]
        qrings = layers[layer]["qrings"]

        for i in range(1, len(rings)):

            std = f"StdRings_{rings[i]}"
            qr_cols = [f"QuarterRings_{j}" for j in range(qrings[4*i-3], qrings[4*i]+1)]
            
            qr_matriz = df[qr_cols].values
            std_abs = np.abs(df[std].values)[:, np.newaxis]

            result = np.zeros_like(qr_matriz)
            np.divide(qr_matriz, std_abs,out=result, where=std_abs !=0)

            result[~np.isfinite(result)] = np.nan
            
            df.loc[:, qr_cols] = result

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

            
            new_cols[name_1_3] = df[q1].values - df[q3].values
            new_cols[name_2_4] = df[q2].values - df[q4].values

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
            
            dx = df[col_dx].values
            dy = df[col_dy].values

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
        
        dx_sum = 0.0
        dy_sum = 0.0

        for i in range(1, len(rings)):
            dx_sum += df[f"diff_QuarterRings_{qrings[4*i-2]}_{qrings[4*i]}"].values
            dy_sum += df[f"diff_QuarterRings_{qrings[4*i-3]}_{qrings[4*i-1]}"].values

        # 1. Cálculo do Módulo (Magnitude)
        mod_name = f"vec_mod_layer_{layer}"
        mod = np.hypot(dx_sum, dy_sum)
        new_cols[mod_name] = mod

        # 2. Padrão Ouro: Seno e Cosseno
        # Usamos np.where para evitar divisão por zero se o módulo for 0
        new_cols[f"vec_cos_layer_{layer}"] = np.where(mod > 0, dx_sum / mod, 0.0)
        new_cols[f"vec_sin_layer_{layer}"] = np.where(mod > 0, dy_sum / mod, 0.0)

        # 3. Flag de Vetor Nulo (Importante para a árvore isolar esses casos)
        zero_name = f"vec_is_zero_layer_{layer}"
        new_cols[zero_name] = (mod == 0).astype(int)

    # Inserção única eficiente
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df

def vector_layers_components(df, layers):
    new_cols = {}

    for layer in layers.keys():
        rings = layers[layer]["rings"]
        qrings = layers[layer]["qrings"]

        dy_sum_q1 = 0
        dx_sum_q2 = 0
        dy_sum_q3 = 0
        dx_sum_q4 = 0

        for i in range(1, len(rings)):
            dy_sum_q1 += df[f"QuarterRings_{qrings[4*i-3]}"].values
            dx_sum_q2 += df[f"QuarterRings_{qrings[4*i-2]}"].values
            dy_sum_q3 += df[f"QuarterRings_{qrings[4*i-1]}"].values
            dx_sum_q4 += df[f"QuarterRings_{qrings[4*i]}"].values

        new_cols[f"vec_Q1_layer_{layer}"] = dy_sum_q1
        new_cols[f"vec_Q2_layer_{layer}"] = dx_sum_q2
        new_cols[f"vec_Q3_layer_{layer}"] = dy_sum_q3
        new_cols[f"vec_Q4_layer_{layer}"] = dx_sum_q4

    # Inserção única eficiente
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df

def rings_layers_sum(df, layers):
    new_cols = {}

    for layer, cfg in layers.items():
        rings = cfg["rings"]

        cols = [f"StdRings_{r}" for r in rings]
        new_cols[f"E_layer_{layer}"] = df[cols].sum(axis=1)

    # inserção única eficiente
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df