import numpy as np

def mspe_objective(preds, train_data):
    """
    Custom objective function for MSPE loss.
    """
    # Extrair os rótulos reais (y)
    y_true = train_data.get_label()
    
    # Para evitar divisão por zero, usamos um epsilon ou garantimos y != 0
    # Ajuste conforme a natureza dos seus dados

    # Cálculo do Gradiente: 2 * (preds - y) / y^2
    grad = 2 * (preds - y_true) / y_true**2
    #/ y_true_sq
    
    # Cálculo da Hessiana: 2 / y^2
    hess = 2 /y_true**2
    #/ y_true_sq
    factor = 1e8
    # print((grad))
    # print((hess))
    grad=grad*factor
    hess=hess*factor
    return grad, hess

def mspe_metric(preds, train_data):
    """
    Custom evaluation metric for MSPE.
    """
    y_true = train_data.get_label()
    
    # Cálculo do erro percentual quadrático
    # Adicionamos epsilon para evitar divisão por zero
    epsilon = 1e-8
    percentage_error = (y_true - preds) / np.maximum(y_true, epsilon)
    mspe = np.mean(np.square(percentage_error))
    
    # Retorna: (nome_da_metrica, valor_da_metrica, is_higher_better)
    return ('mspe', mspe, False)

def calculate_mspe(y_true, y_pred):
    """
    Calcula o Mean Squared Percentage Error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Evita divisão por zero se houver valores reais nulos
    denominator = np.maximum(np.square(y_true), 1e-8)
    mspe = np.mean(np.square(y_true - y_pred) / denominator)
    
    return mspe

def mae_objective(preds, train_data):
    y_true = train_data.get_label()
    
    # Gradiente: sinal da diferença entre predição e real
    grad = (preds - y_true)
    
    # Hessiana: O MAE tem segunda derivada zero. 
    # Para o LGBM funcionar, atribuímos um valor constante (1.0)
    hess = np.ones_like(preds)

    return grad, hess

def mae_metric(preds, train_data):
    y_true = train_data.get_label()
    mae = np.mean(np.abs(y_true - preds))
    
    # Nome, valor, is_higher_better
    return ('mae', mae, False)

def l2_loss(y, data):
    t = data.get_label()
    grad = y - t 
    hess = np.ones_like(y)
    return grad, hess

def l2_eval(y, data):
    t = data.get_label()
    loss = (y - t) ** 2 
    return 'l2', loss.mean(), False