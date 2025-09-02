import lightgbm as lgb
import numpy as np
import pandas as pd
from modules.calibration_algorithm.gbdt import GBDTTrainning
from modules.hyperparameters_opt.bayesian_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from skopt.space import Real, Integer

# --- INÍCIO DO SCRIPT DE TESTE ---

# 1. Carregar o Banco de Dados (California Housing)
print("1. Carregando o banco de dados...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

print("Formato dos dados (features):", X.shape)
print("Amostra dos dados:")
print(X.head())

# 2. Dividir os dados em Conjunto de Treino e Teste
print("\n2. Dividindo os dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Tamanho do treino: {len(X_train)} amostras")
print(f"Tamanho do teste: {len(X_test)} amostras")


# --- OTIMIZAÇÃO PARA LIGHTGBM ---
print("="*50)
print("INICIANDO OTIMIZAÇÃO PARA LIGHTGBM")
print("="*50)

space_lgbm = [
    Integer(20, 100, name='num_leaves'),
    Integer(3, 15, name='max_depth'),
    Real(0.01, 0.2, name='learning_rate', prior='log-uniform')
]
fixed_params_lgbm = {
    'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
    'random_state': 42, 'n_jobs': -1, 'verbose': -1
}

# Simplesmente passe 'model_type="li ghtgbm"'
optimizer_lgbm = BayesianOptimization(
    model_type="lightgbm",
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    space=space_lgbm, fixed_params=fixed_params_lgbm
)
resultado_lgbm = optimizer_lgbm.run(n_calls=15, random_state=42)
print(f"\nMelhor MAE para LightGBM: {resultado_lgbm.fun:.4f}")
print(f"Melhores parâmetros: {resultado_lgbm.x}")

# --- OTIMIZAÇÃO PARA XGBOOST ---
print("\n" + "="*50)
print("INICIANDO OTIMIZAÇÃO PARA XGBOOST")
print("="*50)

space_xgb = [
    Integer(3, 15, name='max_depth'),
    Real(0.01, 0.2, name='learning_rate', prior='log-uniform'),
    Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree')
]
fixed_params_xgb = {
    'objective': 'reg:squarederror', 'eval_metric': 'mae', 'n_estimators': 1000,
    'random_state': 42, 'n_jobs': -1, 'booster': 'gbtree'
}

# Agora, para otimizar outro modelo, só mudamos o tipo e os parâmetros!
optimizer_xgb = BayesianOptimization(
    model_type="xgboost",
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    space=space_xgb, fixed_params=fixed_params_xgb
)
resultado_xgb = optimizer_xgb.run(n_calls=15, random_state=42)
print(f"\nMelhor MAE para XGBoost: {resultado_xgb.fun:.4f}")
print(f"Melhores parâmetros: {resultado_xgb.x}")