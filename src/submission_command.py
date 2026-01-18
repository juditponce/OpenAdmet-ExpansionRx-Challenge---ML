# -*- coding: utf-8 -*-
# %%
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.svm import SVR
from scipy.stats import spearmanr, kendalltau
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin


# %%

tqdm.pandas()
sns.set_style("whitegrid")
sns.set_context("notebook")

# %%

#Carga, exploración y preparación de datos ADMET 
df_ADMET = pd.read_csv("openadmet.csv")

#Preparación librería ADMET -- transformación logarítmica
data = """Assay,Log_Scale,Multiplier,Log_name
LogD,False,1,LogD
KSOL,True,1e-6,LogS
HLM CLint,True,1,Log_HLM_CLint
MLM CLint,True,1,Log_MLM_CLint
Caco-2 Permeability Papp A>B,True,1e-6,Log_Caco_Papp_AB
Caco-2 Permeability Efflux,True,1,Log_Caco_ER
MPPB,True,1,Log_Mouse_PPB
MBPB,True,1,Log_Mouse_BPB
MGMB,True,1,Log_Mouse_MPB
"""

conversion_df = pd.read_csv(StringIO(data))
conversion_dict = dict([(x[0], x[1:]) for x in conversion_df.values])

log_train_df = df_ADMET[["SMILES", "Molecule Name"]].copy()
for col in df_ADMET.columns[2:]:
    log_scale, multiplier, short_name = conversion_dict[col]
    log_train_df[short_name] = df_ADMET[col].astype(float)
    if log_scale:
        log_train_df[short_name] = np.log10(log_train_df[short_name] * multiplier)
        

y_cols = log_train_df.columns[2:]

# %%

#Carga, exploración y preparación mordred
df_mordred = pd.read_csv("openadmet_mordred.csv")

#Mordred puede fallar para algunos descriptores. Se eliminan columnas con NA
na_cols = df_mordred.columns[df_mordred.isna().any()]
mordred_valid = df_mordred.drop(columns=na_cols)

#buscamos columnas cte
cols_const = mordred_valid.columns[2:]
constant_cols = cols_const[mordred_valid[cols_const].nunique() <= 1]
mordred_valid = mordred_valid.drop(columns=constant_cols)

#Ahora vamos a elimianr columnas con varianza inferior a 0.05
mordred_num = mordred_valid.select_dtypes(include='number')
low_var_cols = mordred_num.var()[mordred_num.var() < 0.05].index
mordred_valid = mordred_valid.drop(columns=low_var_cols)

#Se normalizan las columnas numéricas restantes (media 0, desviacion 1)
mordred_valid_num = mordred_valid.iloc[:, 2:]

mordred_valid_num = mordred_valid_num.apply(pd.to_numeric, errors="coerce")
# %%
fp_df = pd.read_csv("openadmet_morgan_fp.csv")
fp_df = fp_df.set_index("Molecule Name").loc[log_train_df["Molecule Name"]].reset_index()


X_fp = fp_df[[col for col in fp_df.columns if col.startswith("FP_")]].values
X_mordred = mordred_valid_num.values
X_full = np.hstack([X_mordred, X_fp])

# Asegurarnos de que X_full sea float
X_full_float = X_full.astype(float)

# Detectar NaN
nan_full = np.isnan(X_full_float)
if nan_full.any():
    rows, cols = np.where(nan_full)
    print(f"Hay {len(rows)} NaN en X_full en las posiciones (fila, columna):")
    for r, c in zip(rows, cols):
        print(f"Fila {r}, Columna {c} -> Mordred+FP columna {c}")
else:
    print("No hay NaN en X_full")
    
df_full = pd.DataFrame(X_full_float, index=log_train_df["Molecule Name"])

nan_cols = df_full.isna().any(axis=0)
print(f"Se van a eliminar {nan_cols.sum()}, columnas con NaN en X_full")

X_full= df_full.loc[:, ~nan_cols].values


# %%

#TEST SET
df_test_mordred = pd.read_csv("openadmet_test_mordred.csv")
mordred_test_num = df_test_mordred.iloc[:, 2:]
mordred_test_num = mordred_test_num.apply(pd.to_numeric, errors="coerce")

fp_test_df = pd.read_csv("openadmet_test_morgan_fp.csv")
fp_test_df = fp_test_df.set_index("Molecule Name")\
                       .loc[df_test_mordred["Molecule Name"]]\
                       .reset_index()
                       
X_fp_test = fp_test_df[[c for c in fp_test_df.columns if c.startswith("FP_")]].values
X_mordred_test = mordred_test_num.values
X_test = np.hstack([X_mordred_test, X_fp_test]) 

X_test_float = X_test.astype(float) 

nan_full_test = np.isnan(X_test_float)
if nan_full_test.any():
    rows, cols = np.where(nan_full_test)
    print(f"Hay {len(rows)} NaN en X_test en las posiciones (fila, columna):")
    for r, c in zip(rows, cols):
        print(f"Fila {r}, Columna {c} -> Mordred+FP columna {c}")
else:
    print("No hay NaN en X_test")
    

df_test = pd.DataFrame(X_test_float, index = df_test_mordred["Molecule Name"])

nan_cols_test = df_test.isna().any(axis=0)
print(f"Se van a eliminar {nan_cols_test.sum()}, columnas con NaN en X_test")

X_test= df_test.loc[:, ~nan_cols_test].values

# %%
cols_no_nan_full = df_full.columns[~df_full.isna().any()]
cols_no_nan_test = df_test.columns[~df_test.isna().any()]

common_cols = cols_no_nan_full.intersection(cols_no_nan_test)

df_full = df_full[common_cols]
df_test = df_test[common_cols]

X_full = df_full.values
X_test = df_test.values



print("NaN en X_full:", np.isnan(X_full).any())
print("NaN en X_test", np.isnan(X_test).any())
print(f"X_full shape: {X_full.shape}, X_test shape: {X_test.shape}")


# %%
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8): 
        self.threshold = threshold
        self.to_keep_ = None  # columnas que vamos a conservar

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        corr = df.corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # columnas a eliminar
        to_drop = [c for c in upper.columns if any(upper[c] > self.threshold)]
        # columnas a conservar
        self.to_keep_ = [c for c in df.columns if c not in to_drop]
        return self

    def transform(self, X): 
        df = pd.DataFrame(X)
        if self.to_keep_ is None:
            raise ValueError("El filtro no ha sido entrenado. Llama primero a fit().")
        # conservamos solo las columnas seleccionadas en train
        return df[self.to_keep_].values

# %%

#Gridsearch y modelos

modelos = {
    "SVM": (
        SVR(kernel="rbf"),
        {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.01],
            "model__epsilon": [0.05, 0.1]
        }
    ),
    "RF": (
        RandomForestRegressor(random_state=42, n_jobs = -1),
        {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 15],
            "model__min_samples_leaf": [1, 2]
        }
    ),
    "XGB": (
        XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1),
        {
            "model__n_estimators": [300,500],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.03, 0.05]
        }
    )
}

cv = KFold(n_splits = 5, shuffle=True, random_state=42)

results_all=[]
predictions_all={}


# %% Loop seguro para entrenamiento y predicción
for model_name, (model, param_grid) in modelos.items():
    print(f"\n --- {model_name} ---")
    
    predictions = pd.DataFrame({"Molecule Name": df_test_mordred["Molecule Name"]})
    results = []
    
    for col in y_cols:
        # Filtramos y no NaN
        mask = ~log_train_df[col].isna()
        X = X_full[mask]
        y = log_train_df.loc[mask, col].values
        
        if len(y) < 10:
            print(f"Skipping {col} (not enough data)")
            continue
        
        # Limpiar X e y de NaN o inf
        finite_mask = np.isfinite(y)
        X = X[finite_mask]
        y = y[finite_mask]
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            print(f"Skipping {col} (no valid data after cleaning)")
            continue
        
        # Pipeline: primero imputar, luego filtrar correlación, luego escalar
        pipe = Pipeline([
            ("var", VarianceThreshold(1e-6)),               # luego eliminar varianza casi 0
            ("corr", CorrelationFilter(threshold=0.8)),     # filtrar correlación
            ("scaler", StandardScaler()),                   # normalización
            ("model", model)                                # modelo
        ])
        
        search = GridSearchCV(
            pipe,
            param_grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1
        )
        
        try:
            search.fit(X, y)
            best_model = search.best_estimator_
            
            # Cross-validated predictions en train
            from sklearn.model_selection import cross_val_predict
            y_cv_pred = cross_val_predict(
                best_model,
                X,
                y,
                cv=cv,
                n_jobs=-1
            )
            
            # Guardamos métricas
            results.append({
                "Model": model_name,
                "Variable": col,
                "R2": r2_score(y, y_cv_pred),
                "MAE": mean_absolute_error(y, y_cv_pred),
                "Spearman_R": spearmanr(y, y_cv_pred)[0],
                "Kendall_Tau": kendalltau(y, y_cv_pred)[0]
            })
            
           
            predictions[col] = best_model.predict(X_test)
                
        except Exception as e:
            print(f"Warning: falló el entrenamiento de {col} con {model_name}: {e}")
            results.append({
                "Model": model_name,
                "Variable": col,
                "R2": np.nan,
                "MAE": np.nan,
                "Spearman_R": np.nan,
                "Kendall_Tau": np.nan
            })
            predictions[col] = np.nan
            
    results_all.append(pd.DataFrame(results))
    predictions_all[model_name] = predictions

# Guardado ensemble seguro
ensemble = predictions_all["SVM"].copy()
for col in y_cols:
    ensemble[col] = (
        predictions_all["SVM"].get(col, 0) +
        predictions_all["RF"].get(col, 0) +
        predictions_all["XGB"].get(col, 0)
    ) / 3

def inverse_log_transform(y_pred_log, multiplier):
    return (10 ** y_pred_log) / multiplier

for col in y_cols:
    for key, (log_scale, multiplier, short_name) in conversion_dict.items():
        if short_name == col and log_scale:
            ensemble[col] = inverse_log_transform(ensemble[col].values, multiplier)

rename_dict = {
    "LogD": "LogD",
    "LogS": "KSOL",
    "Log_HLM_CLint": "HLM CLint",
    "Log_MLM_CLint": "MLM CLint",
    "Log_Caco_Papp_AB": "Caco-2 Permeability Papp A>B",
    "Log_Caco_ER": "Caco-2 Permeability Efflux",
    "Log_Mouse_PPB": "MPPB",
    "Log_Mouse_BPB": "MBPB",
    "Log_Mouse_MPB": "MGMB"
}

ensemble = ensemble.rename(columns={col: rename_dict[col] for col in ensemble.columns if col in rename_dict})

ensemble.to_csv("ensemble_svm_rf_xgb_original_scale.csv", index=False)


