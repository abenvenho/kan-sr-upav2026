"""
01_preprocessamento.py
======================
Pre-processamento da base de modelagem KAN + Regressao Simbolica
Avaliacao de Apartamentos Residenciais - Sao Paulo-SP
Referencia: ABNT NBR 14.653-2:2011 (Grau II)

Entrada : base_modelagem_kan.csv  (15.000 registros, 36 variaveis)
Saidas  : preprocessed/
              X_train.pt, X_val.pt, X_test.pt   (tensores de features)
              y_train.pt, y_val.pt, y_test.pt    (tensores de alvo)
              scaler_X.pkl                        (MinMaxScaler ajustado)
              feature_names.json                  (ordem das colunas)
              splits_index.json                   (indices de cada split)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# 0. Configuracoes
# ---------------------------------------------------------------------------

TARGET = "ln_valor_m2"

FEATURES = [
    # Features originais (8)
    "area_construida_m2",
    "padrao_ordinal",
    "cep_norm",
    "t_ordinal",
    "flag_financiamento",
    "prop_financiamento",
    "flag_subdeclaracao",
    "area_terreno_m2",
    # Features IPTU adicionais (4) -- r=+0.61 e r=+0.36 vs ln_valor_m2
    "valor_m2_terreno",      # r=+0.61 -- valor fiscal do m2 de terreno (IPTU)
    "valor_m2_construcao",   # r=+0.61 -- valor fiscal do m2 de construcao (IPTU)
    "qtd_pavimentos",        # r=+0.36 -- proxy para edificio alto = localizacao premium
    "fator_obsolescencia",   # r=+0.09 -- depreciacao fisica do imovel
]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_STATE = 42

DATA_PATH = Path("base_modelagem_kan.csv")
OUT_DIR   = Path("preprocessed")

# ---------------------------------------------------------------------------
# 1. Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Dispositivo em uso: {device}")

# ---------------------------------------------------------------------------
# 2. Carga e validacao
# ---------------------------------------------------------------------------

print(f"\nCarregando {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, sep=",", encoding="utf-8")
print(f"  Shape original: {df.shape}")

missing_cols = [c for c in [TARGET] + FEATURES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Colunas ausentes na base: {missing_cols}")

df_model = df[[TARGET] + FEATURES].dropna().reset_index(drop=True)
n_removed = len(df) - len(df_model)
print(f"  Registros apos remocao de NaN: {len(df_model)}  (removidos: {n_removed})")

print("\nEstatisticas descritivas - variavel dependente:")
print(df_model[TARGET].describe().to_string())

# ---------------------------------------------------------------------------
# 3. Divisao estratificada  (70 / 15 / 15)
# ---------------------------------------------------------------------------

strat_col = "padrao_ordinal"

sss1 = StratifiedShuffleSplit(
    n_splits=1,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_STATE,
)
idx_train, idx_val_test = next(sss1.split(df_model, df_model[strat_col]))

val_share_within = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
sss2 = StratifiedShuffleSplit(
    n_splits=1,
    test_size=(1 - val_share_within),
    random_state=RANDOM_STATE,
)
df_val_test = df_model.iloc[idx_val_test].reset_index(drop=True)
idx_val_local, idx_test_local = next(
    sss2.split(df_val_test, df_val_test[strat_col])
)
idx_val  = idx_val_test[idx_val_local]
idx_test = idx_val_test[idx_test_local]

print(f"\nDivisao dos dados:")
print(f"  Treino    : {len(idx_train):>6} ({len(idx_train)/len(df_model)*100:.1f}%)")
print(f"  Validacao : {len(idx_val):>6} ({len(idx_val)/len(df_model)*100:.1f}%)")
print(f"  Teste     : {len(idx_test):>6} ({len(idx_test)/len(df_model)*100:.1f}%)")

for split_name, idx in [("Treino", idx_train), ("Val", idx_val), ("Teste", idx_test)]:
    dist = df_model.iloc[idx][strat_col].value_counts(normalize=True).sort_index()
    print(f"\n  Distribuicao por padrao - {split_name}:")
    for padrao, pct in dist.items():
        print(f"    Padrao {int(padrao)}: {pct*100:.1f}%")

# ---------------------------------------------------------------------------
# 4. X / y
# ---------------------------------------------------------------------------

X_all = df_model[FEATURES].values.astype(np.float32)
y_all = df_model[TARGET].values.astype(np.float32)

X_train_raw = X_all[idx_train]
X_val_raw   = X_all[idx_val]
X_test_raw  = X_all[idx_test]

y_train = y_all[idx_train]
y_val   = y_all[idx_val]
y_test  = y_all[idx_test]

# ---------------------------------------------------------------------------
# 5. Normalizacao MinMax (ajuste somente no treino)
# ---------------------------------------------------------------------------

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_norm = scaler.fit_transform(X_train_raw)
X_val_norm   = scaler.transform(X_val_raw)
X_test_norm  = scaler.transform(X_test_raw)

print("\nIntervalo apos normalizacao (treino):")
for i, feat in enumerate(FEATURES):
    print(f"  {feat:<30} min={X_train_norm[:, i].min():.4f}  max={X_train_norm[:, i].max():.4f}")

# ---------------------------------------------------------------------------
# 6. Tensores PyTorch
# ---------------------------------------------------------------------------

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

X_train_t = to_tensor(X_train_norm)
X_val_t   = to_tensor(X_val_norm)
X_test_t  = to_tensor(X_test_norm)
y_train_t = to_tensor(y_train)
y_val_t   = to_tensor(y_val)
y_test_t  = to_tensor(y_test)

print(f"\nFormatos dos tensores:")
print(f"  X_train: {X_train_t.shape}   y_train: {y_train_t.shape}")
print(f"  X_val  : {X_val_t.shape}   y_val  : {y_val_t.shape}")
print(f"  X_test : {X_test_t.shape}   y_test : {y_test_t.shape}")

# ---------------------------------------------------------------------------
# 7. Persistencia dos artefatos
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.save(X_train_t, OUT_DIR / "X_train.pt")
torch.save(X_val_t,   OUT_DIR / "X_val.pt")
torch.save(X_test_t,  OUT_DIR / "X_test.pt")
torch.save(y_train_t, OUT_DIR / "y_train.pt")
torch.save(y_val_t,   OUT_DIR / "y_val.pt")
torch.save(y_test_t,  OUT_DIR / "y_test.pt")

with open(OUT_DIR / "scaler_X.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(OUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
    json.dump(FEATURES, f, ensure_ascii=False, indent=2)

splits = {
    "idx_train": idx_train.tolist(),
    "idx_val":   idx_val.tolist(),
    "idx_test":  idx_test.tolist(),
}
with open(OUT_DIR / "splits_index.json", "w", encoding="utf-8") as f:
    json.dump(splits, f)

print(f"\nArtefatos salvos em '{OUT_DIR}/':")
for p in sorted(OUT_DIR.iterdir()):
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name:<25} {size_kb:>8.1f} KB")

# ---------------------------------------------------------------------------
# 8. Correlacao de Pearson - features vs. ln_valor_m2
# ---------------------------------------------------------------------------

print("\nCorrelacao de Pearson (treino) - features vs. ln_valor_m2:")
df_train = pd.DataFrame(X_train_norm, columns=FEATURES)
df_train[TARGET] = y_train
corr = df_train.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
for feat, r in corr.items():
    print(f"  {feat:<30} r = {r:+.4f}")

print("\nOK - Pre-processamento concluido com sucesso.")
