"""
09_selecionar_equacao_c19.py
============================
Seleciona a equação de complexity=19 do hall of fame existente,
avalia nos três splits e salva como modelo oficial KAN+SR MV c19.

Equação (complexity=19):
  ln(V/m²) = 7,6153
           + sqrt(valor_m2_terreno * 1.9699)
           + (valor_m2_construcao + area_terreno_m2) * 0.4585
           + cube(fator_obsolescencia - flag_subdeclaracao) * padrao_ordinal

Entrada:
  resultados_sr_mv/hall_of_fame/hall_of_fame_multivariado.csv
  preprocessed/  |  base_modelagem_kan.csv

Saída:
  resultados_sr_mv_c19/
    predicoes_sr_mv_c19.npz
    metricas_sr_mv_c19.json
    formula_c19.txt
    formula_c19_original.txt
"""

import json, pickle, re, warnings
import numpy as np, pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PREP  = Path("preprocessed")
HOF   = Path("resultados_sr_mv/hall_of_fame/hall_of_fame_multivariado.csv")
OUT   = Path("resultados_sr_mv_c19")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Carrega dados
# ---------------------------------------------------------------------------
print("Carregando dados ...")
df = pd.read_csv("base_modelagem_kan.csv", sep=None, engine="python")
with open(PREP/"splits_index.json") as f: idx = json.load(f)
with open(PREP/"feature_names.json") as f: FEATURES = json.load(f)
with open(PREP/"scaler_X.pkl","rb") as f: sc = pickle.load(f)

def get_X(split):
    return sc.transform(df.iloc[idx[f"idx_{split}"]][FEATURES].values.astype(float))

def get_y(split):
    return df.iloc[idx[f"idx_{split}"]]["ln_valor_m2"].values.astype(float)

X_tr, X_va, X_te = get_X("train"), get_X("val"), get_X("test")
y_tr, y_va, y_te = get_y("train"), get_y("val"), get_y("test")

# KAN preds para fidelidade
with np.load("modelos/predicoes.npz") as d:
    ykan_tr = d["y_train_pred"].copy()
    ykan_va = d["y_val_pred"].copy()
    ykan_te = d["y_test_pred"].copy()

# ---------------------------------------------------------------------------
# 2. Seleciona equação complexity=19 do hall of fame
# ---------------------------------------------------------------------------
hof = pd.read_csv(HOF)
row19 = hof[hof["complexity"] == 19].iloc[0]
EQ_STR = str(row19["equation"])
print(f"\nEquação complexity=19:")
print(f"  {EQ_STR}")
print(f"  loss(treino KAN)={float(row19['loss']):.6f}")

# ---------------------------------------------------------------------------
# 3. Avaliação numérica da equação
# ---------------------------------------------------------------------------
def eval_eq(eq, X):
    ns = {f: X[:,i] for i,f in enumerate(FEATURES)}
    ns.update({"np": np,
               "sqrt":   lambda x: np.sqrt(np.clip(x, 0, None)),
               "log":    lambda x: np.log(np.clip(np.abs(x)+1e-10, 1e-10, None)),
               "exp":    lambda x: np.exp(np.clip(x, -50, 50)),
               "square": lambda x: x**2,
               "cube":   lambda x: x**3})
    r = eval(eq, ns)
    return np.asarray(r, dtype=float)

def metricas(yt, yp, label=""):
    res  = yp - yt
    r2   = float(1 - np.sum(res**2) / max(np.sum((yt-yt.mean())**2), 1e-12))
    rmse = float(np.sqrt(np.mean(res**2)))
    mae  = float(np.abs(res).mean())
    mape = float(np.mean(np.abs(np.exp(yp)-np.exp(yt))/np.exp(yt))*100)
    if label:
        print(f"  {label:<10}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%")
    return {"r2":r2,"rmse":rmse,"mae":mae,"mape":mape,"n":int(len(yt))}

yp_tr = eval_eq(EQ_STR, X_tr)
yp_va = eval_eq(EQ_STR, X_va)
yp_te = eval_eq(EQ_STR, X_te)

print("\n=== DESEMPENHO PREDITIVO (vs y_real) ===")
m_tr = metricas(y_tr, yp_tr, "treino")
m_va = metricas(y_va, yp_va, "val")
m_te = metricas(y_te, yp_te, "teste")

print("\n=== FIDELIDADE AO KAN ===")
f_tr = metricas(ykan_tr, yp_tr, "treino")
f_va = metricas(ykan_va, yp_va, "val")
f_te = metricas(ykan_te, yp_te, "teste")

ok_r2   = m_te["r2"]   >= 0.64
ok_mape = m_te["mape"] <= 30.0
print(f"\n  NBR 14.653-2 Grau II (teste):")
print(f"    R² >= 0,64: {m_te['r2']:.4f}  {'OK' if ok_r2 else 'NAO ATENDE'}")
print(f"    MAPE <= 30%: {m_te['mape']:.2f}%  {'OK' if ok_mape else 'NAO ATENDE'}")
print(f"    => {'ATENDE GRAU II' if (ok_r2 and ok_mape) else 'NAO ATENDE GRAU II'}")

# ---------------------------------------------------------------------------
# 4. Salva predições
# ---------------------------------------------------------------------------
np.savez(OUT/"predicoes_sr_mv_c19.npz",
         y_train_true=y_tr, y_train_pred=yp_tr,
         y_val_true=y_va,   y_val_pred=yp_va,
         y_test_true=y_te,  y_test_pred=yp_te,
         y_train_kan=ykan_tr, y_val_kan=ykan_va, y_test_kan=ykan_te)

# ---------------------------------------------------------------------------
# 5. Fórmula textual — escala normalizada
# ---------------------------------------------------------------------------
with open(PREP/"scaler_X.pkl","rb") as f: sc2 = pickle.load(f)
xmin = sc2.data_min_; xrng = sc2.data_range_

formula_norm = (
    f"ln(V/m²) = {EQ_STR}\n\n"
    "onde as variáveis são normalizadas MinMax [0,1]:\n"
    + "\n".join(f"  {feat} -> (orig - {xmin[i]:.4f}) / {xrng[i]:.4f}"
                for i,feat in enumerate(FEATURES))
)

# Fórmula com variáveis originais
expr_orig = EQ_STR
for i,feat in enumerate(FEATURES):
    subst = f"(({feat} - {xmin[i]:.4f}) / {xrng[i]:.4f})"
    expr_orig = re.sub(r'\b' + re.escape(feat) + r'\b', subst, expr_orig)

formula_orig = f"ln(valor_mercado / area_construida_m2) =\n  {expr_orig}"

print(f"\n=== FÓRMULA (escala normalizada) ===")
print(f"ln(V/m²) = {EQ_STR}")
print(f"\n=== FÓRMULA (escala original) ===")
print(formula_orig)

with open(OUT/"formula_c19.txt","w",encoding="utf-8") as f: f.write(formula_norm)
with open(OUT/"formula_c19_original.txt","w",encoding="utf-8") as f: f.write(formula_orig)

# ---------------------------------------------------------------------------
# 6. Salva métricas
# ---------------------------------------------------------------------------
metricas_json = {
    "metodo": "KAN_SR_MV_COMPLEXITY_19",
    "complexidade": 19,
    "equacao_normalizada": EQ_STR,
    "features_usadas": ["valor_m2_terreno","valor_m2_construcao","area_terreno_m2",
                        "fator_obsolescencia","flag_subdeclaracao","padrao_ordinal"],
    "n_features": 6,
    "metricas_vs_ytrue": {"treino":m_tr,"val":m_va,"teste":m_te},
    "fidelidade_kan": {"treino":f_tr,"val":f_va,"teste":f_te},
    "nbr14653_grau2_teste": {"r2":m_te["r2"],"mape":m_te["mape"],
                              "ok_r2":ok_r2,"ok_mape":ok_mape,
                              "atende":bool(ok_r2 and ok_mape)},
}
with open(OUT/"metricas_sr_mv_c19.json","w",encoding="utf-8") as f:
    json.dump(metricas_json, f, indent=2, ensure_ascii=False)

print(f"\nArquivos em: {OUT.resolve()}")
print("  predicoes_sr_mv_c19.npz")
print("  metricas_sr_mv_c19.json")
print("  formula_c19.txt")
print("  formula_c19_original.txt")
print("\n09_selecionar_equacao_c19.py CONCLUÍDO.")
