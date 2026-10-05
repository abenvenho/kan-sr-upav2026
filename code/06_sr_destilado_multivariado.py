"""
06_sr_destilado_multivariado.py
================================
Destilação KAN → Regressão Simbólica Multivariada (abordagem correta).

Estratégia: PySR treinado diretamente sobre as predições do KAN.
  Entrada:  X_train (12 features normalizadas)  →  y_KAN_train
  Objetivo: encontrar f(x1,...,x12) ≈ y_KAN  com expressão simbólica fechada

Diferença em relação a 03_regressao_simbolica.py:
  - 03: decomposição aditiva 1D sobre perfis marginais → perde interações
  - 06: expressão multivariada livre → captura interações entre features

Pipeline:
  1. Carrega X_train e y_KAN_train (predições KAN no treino, escala ln)
  2. Treina PySR multivariado (maxsize=20, niterations=300)
  3. Seleciona melhor equação pelo hall of fame
  4. Avalia em train/val/test (comparando com KAN)
  5. Salva expressão, métricas e predições

Entrada:
  preprocessed/X_*.pt, feature_names.json
  modelos/predicoes.npz   (y_KAN em escala ln)

Saída:
  resultados_sr_mv/
    formula_multivariada.txt        expressão simbólica final
    formula_multivariada_orig.txt   com variáveis em escala original
    metricas_sr_mv.json             R², RMSE, MAE, MAPE por split
    predicoes_sr_mv.npz             predições nos 3 splits
    hall_of_fame/                   todas as equações do PySR

Executar com:
  set KMP_DUPLICATE_LIB_OK=TRUE
  python 06_sr_destilado_multivariado.py
"""

import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Configurações
# ---------------------------------------------------------------------------

PREP_DIR = Path("preprocessed")
MOD_DIR  = Path("modelos")
OUT_DIR  = Path("resultados_sr_mv")
HOF_DIR  = OUT_DIR / "hall_of_fame"

for d in [OUT_DIR, HOF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# PySR — configuração para expressão multivariada
PYSR_CONFIG = dict(
    binary_operators   = ["+", "-", "*", "/"],
    unary_operators    = ["log", "sqrt", "square", "cube", "exp"],
    niterations        = 300,
    maxsize            = 20,
    parsimony          = 3e-4,      # penalidade de complexidade
    model_selection    = "best",
    verbosity          = 1,         # mostra progresso
    random_state       = 42,
    progress           = True,
    warm_start         = False,
    populations        = 30,        # mais populações = busca mais ampla
    population_size    = 33,
    ncycles_per_iteration = 550,
    turbo              = True,       # usa LLVM JIT se disponível
)

# ---------------------------------------------------------------------------
# 1. Verificação PySR
# ---------------------------------------------------------------------------

try:
    from pysr import PySRRegressor
    print("Biblioteca PySR: OK")
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysr", "--quiet"])
    from pysr import PySRRegressor

# ---------------------------------------------------------------------------
# 2. Carga dos dados
# ---------------------------------------------------------------------------

print("\nCarregando dados ...")

try:
    import torch
    def lt(p):
        try:
            return torch.load(p, weights_only=True).numpy()
        except TypeError:
            return torch.load(p).numpy()
    X_train = lt(PREP_DIR / "X_train.pt")
    X_val   = lt(PREP_DIR / "X_val.pt")
    X_test  = lt(PREP_DIR / "X_test.pt")
    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
except Exception as e:
    print(f"ERRO ao carregar tensores: {e}")
    sys.exit(1)

with open(PREP_DIR / "feature_names.json", encoding="utf-8") as f:
    FEATURES = json.load(f)

with open(PREP_DIR / "scaler_X.pkl", "rb") as f:
    scaler = pickle.load(f)
data_min   = scaler.data_min_
data_range = scaler.data_range_

# Predições KAN (alvo da destilação) — já em escala ln(valor/m²)
with np.load(MOD_DIR / "predicoes.npz") as pk:
    y_kan_train = pk["y_train_pred"].copy().astype(np.float64)
    y_kan_val   = pk["y_val_pred"].copy().astype(np.float64)
    y_kan_test  = pk["y_test_pred"].copy().astype(np.float64)
    y_true_train = pk["y_train_true"].copy().astype(np.float64)
    y_true_val   = pk["y_val_true"].copy().astype(np.float64)
    y_true_test  = pk["y_test_true"].copy().astype(np.float64)

print(f"  y_KAN_train: min={y_kan_train.min():.3f}  max={y_kan_train.max():.3f}  "
      f"mean={y_kan_train.mean():.4f}")

# ---------------------------------------------------------------------------
# 3. PySR multivariado: X_train → y_KAN_train
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("PYSR MULTIVARIADO — Destilação KAN → Expressão Simbólica")
print(f"Amostras de treino: {len(X_train):,}  |  Features: {len(FEATURES)}")
print("="*65)

model = PySRRegressor(**PYSR_CONFIG)

# Treina sobre predições KAN (não sobre y verdadeiro)
model.fit(
    X_train.astype(np.float64),
    y_kan_train,
    variable_names=FEATURES,
)

print("\nTreinamento concluído. Selecionando melhor equação ...")

# ---------------------------------------------------------------------------
# 4. Melhor expressão
# ---------------------------------------------------------------------------

best     = model.get_best()
expr_str = str(best["equation"])
cmplx    = int(best["complexity"])

print(f"\nMelhor equação (complexidade={cmplx}):")
print(f"  {expr_str}")

# Salva hall of fame completo
try:
    model.equations_.to_csv(HOF_DIR / "hall_of_fame_multivariado.csv", index=False)
    print(f"\nHall of fame salvo em {HOF_DIR}/hall_of_fame_multivariado.csv")
    print("\nTop 10 equações (por score):")
    cols = ["complexity", "score", "equation"]
    cols_disp = [c for c in cols if c in model.equations_.columns]
    print(model.equations_.sort_values("score", ascending=False).head(10)[cols_disp].to_string(index=False))
except Exception as e:
    print(f"  AVISO: não foi possível salvar hall of fame: {e}")

# ---------------------------------------------------------------------------
# 5. Função de avaliação
# ---------------------------------------------------------------------------

def metricas(y_true, y_pred, label=""):
    res   = y_pred - y_true
    ss_r  = np.sum(res**2)
    ss_t  = np.sum((y_true - y_true.mean())**2)
    r2    = float(1 - ss_r / max(ss_t, 1e-12))
    rmse  = float(np.sqrt(np.mean(res**2)))
    mae   = float(np.abs(res).mean())
    mape  = float(np.mean(np.abs(np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true)) * 100)
    if label:
        print(f"  {label:<12}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%"
              f"  n={len(y_true)}")
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "n": int(len(y_true))}


def predict(X):
    return model.predict(X.astype(np.float64))


# ---------------------------------------------------------------------------
# 6. Avaliação — fidelidade ao KAN (y_pred vs y_KAN)
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("FIDELIDADE AO KAN (SR_MV vs y_KAN)")
print("="*65)

fidelidade = {}
for sname, X_s, y_k in [("treino", X_train, y_kan_train),
                          ("val",    X_val,   y_kan_val),
                          ("teste",  X_test,  y_kan_test)]:
    y_p = predict(X_s)
    fidelidade[sname] = metricas(y_k, y_p, label=sname)

# ---------------------------------------------------------------------------
# 7. Avaliação — qualidade preditiva real (y_pred vs y_true)
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("DESEMPENHO PREDITIVO (SR_MV vs y_verdadeiro)")
print("="*65)

preds_mv = {}
metr_mv  = {}
for sname, X_s, y_t in [("treino", X_train, y_true_train),
                          ("val",    X_val,   y_true_val),
                          ("teste",  X_test,  y_true_test)]:
    y_p = predict(X_s)
    preds_mv[sname] = y_p
    metr_mv[sname]  = metricas(y_t, y_p, label=sname)

# Desempenho original do KAN para comparação
print("\n  KAN (referência):")
for sname, y_k, y_t in [("treino", y_kan_train, y_true_train),
                          ("val",    y_kan_val,   y_true_val),
                          ("teste",  y_kan_test,  y_true_test)]:
    metricas(y_t, y_k, label=sname + " (KAN)")

# ---------------------------------------------------------------------------
# 8. Fórmula em escala normalizada
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("FÓRMULA SIMBÓLICA (escala normalizada [0,1])")
print("="*65)
print(f"\nln(V/m²) = {expr_str}")
print(f"\nVariáveis: x_i = (x_orig_i - x_min_i) / x_range_i")
for i, feat in enumerate(FEATURES):
    print(f"  {feat:<28}  x = (orig - {data_min[i]:.2f}) / {data_range[i]:.2f}")

formula_norm = (
    f"ln(valor/m2) = {expr_str}\n\n"
    f"onde as variaveis sao normalizadas MinMax [0,1]:\n"
    + "\n".join(
        f"  {feat} -> x_{i} = ({feat} - {data_min[i]:.4f}) / {data_range[i]:.4f}"
        for i, feat in enumerate(FEATURES)
    )
)
with open(OUT_DIR / "formula_multivariada.txt", "w", encoding="utf-8") as f:
    f.write(formula_norm)

# ---------------------------------------------------------------------------
# 9. Fórmula com variáveis em escala original
#    Substitui cada nome de feature pela expressão de desnormalização
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("FÓRMULA (escala original das variáveis)")
print("="*65)

expr_orig = expr_str
for i, feat in enumerate(FEATURES):
    dmin = data_min[i]
    drng = data_range[i]
    # Substitui o nome da feature pelo equivalente em escala original
    subst = f"(({feat} - {dmin:.4f}) / {drng:.4f})"
    expr_orig = re.sub(r'\b' + re.escape(feat) + r'\b', subst, expr_orig)

print(f"\nln(valor/m²) = {expr_orig}")

formula_orig_txt = f"ln(valor_mercado / area_construida_m2) = \n  {expr_orig}"
with open(OUT_DIR / "formula_multivariada_orig.txt", "w", encoding="utf-8") as f:
    f.write(formula_orig_txt)

# ---------------------------------------------------------------------------
# 10. Salva predições e métricas
# ---------------------------------------------------------------------------

np.savez(
    OUT_DIR / "predicoes_sr_mv.npz",
    y_train_true=y_true_train,
    y_train_pred=preds_mv["treino"],
    y_val_true=y_true_val,
    y_val_pred=preds_mv["val"],
    y_test_true=y_true_test,
    y_test_pred=preds_mv["teste"],
    y_train_kan=y_kan_train,
    y_val_kan=y_kan_val,
    y_test_kan=y_kan_test,
)

# NBR 14.653-2 Grau II (teste)
m_te = metr_mv["teste"]
ok_r2  = m_te["r2"]   >= 0.64
ok_err = m_te["mape"] <= 30.0

metricas_out = {
    "metodo": "SR_MULTIVARIADO_DESTILADO_KAN",
    "descricao": "PySR multivariado treinado sobre predicoes KAN (preserva interacoes)",
    "expressao_normalizada": expr_str,
    "complexidade": cmplx,
    "metricas_vs_ytrue": metr_mv,
    "metricas_fidelidade_kan": fidelidade,
    "nbr14653_grau2_teste": {
        "r2_teste": m_te["r2"], "mape_teste": m_te["mape"],
        "ok_r2": ok_r2, "ok_mape": ok_err,
        "atende": bool(ok_r2 and ok_err),
    },
}

with open(OUT_DIR / "metricas_sr_mv.json", "w", encoding="utf-8") as f:
    json.dump(metricas_out, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 11. Resumo final
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("RESUMO FINAL")
print("="*65)
print(f"\n  Expressão (complexidade={cmplx}):")
print(f"  {expr_str}")
print(f"\n  Desempenho vs y_verdadeiro:")
for s, m in metr_mv.items():
    print(f"  {s.upper():<8}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  MAPE={m['mape']:.2f}%")
print(f"\n  NBR 14.653-2 Grau II (teste):")
print(f"    R² >= 0.64:  {m_te['r2']:.4f}  {'OK' if ok_r2 else 'NAO ATENDE'}")
print(f"    Erro <= 30%: {m_te['mape']:.2f}%  {'OK' if ok_err else 'NAO ATENDE'}")
print(f"    => {'ATENDE GRAU II' if (ok_r2 and ok_err) else 'NAO ATENDE GRAU II'}")
print(f"\nArquivos em: {OUT_DIR.resolve()}")
print("  formula_multivariada.txt")
print("  formula_multivariada_orig.txt")
print("  metricas_sr_mv.json")
print("  predicoes_sr_mv.npz")
print("  hall_of_fame/hall_of_fame_multivariado.csv")
print("\n06_sr_destilado_multivariado.py CONCLUÍDO.")
