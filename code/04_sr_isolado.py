"""
04_sr_isolado.py
================
Regressao Simbolica (PySR) treinada DIRETAMENTE nos dados de entrada.
Nao usa as ativacoes/perfis do KAN — modelo independente para comparacao.

Estrategia: Generalized Additive Model Simbolico (SAMI)
    ln(V/m2) = C + sum_i g_i(x_i_norm)

Algoritmo:
  1. Ordena features por |correlacao de Pearson| com y (descendente)
  2. C = mean(y_train)
  3. Para cada feature i (em ordem):
       - Computa residuo parcial: r = y - C - sum_{j<i} g_j(x_j)
       - Bin-smoothing: 100 percentis de x_i vs media(r) em cada bin
       - PySR 1D sobre o perfil suavizado
       - g_i armazenada como interpolacao do perfil PySR
  4. Avalia o modelo aditivo em train/val/test
  5. Gera formula simbolica e metricas

Entrada : preprocessed/  (X_*.pt, y_*.pt, scaler_X.pkl, feature_names.json)
          modelos/norm_target.json  (parametros de desnormalizacao do target)
Saidas  : resultados_sr_isolado/
              formulas_isolado.json
              formula_final_isolado.txt
              formula_original_isolado.txt
              metricas_sr_isolado.json
              perfis/  (perfis marginais .npz para comparacao)
              hall_of_fame/

Executar com:
    set KMP_DUPLICATE_LIB_OK=TRUE
    python 04_sr_isolado.py
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
# 0. Configuracoes
# ---------------------------------------------------------------------------

PREP_DIR  = Path("preprocessed")
MOD_DIR   = Path("modelos")
OUT_DIR   = Path("resultados_sr_isolado")
HOF_DIR   = OUT_DIR / "hall_of_fame"
PERF_DIR  = OUT_DIR / "perfis"

for d in [OUT_DIR, HOF_DIR, PERF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_BINS       = 100   # bins para suavizacao do perfil
MIN_BIN_PTS  = 5     # minimo de pontos por bin para incluir no ajuste PySR

# PySR — configuracao base
PYSR_BASE = dict(
    binary_operators = ["+", "-", "*", "/"],
    unary_operators  = ["log", "sqrt", "square", "cube", "exp"],
    niterations      = 200,
    maxsize          = 15,
    parsimony        = 5e-4,
    model_selection  = "best",
    verbosity        = 0,
    random_state     = 42,
    progress         = False,
    warm_start       = False,
)

BINARY_FEATS = {"flag_financiamento", "flag_subdeclaracao"}

# ---------------------------------------------------------------------------
# 1. Verificacao PySR
# ---------------------------------------------------------------------------

def instalar_pysr():
    import subprocess
    print("  Instalando pysr ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysr", "--quiet"])

try:
    from pysr import PySRRegressor
    print("Biblioteca PySR: OK")
except ImportError:
    instalar_pysr()
    from pysr import PySRRegressor

# ---------------------------------------------------------------------------
# 2. Carga dos tensores pre-processados
# ---------------------------------------------------------------------------

print("\nCarregando tensores pre-processados ...")

try:
    import torch
    def load_tensor(path):
        try:
            return torch.load(path, weights_only=True).numpy()
        except TypeError:
            return torch.load(path).numpy()
    X_train = load_tensor(PREP_DIR / "X_train.pt")
    X_val   = load_tensor(PREP_DIR / "X_val.pt")
    X_test  = load_tensor(PREP_DIR / "X_test.pt")
    y_train = load_tensor(PREP_DIR / "y_train.pt").ravel()
    y_val   = load_tensor(PREP_DIR / "y_val.pt").ravel()
    y_test  = load_tensor(PREP_DIR / "y_test.pt").ravel()
    print(f"  Torch OK — X_train: {X_train.shape}  y_train: {y_train.shape}")
except Exception as e:
    print(f"  ERRO ao carregar tensores: {e}")
    sys.exit(1)

# Os tensores y_*.pt ja estao em escala ln(valor_m2) — sem desnormalizacao
# (a normalizacao [0,1] e aplicada internamente pelo KAN durante treino;
#  os tensores salvos em preprocessed/ mantem a escala original ln)
y_train_ln = y_train
y_val_ln   = y_val
y_test_ln  = y_test

print(f"  Target ln(V/m2) in [{y_train_ln.min():.3f}, {y_train_ln.max():.3f}]")

with open(PREP_DIR / "feature_names.json", encoding="utf-8") as f:
    FEATURES = json.load(f)

with open(PREP_DIR / "scaler_X.pkl", "rb") as f:
    scaler = pickle.load(f)
data_min   = scaler.data_min_
data_range = scaler.data_range_

print(f"  Features ({len(FEATURES)}): {FEATURES}")

# ---------------------------------------------------------------------------
# 3. Ordenacao das features por correlacao com y_train
# ---------------------------------------------------------------------------

print("\nCorrelacoes de Pearson com ln(V/m2):")
correlacoes = {}
for i, feat in enumerate(FEATURES):
    x = X_train[:, i]
    if x.std() > 0:
        corr = float(np.corrcoef(x, y_train_ln)[0, 1])
    else:
        corr = 0.0
    correlacoes[feat] = corr
    print(f"  {feat:<28}  r = {corr:+.4f}")

feature_order = sorted(FEATURES, key=lambda f: abs(correlacoes[f]), reverse=True)
print(f"\nOrdem de ajuste: {feature_order}")

# ---------------------------------------------------------------------------
# 4. Backfitting sequencial (1 passo)
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("BACKFITTING SEQUENCIAL — PySR 1D por feature")
print("="*65)

def bin_smooth_profile(x_feat, y_residual, n_bins=N_BINS, min_pts=MIN_BIN_PTS):
    """
    Suaviza residuo parcial agrupando por percentis de x.
    Para features com poucos valores unicos (ex: variaveis ordinais discretas),
    usa os proprios valores unicos como centros dos bins.
    Retorna (x_centros, y_medias) prontos para PySR.
    """
    unique_vals = np.unique(x_feat)

    # Feature discreta com poucos niveis — usa valores unicos diretamente
    if len(unique_vals) <= 20:
        x_centers, y_means = [], []
        for val in unique_vals:
            mask = np.abs(x_feat - val) < 1e-9
            if mask.sum() >= min_pts:
                x_centers.append(float(val))
                y_means.append(float(y_residual[mask].mean()))
        return np.array(x_centers), np.array(y_means)

    # Feature continua — binning por percentis
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x_feat, quantiles)
    edges = np.unique(edges)

    x_centers, y_means = [], []
    for j in range(len(edges) - 1):
        mask = (x_feat >= edges[j]) & (x_feat < edges[j + 1])
        if j == len(edges) - 2:
            mask = (x_feat >= edges[j]) & (x_feat <= edges[j + 1])
        if mask.sum() >= min_pts:
            x_centers.append(float((edges[j] + edges[j + 1]) / 2))
            y_means.append(float(y_residual[mask].mean()))

    return np.array(x_centers), np.array(y_means)


C          = float(y_train_ln.mean())
residual   = y_train_ln - C   # residuo a ser repartido pelas features

# g_train[feat] = predicao de g_i nos pontos de treino
g_train = {}
formulas    = {}
sr_models   = {}
perfil_data = {}   # para salvar perfis e comparar com KAN

for feat in feature_order:
    i_feat = FEATURES.index(feat)
    x_feat = X_train[:, i_feat]

    rng_x   = x_feat.max() - x_feat.min()
    rng_res = residual.max() - residual.min()

    print(f"\n[{i_feat:2d}] {feat:<28}  |r|={abs(correlacoes[feat]):.4f}  range_residuo={rng_res:.4f}")

    # --- Features binarias: funcao degrau ---
    if feat in BINARY_FEATS or rng_x < 0.01:
        mask0 = x_feat < 0.5
        mask1 = x_feat >= 0.5
        d0 = float(residual[mask0].mean()) if mask0.sum() > 0 else 0.0
        d1 = float(residual[mask1].mean()) if mask1.sum() > 0 else 0.0
        coef  = d1 - d0
        inter = d0
        if abs(coef) < 1e-6:
            expr = "0.0"
        else:
            expr = f"{coef:+.6f} * x + {inter:+.6f}"

        g_hat = np.where(x_feat < 0.5, d0, d1)
        pred_pts = np.array([d0, d1])
        x_pts    = np.array([0.0, 1.0])

        r2_hat = float(1 - np.sum((residual - g_hat)**2) /
                       max(np.sum((residual - residual.mean())**2), 1e-12))
        rmse_hat = float(np.sqrt(np.mean((g_hat - residual)**2)))
        print(f"  => degrau: delta(0)={d0:+.5f}  delta(1)={d1:+.5f}")
        print(f"     RMSE={rmse_hat:.5f}  R2={r2_hat:.4f}")

        formulas[feat] = {
            "expression": expr, "complexity": 4,
            "rmse": rmse_hat, "r2": r2_hat,
            "delta_x0": d0, "delta_x1": d1,
        }
        g_train[feat] = g_hat
        residual      = residual - g_hat
        perfil_data[feat] = {"x_norm": x_pts, "g_norm": pred_pts}
        np.savez(PERF_DIR / f"perfil_{i_feat:02d}_{feat}.npz", x_norm=x_pts, g_norm=pred_pts)
        continue

    # --- Suavizacao por bins ---
    x_sm, y_sm = bin_smooth_profile(x_feat, residual, n_bins=N_BINS)
    if len(x_sm) < 4:
        print(f"  AVISO: poucos bins validos ({len(x_sm)}) — usando media zero")
        g_train[feat] = np.zeros(len(x_feat))
        formulas[feat] = {"expression": "0.0", "complexity": 1, "rmse": 0.0, "r2": 0.0}
        continue

    # --- PySR 1D ---
    pysr_cfg = PYSR_BASE.copy()
    rng_sm   = float(y_sm.max() - y_sm.min())
    if rng_sm > 0.5:
        pysr_cfg["niterations"] = 400
        pysr_cfg["maxsize"]     = 20
    elif rng_sm < 0.05:
        pysr_cfg["niterations"] = 100
        pysr_cfg["maxsize"]     = 8

    try:
        model = PySRRegressor(**pysr_cfg)
        model.fit(x_sm.reshape(-1, 1), y_sm, variable_names=["x"])
        best  = model.get_best()
        expr  = str(best["equation"])
        cmplx = int(best["complexity"])

        # Avalia sobre os pontos suavizados (perfil)
        pred_sm   = model.predict(x_sm.reshape(-1, 1))
        rmse_perf = float(np.sqrt(np.mean((pred_sm - y_sm)**2)))
        r2_perf   = float(1 - np.sum((y_sm - pred_sm)**2) /
                          max(np.sum((y_sm - y_sm.mean())**2), 1e-12))

        # Interpola para todos os pontos de treino
        g_hat = np.interp(x_feat, x_sm, pred_sm)

        print(f"  => expressao: {expr}")
        print(f"     complexidade={cmplx}  RMSE_perfil={rmse_perf:.5f}  R2_perfil={r2_perf:.4f}")

        formulas[feat] = {
            "expression": expr, "complexity": cmplx,
            "rmse": rmse_perf, "r2": r2_perf,
        }
        sr_models[feat] = model

        # Salva hall of fame
        try:
            model.equations_.to_csv(HOF_DIR / f"{feat}.csv", index=False)
        except Exception:
            pass

    except Exception as e:
        print(f"  ERRO PySR: {e} — fallback polinomial grau 3")
        coefs = np.polyfit(x_sm, y_sm, 3)
        pred_sm = np.polyval(coefs, x_sm)
        g_hat   = np.interp(x_feat, x_sm, pred_sm)
        expr = (f"{coefs[0]:+.4f}*x**3 {coefs[1]:+.4f}*x**2 "
                f"{coefs[2]:+.4f}*x {coefs[3]:+.4f}")
        formulas[feat] = {
            "expression": expr, "complexity": 10,
            "rmse": float(np.sqrt(np.mean((pred_sm - y_sm)**2))),
            "r2":   float(1 - np.sum((y_sm - pred_sm)**2) /
                         max(np.sum((y_sm - y_sm.mean())**2), 1e-12)),
            "fallback": True,
        }

    g_train[feat]     = g_hat
    residual          = residual - g_hat
    perfil_data[feat] = {"x_norm": x_sm, "g_norm": pred_sm}
    np.savez(PERF_DIR / f"perfil_{i_feat:02d}_{feat}.npz", x_norm=x_sm, g_norm=pred_sm)

# Ajuste fino do intercepto: C + bias_residual
bias = float(residual.mean())
C   += bias
print(f"\nIntercepto ajustado: C = {C:.5f}  (bias_residual={bias:+.5f})")

# ---------------------------------------------------------------------------
# 5. Funcao de previsao geral
# ---------------------------------------------------------------------------

def predict_sr_isolado(X, split_name="", y_true=None):
    """
    Aplica o modelo aditivo simbolico em uma matriz X normalizada.
    Retorna predicoes em escala ln(valor_m2).
    """
    n   = X.shape[0]
    y_p = np.full(n, C)
    for feat in FEATURES:
        if feat not in formulas or feat not in perfil_data:
            continue
        i_feat = FEATURES.index(feat)
        x_vals = X[:, i_feat]
        x_grid = perfil_data[feat]["x_norm"]
        g_grid = perfil_data[feat]["g_norm"]
        y_p   += np.interp(x_vals, x_grid, g_grid, left=g_grid[0], right=g_grid[-1])

    if y_true is not None:
        rmse = float(np.sqrt(np.mean((y_p - y_true)**2)))
        ss_res = np.sum((y_true - y_p)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        r2   = float(1 - ss_res / max(ss_tot, 1e-12))
        mae  = float(np.abs(y_p - y_true).mean())
        # Erro percentual medio no espaco original
        err_pct = float(np.mean(np.abs(np.exp(y_p) - np.exp(y_true)) / np.exp(y_true) * 100))
        if split_name:
            print(f"  {split_name.upper():<8}  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
                  f"MAPE={err_pct:.2f}%  n={n}")
        return y_p, {"r2": r2, "rmse": rmse, "mae": mae, "mape": err_pct, "n": n}
    return y_p, {}

# ---------------------------------------------------------------------------
# 6. Avaliacao em todos os splits
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("AVALIACAO DO MODELO SR ISOLADO")
print("="*65)

splits_metricas = {}
predicoes = {}

splits_data = [
    ("train", X_train, y_train_ln),
    ("val",   X_val,   y_val_ln),
    ("test",  X_test,  y_test_ln),
]

for sname, X_s, y_s in splits_data:
    y_p, metr = predict_sr_isolado(X_s, split_name=sname, y_true=y_s)
    splits_metricas[sname] = metr
    predicoes[sname]       = {"y_true": y_s, "y_pred": y_p}

# Salva predicoes
np.savez(
    OUT_DIR / "predicoes_sr_isolado.npz",
    y_train_true=predicoes["train"]["y_true"],
    y_train_pred=predicoes["train"]["y_pred"],
    y_val_true=predicoes["val"]["y_true"],
    y_val_pred=predicoes["val"]["y_pred"],
    y_test_true=predicoes["test"]["y_true"],
    y_test_pred=predicoes["test"]["y_pred"],
)
print("\nPredicoes salvas em resultados_sr_isolado/predicoes_sr_isolado.npz")

# ---------------------------------------------------------------------------
# 7. Formula aditiva — escala normalizada
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("FORMULA ADITIVA (espaco normalizado)")
print("="*65)

linhas_norm = ["ln(V/m2) =", f"  {C:+.5f}  # intercepto"]
for feat in FEATURES:
    if feat not in formulas:
        continue
    expr   = formulas[feat]["expression"]
    i_feat = FEATURES.index(feat)
    linhas_norm.append(
        f"  + ({expr})  # {feat}  "
        f"[x = (x_orig - {data_min[i_feat]:.2f}) / {data_range[i_feat]:.2f}]"
    )

formula_norm = "\n".join(linhas_norm)
print(formula_norm)

with open(OUT_DIR / "formula_final_isolado.txt", "w", encoding="utf-8") as f:
    f.write(formula_norm)

# ---------------------------------------------------------------------------
# 8. Formula em escala original
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("FORMULA (escala original das variaveis)")
print("="*65)

linhas_orig = ["ln(valor_mercado / area_construida_m2) =", f"  {C:+.5f}"]
for feat in FEATURES:
    if feat not in formulas:
        continue
    expr   = formulas[feat]["expression"]
    i_feat = FEATURES.index(feat)
    dmin   = data_min[i_feat]
    drng   = data_range[i_feat]
    subst  = f"(({feat} - {dmin:.4f}) / {drng:.4f})" if (drng != 1.0 or dmin != 0.0) else feat
    expr_orig = re.sub(r'\bx\b', subst, expr)
    linhas_orig.append(f"  + [{expr_orig}]")

formula_orig = "\n".join(linhas_orig)
print(formula_orig)

with open(OUT_DIR / "formula_original_isolado.txt", "w", encoding="utf-8") as f:
    f.write(formula_orig)

# ---------------------------------------------------------------------------
# 9. Metricas consolidadas
# ---------------------------------------------------------------------------

metricas_out = {
    "intercepto": C,
    "metodo": "SR_ISOLADO_ADITIVO_1D",
    "descricao": "PySR treinado diretamente nos dados via backfitting sequencial",
    "formulas": formulas,
    "metricas": splits_metricas,
    "correlacoes_pearson": {f: round(v, 4) for f, v in correlacoes.items()},
    "feature_order": feature_order,
}

with open(OUT_DIR / "metricas_sr_isolado.json", "w", encoding="utf-8") as f:
    json.dump(metricas_out, f, indent=2, ensure_ascii=False)

print("\n" + "="*65)
print("RESULTADOS FINAIS")
print("="*65)
for sname, metr in splits_metricas.items():
    print(f"  {sname.upper():<8}  R2={metr['r2']:.4f}  RMSE={metr['rmse']:.4f}  "
          f"MAE={metr['mae']:.4f}  MAPE={metr['mape']:.2f}%")

# Verifica NBR 14.653-2 Grau II no teste
r2_test = splits_metricas["test"]["r2"]
rmse_test = splits_metricas["test"]["rmse"]
err_test  = splits_metricas["test"]["mape"]
ok_r2   = r2_test >= 0.64
ok_err  = err_test <= 30.0
print(f"\n  NBR 14.653-2 Grau II (conjunto teste):")
print(f"    R2 >= 0.64:  {r2_test:.4f}  {'OK' if ok_r2 else 'NAO ATENDE'}")
print(f"    Erro <= 30%: {err_test:.2f}%  {'OK' if ok_err else 'NAO ATENDE'}")
print(f"    => {'ATENDE GRAU II' if (ok_r2 and ok_err) else 'NAO ATENDE GRAU II'}")

metricas_out["nbr14653_grau2_teste"] = {
    "r2_teste": r2_test, "mape_teste": err_test,
    "ok_r2": ok_r2, "ok_erro": ok_err,
    "atende_grau2": bool(ok_r2 and ok_err),
}
with open(OUT_DIR / "metricas_sr_isolado.json", "w", encoding="utf-8") as f:
    json.dump(metricas_out, f, indent=2, ensure_ascii=False)

print(f"\nArquivos salvos em: {OUT_DIR.resolve()}")
print("  formulas_isolado.json        (renomeado de metricas_sr_isolado.json)")
print("  formula_final_isolado.txt")
print("  formula_original_isolado.txt")
print("  predicoes_sr_isolado.npz")
print("  perfis/                      (perfis .npz por feature)")
print("  hall_of_fame/                (equacoes PySR por feature)")
print("\n04_sr_isolado.py CONCLUIDO.")
