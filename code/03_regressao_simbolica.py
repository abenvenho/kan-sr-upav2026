"""
03_regressao_simbolica.py
=========================
Regressao Simbolica (PySR) sobre os perfis marginais aprendidos pelo KAN.

Estrategia aditiva:
    ln(V/m2) = C + sum_i g_i(x_i_norm)

onde g_i sao expressoes simbolicas ajustadas a cada perfil marginal
e x_i_norm = (x_i_orig - min_i) / range_i  (MinMax normalizado).

Entrada : modelos/ativacoes_pysr/  (perfis .npz gerados pelo 02)
          preprocessed/scaler_X.pkl  (para desnormalizacao)
          preprocessed/feature_names.json
          modelos/predicoes.npz       (opcional — para avaliacao vs KAN)
Saidas  : resultados_sr/
              formulas.json           expressoes simbolicas por feature
              formula_final.txt       formula aditiva completa
              formula_original.txt    formula com variaveis em escala original
              metricas_sr.json        R2, RMSE do modelo simbolico
              hall_of_fame/           tabelas PySR por feature
"""

import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Configuracoes
# ---------------------------------------------------------------------------

ATIV_DIR  = Path("modelos/ativacoes_pysr")
PREP_DIR  = Path("preprocessed")
OUT_DIR   = Path("resultados_sr")
HOF_DIR   = OUT_DIR / "hall_of_fame"

OUT_DIR.mkdir(parents=True, exist_ok=True)
HOF_DIR.mkdir(parents=True, exist_ok=True)

# PySR — configuracoes por tipo de feature
PYSR_BASE = dict(
    binary_operators   = ["+", "-", "*", "/"],
    unary_operators    = ["log", "sqrt", "square", "cube", "exp"],
    niterations        = 200,
    maxsize            = 15,
    parsimony          = 5e-4,    # penalidade de complexidade
    model_selection    = "best",  # melhor tradeoff complexidade/ajuste
    verbosity          = 0,
    random_state       = 42,
    progress           = False,
    warm_start         = False,
)

# Limitar operadores para features binárias (0/1) — expressoes triviais
BINARY_FEATS = {"flag_financiamento", "flag_subdeclaracao"}

# ---------------------------------------------------------------------------
# 1. Verificacao do PySR
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
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("\nERRO: Nao foi possivel importar pysr.")
        print("Instale manualmente:")
        print("  pip install pysr")
        print("  python -c \"import pysr; pysr.install()\"")
        print("\nNote: PySR requer Julia. Se Julia nao estiver instalado:")
        print("  Baixe em https://julialang.org/downloads/")
        print("  Ou via winget: winget install julia -s msstore")
        sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Carga dos perfis e metadados
# ---------------------------------------------------------------------------

with open(PREP_DIR / "feature_names.json", encoding="utf-8") as f:
    FEATURES = json.load(f)

with open(Path("modelos/config_kan.json"), encoding="utf-8") as f:
    cfg_kan = json.load(f)
y_min   = cfg_kan["y_norm"]["y_min"]
y_max   = cfg_kan["y_norm"]["y_max"]
y_range = cfg_kan["y_norm"]["y_range"]

with open(PREP_DIR / "scaler_X.pkl", "rb") as f:
    scaler = pickle.load(f)
data_min   = scaler.data_min_
data_range = scaler.data_range_

print(f"\nCarregando {len(FEATURES)} perfis marginais ...")
perfis = {}
for i, feat in enumerate(FEATURES):
    fpath = ATIV_DIR / f"perfil_{i:02d}_{feat}.npz"
    if not fpath.exists():
        print(f"  AVISO: {fpath.name} nao encontrado — pulando {feat}")
        continue
    d = np.load(fpath)
    x_norm = d["x_norm"].astype(np.float64)   # 300 pontos em [0,1]
    y_ln   = d["y_ln"].astype(np.float64)      # ln(valor_m2) correspondente
    perfis[feat] = {"x": x_norm, "y": y_ln, "idx": i}
    print(f"  [{i:2d}] {feat:<28}  range_y={y_ln.max()-y_ln.min():.4f}")

# ---------------------------------------------------------------------------
# 3. Decomposicao aditiva: delta_i = y_i(x) - y_i(x=0.5)
#
# Modelo aditivo:  ln(V/m2) = C + sum_i delta_i(x_i_norm)
# Intercept C = valor do perfil medio (aproximadamente mean(y_train))
# ---------------------------------------------------------------------------

# Calcula o intercept como media ponderada dos perfis no ponto central x=0.5
# (aproximacao: assumimos que a media de cada feature normalizada ~ 0.5)
x_center = 0.5
intercept = 0.0
n_valid = 0
for feat, p in perfis.items():
    idx_center = np.argmin(np.abs(p["x"] - x_center))
    intercept += p["y"][idx_center]
    n_valid   += 1

# O intercept correto para o modelo aditivo e:
# C = y_kan_at_all_means - sum_i y_i(x=0.5) + sum_i y_i(x=0.5)
# Como nao temos y_kan_at_all_means, usamos a media da variavel dependente
# como estimativa. Sera ajustado no final pela equacao de balanco.
try:
    with np.load("modelos/predicoes.npz") as preds:
        y_train_true = preds["y_train_true"]
        intercept = float(y_train_true.mean())
        print(f"\nIntercept (media treino): {intercept:.4f}")
except FileNotFoundError:
    # Estimativa: media das projecoes no ponto central
    intercept = sum(
        perfis[f]["y"][np.argmin(np.abs(perfis[f]["x"] - x_center))]
        for f in perfis
    ) / n_valid
    print(f"\nIntercept (estimado pelos perfis): {intercept:.4f}")

# Deltas: efeito de cada feature relativo ao valor central
deltas = {}
for feat, p in perfis.items():
    idx_center = np.argmin(np.abs(p["x"] - x_center))
    y_center   = p["y"][idx_center]
    deltas[feat] = {"x": p["x"], "delta": p["y"] - y_center, "y_center": y_center}

# ---------------------------------------------------------------------------
# 4. Regressao Simbolica por feature
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("REGRESSAO SIMBOLICA — PySR (1D por feature)")
print("="*60)

formulas      = {}
sr_models     = {}
delta_preds   = {}

for feat in FEATURES:
    if feat not in deltas:
        print(f"\n  [{feat}] PULADO (perfil ausente)")
        continue

    x_data = deltas[feat]["x"].reshape(-1, 1)   # (300, 1)
    y_data = deltas[feat]["delta"]               # (300,)
    rng    = y_data.max() - y_data.min()

    print(f"\n  [{FEATURES.index(feat):2d}] {feat:<28}  range_delta={rng:.4f}")

    # Features binarias: funcao degrau — avalia delta nos extremos x=0 e x=1
    if feat in BINARY_FEATS or rng < 0.001:
        # Valor do delta nos pontos reais de uma feature binaria
        idx0 = 0                            # x = 0.0
        idx1 = len(deltas[feat]["x"]) - 1  # x = 1.0
        d0 = float(deltas[feat]["delta"][idx0])
        d1 = float(deltas[feat]["delta"][idx1])
        # Expressao: degrau centrado em x=0.5
        # delta = d0 + (d1 - d0) * x  (linear entre os dois extremos)
        coef  = d1 - d0
        inter = d0
        if abs(coef) < 1e-6:
            expr = "0.0"
        elif abs(inter) < 1e-6:
            expr = f"{coef:+.6f} * x"
        else:
            expr = f"{coef:+.6f} * x + {inter:+.6f}"
        pred_bin = d0 + (d1 - d0) * x_data.ravel()
        rmse_bin = float(np.sqrt(np.mean((pred_bin - y_data)**2)))
        r2_bin   = float(1 - np.sum((y_data - pred_bin)**2) /
                         np.sum((y_data - y_data.mean())**2)) if y_data.std() > 0 else 1.0
        print(f"    => funcao degrau: delta(0)={d0:+.4f}  delta(1)={d1:+.4f}  expr={expr}")
        print(f"       RMSE={rmse_bin:.5f}  R2={r2_bin:.4f}")
        formulas[feat] = {
            "expression": expr,
            "complexity": 4,
            "rmse": rmse_bin,
            "r2":   r2_bin,
            "delta_x0": d0,
            "delta_x1": d1,
        }
        delta_preds[feat] = pred_bin
        continue

    # Configuracao PySR adaptada ao tipo de feature
    pysr_cfg = PYSR_BASE.copy()
    if rng > 0.5:  # features de alta influencia: mais iteracoes
        pysr_cfg["niterations"] = 400
        pysr_cfg["maxsize"]     = 20
    elif rng < 0.05:  # features de baixa influencia: expressao simples
        pysr_cfg["niterations"] = 100
        pysr_cfg["maxsize"]     = 8

    try:
        model = PySRRegressor(**pysr_cfg)
        model.fit(x_data, y_data, variable_names=["x"])
        best = model.get_best()
        expr   = str(best["equation"])
        cmplx  = int(best["complexity"])
        r2_sr  = float(best["score"]) if "score" in best else np.nan
        pred_d = model.predict(x_data)
        rmse_d = float(np.sqrt(np.mean((pred_d - y_data)**2)))
        r2_d   = float(1 - np.sum((y_data - pred_d)**2) / np.sum((y_data - y_data.mean())**2)) \
                 if y_data.std() > 0 else 1.0

        print(f"    => expressao: delta = {expr}")
        print(f"       complexidade={cmplx}  RMSE={rmse_d:.5f}  R2={r2_d:.4f}")

        formulas[feat]    = {"expression": expr, "complexity": cmplx, "rmse": rmse_d, "r2": r2_d}
        sr_models[feat]   = model
        delta_preds[feat] = pred_d

        # Salva hall of fame
        try:
            hof_path = HOF_DIR / f"{feat}.csv"
            model.equations_.to_csv(hof_path, index=False)
        except Exception:
            pass

    except Exception as e:
        print(f"    ERRO PySR em {feat}: {e}")
        # Fallback: regressao polinomial grau 3
        coefs = np.polyfit(x_data.ravel(), y_data, 3)
        pred_d = np.polyval(coefs, x_data.ravel())
        expr = (f"{coefs[0]:+.4f}*x**3 {coefs[1]:+.4f}*x**2 "
                f"{coefs[2]:+.4f}*x {coefs[3]:+.4f}")
        formulas[feat] = {
            "expression":  expr,
            "complexity":  10,
            "rmse":        float(np.sqrt(np.mean((pred_d - y_data)**2))),
            "r2":          float(1 - np.sum((y_data - pred_d)**2) /
                                np.sum((y_data - y_data.mean())**2)),
            "fallback":    True,
        }
        delta_preds[feat] = pred_d

# ---------------------------------------------------------------------------
# 5. Formula aditiva em espaco normalizado
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("FORMULA ADITIVA (espaco normalizado)")
print("="*60)

linhas_norm = ["ln(V/m2) =", f"  {intercept:+.4f}  # intercepto"]
for feat in FEATURES:
    if feat not in formulas:
        continue
    expr  = formulas[feat]["expression"]
    i_feat = FEATURES.index(feat)
    linhas_norm.append(f"  + ({expr})  # {feat}  [x = (x_orig - {data_min[i_feat]:.2f}) / {data_range[i_feat]:.2f}]")

formula_norm = "\n".join(linhas_norm)
print(formula_norm)

with open(OUT_DIR / "formula_final.txt", "w", encoding="utf-8") as f:
    f.write(formula_norm)

# ---------------------------------------------------------------------------
# 6. Formula em escala original
#    Substituicao: x = (x_orig - min) / range
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("FORMULA (escala original das variaveis)")
print("="*60)

linhas_orig = ["ln(valor_mercado / area_construida_m2) =", f"  {intercept:+.4f}"]

for feat in FEATURES:
    if feat not in formulas:
        continue
    expr   = formulas[feat]["expression"]
    i_feat = FEATURES.index(feat)
    dmin   = data_min[i_feat]
    drng   = data_range[i_feat]

    # Substituicao segura: \bx\b garante que so o token 'x' eh substituido,
    # preservando funcoes como exp(), sqrt(), square(), cube()
    subst = f"(({feat} - {dmin:.4f}) / {drng:.4f})" if (drng != 1.0 or dmin != 0.0) else feat
    expr_orig = re.sub(r'\bx\b', subst, expr)

    linhas_orig.append(f"  + [{expr_orig}]")

formula_orig = "\n".join(linhas_orig)
print(formula_orig)

with open(OUT_DIR / "formula_original.txt", "w", encoding="utf-8") as f:
    f.write(formula_orig)

# ---------------------------------------------------------------------------
# 7. Avaliacao do modelo simbolico
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("AVALIACAO DO MODELO SIMBOLICO")
print("="*60)

metricas_sr = {"formulas": formulas, "intercept": intercept}

# Avaliacao contra predicoes KAN e dados reais
pred_path = Path("modelos/predicoes.npz")
if pred_path.exists():
    with np.load(pred_path) as preds:
        print(f"\n  {'Split':<12}  {'RMSE_KAN':>10}  {'R2_KAN':>8}")
        print(f"  {'-'*35}")
        for split in ["train", "val", "test"]:
            y_true = preds[f"y_{split}_true"]
            y_kan  = preds[f"y_{split}_pred"]
            rmse_kan = float(np.sqrt(np.mean((y_kan - y_true)**2)))
            r2_kan   = float(1 - np.sum((y_true - y_kan)**2) / np.sum((y_true - y_true.mean())**2))
            print(f"  {split.upper():<12}  {rmse_kan:>10.4f}  {r2_kan:>8.4f}")
            metricas_sr[split] = {"rmse_kan": rmse_kan, "r2_kan": r2_kan, "n": int(len(y_true))}

    # Tenta reconstruir predicao simbolica no conjunto de treino
    # usando os tensores normalizados salvos
    try:
        import pickle as pkl
        X_path = Path("preprocessed/X_train.pt")
        if X_path.exists():
            # Carrega sem torch se necessario
            try:
                import torch
                X_tr = torch.load(X_path, weights_only=True).numpy()
            except Exception:
                X_tr = None

            if X_tr is not None:
                y_sr = np.full(len(X_tr), intercept)
                for i_f, feat in enumerate(FEATURES):
                    if feat not in formulas or feat not in delta_preds:
                        continue
                    x_vals = X_tr[:, i_f]
                    # Interpola delta para cada amostra usando o perfil PySR
                    x_grid = deltas[feat]["x"]
                    d_grid = delta_preds[feat]
                    y_sr  += np.interp(x_vals, x_grid, d_grid)

                with np.load(pred_path) as p:
                    y_tr_true = p["y_train_true"]
                rmse_sr = float(np.sqrt(np.mean((y_sr - y_tr_true)**2)))
                r2_sr   = float(1 - np.sum((y_tr_true - y_sr)**2) /
                                np.sum((y_tr_true - y_tr_true.mean())**2))
                print(f"\n  Modelo Simbolico (treino):")
                print(f"    RMSE_SR={rmse_sr:.4f}  R2_SR={r2_sr:.4f}")
                metricas_sr["simbolico_treino"] = {"rmse": rmse_sr, "r2": r2_sr}
    except Exception as e:
        print(f"  (avaliacao simbolica ignorada: {e})")
else:
    print("  modelos/predicoes.npz nao encontrado.")
    print("  Re-execute 02_kan_treinamento.py para salvar predicoes.")

# Avaliacao da qualidade de ajuste dos perfis
print("\nQualidade do ajuste PySR por feature (em escala de delta):")
total_rmse = 0.0
n_feats    = 0
for feat in FEATURES:
    if feat in formulas:
        r  = formulas[feat]["r2"]
        rm = formulas[feat]["rmse"]
        print(f"  {feat:<28}  R2={r:.4f}  RMSE={rm:.5f}")
        total_rmse += rm
        n_feats    += 1
if n_feats > 0:
    print(f"\n  RMSE medio dos perfis: {total_rmse/n_feats:.5f}")
    metricas_sr["rmse_medio_perfis"] = round(total_rmse/n_feats, 6)

# Salva metricas
with open(OUT_DIR / "metricas_sr.json", "w", encoding="utf-8") as f:
    json.dump(metricas_sr, f, indent=2, ensure_ascii=False, default=str)

# Salva formulas separadas
with open(OUT_DIR / "formulas.json", "w", encoding="utf-8") as f:
    json.dump(formulas, f, indent=2, ensure_ascii=False, default=str)

# ---------------------------------------------------------------------------
# 8. Resumo final
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("RESUMO REGRESSAO SIMBOLICA")
print("="*60)
print(f"  Features processadas : {n_feats} / {len(FEATURES)}")
print(f"  Hall of fame salvo em: {HOF_DIR}/")
print(f"  Formulas salvas em   : {OUT_DIR}/")
print(f"  formula_final.txt    : formula em espaco normalizado")
print(f"  formula_original.txt : formula em escala original das variaveis")
print("\nProximo passo: 04_validacao_nbr14653.py")
print("OK - Regressao Simbolica concluida.")
