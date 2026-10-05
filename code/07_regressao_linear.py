"""
07_regressao_linear.py
======================
Regressão Linear Múltipla (OLS) com transformações Box-Cox.
Modelo de referência para comparação com KAN, KAN+SR MV e SR Isolado.

Pipeline:
  1. Carrega dados originais (base_modelagem_kan.csv) com splits de treino/val/teste
  2. Analisa Box-Cox para features contínuas e para o target
  3. Constrói modelo completo (todas as features transformadas)
  4. Seleção stepwise por AIC (forward + backward)
  5. Diagnósticos: Breusch-Pagan, White, Shapiro-Wilk, Jarque-Bera, VIF, Durbin-Watson
  6. Avalia R², RMSE, MAE, MAPE em treino/val/teste
  7. Salva predições, métricas, coeficientes e relatório de diagnóstico

Saída:
  resultados_ols/
    predicoes_ols.npz
    metricas_ols.json
    coeficientes.csv
    diagnosticos.json
    formula_ols.txt
    boxcox_lambdas.json

Executar com:
  set KMP_DUPLICATE_LIB_OK=TRUE
  python 07_regressao_linear.py
"""

import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import boxcox1p, inv_boxcox1p

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# 0. Verificação de dependências
# ---------------------------------------------------------------------------
try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import (
        het_breuschpagan, het_white, linear_reset
    )
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    print("statsmodels: OK")
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "statsmodels", "--quiet"])
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------------------------
# 1. Caminhos
# ---------------------------------------------------------------------------
PREP_DIR = Path("preprocessed")
OUT_DIR  = Path("resultados_ols")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Carrega dados originais e splits
# ---------------------------------------------------------------------------
print("\nCarregando dados ...")
df = pd.read_csv("base_modelagem_kan.csv", sep=None, engine="python")
print(f"  base_modelagem_kan.csv: {df.shape}")

with open(PREP_DIR / "splits_index.json") as f:
    idx = json.load(f)
idx_train = np.array(idx["idx_train"])
idx_val   = np.array(idx["idx_val"])
idx_test  = np.array(idx["idx_test"])

# Features e target na escala original
FEATURES_CONT = [          # features contínuas — candidatas a Box-Cox
    "area_construida_m2",
    "area_terreno_m2",
    "valor_m2_terreno",
    "valor_m2_construcao",
    "qtd_pavimentos",
    "fator_obsolescencia",
    "prop_financiamento",
]
FEATURES_OUTROS = [        # features sem transformação Box-Cox
    "padrao_ordinal",      # ordinal 1-6
    "cep_norm",            # [0,1]
    "t_ordinal",           # temporal
    "flag_financiamento",  # binária
    "flag_subdeclaracao",  # binária
]
ALL_FEATURES = FEATURES_CONT + FEATURES_OUTROS
TARGET = "ln_valor_m2"

# ---------------------------------------------------------------------------
# 3. Análise Box-Cox para o target e features contínuas
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("ANÁLISE BOX-COX")
print("="*60)

df_train = df.iloc[idx_train].copy()

# Target: ln(valor_m2) — verifica se λ≈0 é ótimo
y_train_orig = np.exp(df_train[TARGET].values)  # valor/m² em R$
_, lam_target = stats.boxcox(y_train_orig[y_train_orig > 0])
print(f"\nTarget (valor_m2):")
print(f"  λ_BoxCox = {lam_target:.4f} (λ≈0 confirma transformação logarítmica)")
print(f"  => Mantém target = ln(valor_m2) [já aplicado]")

# Features contínuas
lambdas = {}
print("\nFeatures contínuas:")
for feat in FEATURES_CONT:
    x = df_train[feat].values
    # Garante positividade (shift se necessário)
    shift = 0.0
    if x.min() <= 0:
        shift = abs(x.min()) + 1.0
        x = x + shift
    # Apenas para valores com variância
    if x.std() < 1e-10:
        lambdas[feat] = {"lambda": 1.0, "shift": shift, "transform": "none"}
        print(f"  {feat:<28} sem variancia — sem transformacao")
        continue
    try:
        _bc, lam = stats.boxcox(x)
        # Interpreta lambda
        if abs(lam) < 0.1:
            tname = "log"
        elif abs(lam - 0.5) < 0.15:
            tname = "sqrt"
        elif abs(lam - 1.0) < 0.15:
            tname = "none (linear)"
        elif abs(lam + 1.0) < 0.15:
            tname = "1/x"
        elif abs(lam - 2.0) < 0.15:
            tname = "square"
        else:
            tname = f"x^{lam:.3f}"
        lambdas[feat] = {"lambda": float(lam), "shift": float(shift), "transform": tname}
        print(f"  {feat:<28} λ={lam:+.4f}  => {tname}  (shift={shift:.1f})")
    except Exception as e:
        lambdas[feat] = {"lambda": 1.0, "shift": shift, "transform": "none"}
        print(f"  {feat:<28} erro Box-Cox: {e}")

with open(OUT_DIR / "boxcox_lambdas.json", "w") as f:
    json.dump(lambdas, f, indent=2)

# ---------------------------------------------------------------------------
# 4. Aplica transformações
# ---------------------------------------------------------------------------
def apply_boxcox(x, lam, shift=0.0):
    """Aplica Box-Cox com shift."""
    x = np.asarray(x, dtype=float) + shift
    x = np.clip(x, 1e-10, None)
    if abs(lam) < 0.05:
        return np.log(x)
    return (x**lam - 1) / lam

def build_matrix(df_subset):
    """Constrói a matriz de features transformadas."""
    rows = []
    for idx_ in df_subset.index:
        row = {}
        # Features contínuas com Box-Cox
        for feat in FEATURES_CONT:
            val = df_subset.loc[idx_, feat]
            info = lambdas[feat]
            lam  = info["lambda"]
            shift= info["shift"]
            tval = apply_boxcox(np.array([val]), lam, shift)[0]
            row[f"{feat}_bc"] = tval
        # Features sem transformação
        for feat in FEATURES_OUTROS:
            row[feat] = float(df_subset.loc[idx_, feat])
        rows.append(row)
    return pd.DataFrame(rows)

print("\nConstruindo matrizes de features ...")
df_val  = df.iloc[idx_val].copy()
df_test = df.iloc[idx_test].copy()

X_train_bc = build_matrix(df_train)
X_val_bc   = build_matrix(df_val)
X_test_bc  = build_matrix(df_test)

y_train = df_train[TARGET].values
y_val   = df_val[TARGET].values
y_test  = df_test[TARGET].values

print(f"  X_train_bc: {X_train_bc.shape}")
print(f"  Colunas: {list(X_train_bc.columns)}")

# Remove NaN/Inf
def clean(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    col_means = X.mean()
    X = X.fillna(col_means)
    return X

X_train_bc = clean(X_train_bc)
X_val_bc   = clean(X_val_bc)
X_test_bc  = clean(X_test_bc)

# ---------------------------------------------------------------------------
# 5. Modelo completo (todas as features transformadas)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("MODELO COMPLETO (OLS — todas as features)")
print("="*60)

X_train_sm = sm.add_constant(X_train_bc.values)
modelo_completo = sm.OLS(y_train, X_train_sm).fit()

print(modelo_completo.summary())
print(f"\n  R² treino:      {modelo_completo.rsquared:.4f}")
print(f"  R² ajustado:    {modelo_completo.rsquared_adj:.4f}")
print(f"  AIC:            {modelo_completo.aic:.2f}")
print(f"  BIC:            {modelo_completo.bic:.2f}")

col_names = ["const"] + list(X_train_bc.columns)

# VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = col_names[1:]
vif_data["VIF"] = [
    variance_inflation_factor(X_train_sm, i+1)
    for i in range(X_train_bc.shape[1])
]
print("\nVIF (modelo completo):")
print(vif_data.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Seleção Stepwise por AIC
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SELEÇÃO STEPWISE — AIC (forward + backward)")
print("="*60)

def stepwise_ols(X_df, y, col_names_in, direction="both", max_vif=10.0):
    """
    Seleção stepwise por AIC com remoção de variáveis com VIF alto.
    Retorna lista de variáveis selecionadas e modelo final.
    """
    included = list(col_names_in)
    best_aic = np.inf
    changed  = True

    # Fase forward: adiciona variáveis
    if direction in ("forward", "both"):
        included = []
        remaining = list(col_names_in)
        while remaining:
            best_feat = None
            for feat in remaining:
                trial = included + [feat]
                X_t = sm.add_constant(X_df[trial].values)
                try:
                    m = sm.OLS(y, X_t).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_feat = feat
                except Exception:
                    pass
            if best_feat:
                included.append(best_feat)
                remaining.remove(best_feat)
            else:
                break

    # Fase backward: remove variáveis com p>0.05 ou VIF alto
    if direction in ("backward", "both"):
        changed = True
        while changed:
            changed = False
            X_t = sm.add_constant(X_df[included].values)
            m   = sm.OLS(y, X_t).fit()
            pvals = pd.Series(m.pvalues[1:], index=included)
            worst_pval = pvals.idxmax()
            if pvals[worst_pval] > 0.05:
                print(f"  Removendo {worst_pval} (p={pvals[worst_pval]:.4f})")
                included.remove(worst_pval)
                changed = True
                continue
            # Checa VIF
            if len(included) > 1:
                X_vif = sm.add_constant(X_df[included].values)
                vifs = [variance_inflation_factor(X_vif, i+1)
                        for i in range(len(included))]
                if max(vifs) > max_vif:
                    worst_vif_idx = np.argmax(vifs)
                    worst_vif_feat = included[worst_vif_idx]
                    print(f"  Removendo {worst_vif_feat} (VIF={max(vifs):.2f})")
                    included.remove(worst_vif_feat)
                    changed = True

    X_final = sm.add_constant(X_df[included].values)
    modelo_final = sm.OLS(y, X_final).fit()
    return included, modelo_final

selected_features, modelo_step = stepwise_ols(
    X_train_bc, y_train, list(X_train_bc.columns)
)

print(f"\nVariáveis selecionadas ({len(selected_features)}): {selected_features}")
print(modelo_step.summary())
print(f"\n  R² treino:      {modelo_step.rsquared:.4f}")
print(f"  R² ajustado:    {modelo_step.rsquared_adj:.4f}")
print(f"  AIC:            {modelo_step.aic:.2f}")

# VIF final
if len(selected_features) > 1:
    X_sel_sm = sm.add_constant(X_train_bc[selected_features].values)
    vif_final = pd.DataFrame({
        "Feature": selected_features,
        "VIF": [variance_inflation_factor(X_sel_sm, i+1)
                for i in range(len(selected_features))]
    })
    print("\nVIF (modelo stepwise):")
    print(vif_final.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Diagnósticos do modelo selecionado
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("DIAGNÓSTICOS ESTATÍSTICOS")
print("="*60)

residuos   = modelo_step.resid
y_fit      = modelo_step.fittedvalues
X_sel_diag = sm.add_constant(X_train_bc[selected_features].values)

# 7.1 Homocedasticidade
bp_lm, bp_pval, bp_f, bp_fpval = het_breuschpagan(residuos, X_sel_diag)
print(f"\nBreusch-Pagan (homocedasticidade):")
print(f"  LM={bp_lm:.4f}  p={bp_pval:.4f}  {'HOMOC.' if bp_pval>0.05 else 'HETEROC. (p<0.05)'}")

try:
    w_lm, w_pval, _, _ = het_white(residuos, X_sel_diag)
    print(f"White test:")
    print(f"  LM={w_lm:.4f}  p={w_pval:.4f}  {'OK' if w_pval>0.05 else 'Heterocedasticidade detectada'}")
except Exception as e:
    w_lm, w_pval = np.nan, np.nan
    print(f"  White test: {e}")

# 7.2 Normalidade dos resíduos
res_std = (residuos - residuos.mean()) / residuos.std()
sample  = res_std[:5000] if len(res_std) > 5000 else res_std
sw_stat, sw_pval = stats.shapiro(sample)
print(f"\nShapiro-Wilk (normalidade dos resíduos, n={len(sample)}):")
print(f"  W={sw_stat:.4f}  p={sw_pval:.4f}  {'NORMAL' if sw_pval>0.05 else 'NAO-NORMAL (p<0.05)'}")

jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(residuos)
print(f"Jarque-Bera:")
print(f"  JB={jb_stat:.4f}  p={jb_pval:.4f}  skew={jb_skew:.4f}  kurt={jb_kurt:.4f}")

# 7.3 Autocorrelação
dw = durbin_watson(residuos)
print(f"\nDurbin-Watson (autocorrelação):")
print(f"  DW={dw:.4f}  (ref: 2.0 = sem autocorrelação; <1.5 ou >2.5 = suspeito)")

# 7.4 Multicolinearidade — VIF já calculado acima

# 7.5 Normalidade Q-Q skewness
print(f"\nEstatísticas dos resíduos:")
print(f"  média={residuos.mean():.6f}  std={residuos.std():.4f}")
print(f"  skew={float(stats.skew(residuos)):.4f}  kurt={float(stats.kurtosis(residuos)):.4f}")

diag_out = {
    "breusch_pagan": {"LM": float(bp_lm), "p": float(bp_pval),
                      "homoscedastico": bool(bp_pval > 0.05)},
    "white":         {"LM": float(w_lm) if not np.isnan(w_lm) else None,
                      "p": float(w_pval) if not np.isnan(w_pval) else None},
    "shapiro_wilk":  {"W": float(sw_stat), "p": float(sw_pval),
                      "normal": bool(sw_pval > 0.05)},
    "jarque_bera":   {"JB": float(jb_stat), "p": float(jb_pval),
                      "skew": float(jb_skew), "kurt": float(jb_kurt)},
    "durbin_watson": {"DW": float(dw)},
    "residuos":      {"mean": float(residuos.mean()), "std": float(residuos.std()),
                      "skew": float(stats.skew(residuos)),
                      "kurt": float(stats.kurtosis(residuos))},
}

# ---------------------------------------------------------------------------
# 8. Avaliação em treino / val / teste
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("AVALIAÇÃO DO MODELO STEPWISE")
print("="*60)

def pred_ols(X_df):
    X = sm.add_constant(X_df[selected_features].values, has_constant='add')
    if X.shape[1] != len(modelo_step.params):
        # Adiciona colunas faltantes
        diff = len(modelo_step.params) - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], diff))])
    return modelo_step.predict(X)

def metricas_ols(y_true, y_pred, label=""):
    res   = y_pred - y_true
    ss_r  = np.sum(res**2)
    ss_t  = np.sum((y_true - y_true.mean())**2)
    r2    = float(1 - ss_r / max(ss_t, 1e-12))
    rmse  = float(np.sqrt(np.mean(res**2)))
    mae   = float(np.abs(res).mean())
    mape  = float(np.mean(np.abs(np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true)) * 100)
    if label:
        print(f"  {label:<10}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%")
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "n": len(y_true)}

preds_ols = {}
metr_ols  = {}

for sname, X_df, y_true in [
        ("treino",   X_train_bc, y_train),
        ("val",      X_val_bc,   y_val),
        ("teste",    X_test_bc,  y_test)]:
    yp = pred_ols(X_df)
    preds_ols[sname] = yp
    metr_ols[sname]  = metricas_ols(y_true, yp, sname)

# NBR 14.653-2 Grau II
m_te    = metr_ols["teste"]
ok_r2   = m_te["r2"]   >= 0.64
ok_mape = m_te["mape"] <= 30.0
print(f"\n  NBR 14.653-2 Grau II (teste):")
print(f"    R²  >= 0.64:  {m_te['r2']:.4f}  {'OK' if ok_r2 else 'NAO ATENDE'}")
print(f"    Erro <= 30%: {m_te['mape']:.2f}%  {'OK' if ok_mape else 'NAO ATENDE'}")
print(f"    => {'ATENDE GRAU II' if (ok_r2 and ok_mape) else 'NAO ATENDE GRAU II'}")

# ---------------------------------------------------------------------------
# 9. Fórmula e coeficientes
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("COEFICIENTES DO MODELO STEPWISE")
print("="*60)

coef_names = ["const"] + selected_features
coefs = pd.DataFrame({
    "Feature": coef_names,
    "Coeficiente": modelo_step.params,
    "Std_Error": modelo_step.bse,
    "t_stat": modelo_step.tvalues,
    "p_valor": modelo_step.pvalues,
})
coefs.to_csv(OUT_DIR / "coeficientes.csv", index=False)
print(coefs.to_string(index=False))

# Fórmula textual
formula_lines = [f"ln(V/m²) = {modelo_step.params[0]:+.6f}"]
for i, feat in enumerate(selected_features):
    info = lambdas.get(feat.replace("_bc",""), None)
    coef = modelo_step.params[i+1]
    # Rotulo coerente com apply_boxcox(): log somente quando |lambda| < 0,05.
    # (Versao anterior rotulava como log qualquer lambda < 0,1, inclusive negativos,
    #  e como sqrt lambda ~ 0,5 - rotulos que nao correspondem ao calculo efetivo.)
    if info and abs(info["lambda"]) < 0.05:
        tname = f"log({feat.replace('_bc','')})"
    elif info:
        tname = f"BoxCox({feat.replace('_bc','')}, λ={info['lambda']:.3f})"
    else:
        tname = feat
    formula_lines.append(f"  {coef:+.6f} × {tname}")

formula_lines.append("")
formula_lines.append("onde BoxCox(x, λ) = (x^λ − 1) / λ   e   log = logaritmo natural")
formula_txt = "\n".join(formula_lines)
print("\n" + formula_txt)
with open(OUT_DIR / "formula_ols.txt", "w", encoding="utf-8") as f:
    f.write(formula_txt)

# ---------------------------------------------------------------------------
# 10. Salva predições e métricas
# ---------------------------------------------------------------------------
np.savez(
    OUT_DIR / "predicoes_ols.npz",
    y_train_true=y_train, y_train_pred=preds_ols["treino"],
    y_val_true=y_val,     y_val_pred=preds_ols["val"],
    y_test_true=y_test,   y_test_pred=preds_ols["teste"],
)

metricas_json = {
    "metodo":            "OLS_BOXCOX_STEPWISE_AIC",
    "features_selecionadas": selected_features,
    "n_features":        len(selected_features),
    "r2_treino_ajustado": float(modelo_step.rsquared_adj),
    "aic":               float(modelo_step.aic),
    "bic":               float(modelo_step.bic),
    "metricas":          metr_ols,
    "diagnosticos":      diag_out,
    "nbr14653_grau2_teste": {
        "r2_teste": m_te["r2"], "mape_teste": m_te["mape"],
        "ok_r2": ok_r2, "ok_mape": ok_mape,
        "atende": bool(ok_r2 and ok_mape),
    },
    "boxcox_lambdas":    lambdas,
    "coeficientes":      coefs.to_dict(orient="records"),
}
with open(OUT_DIR / "metricas_ols.json", "w", encoding="utf-8") as f:
    json.dump(metricas_json, f, indent=2, ensure_ascii=False)

print(f"\nArquivos em: {OUT_DIR.resolve()}")
print("  predicoes_ols.npz")
print("  metricas_ols.json")
print("  coeficientes.csv")
print("  formula_ols.txt")
print("  boxcox_lambdas.json")
print("\n07_regressao_linear.py CONCLUÍDO.")
