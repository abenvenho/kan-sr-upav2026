"""
05_analise_comparativa.py
=========================
Analise comparativa completa das tres abordagens:
  A. KAN (Kolmogorov-Arnold Network)
  B. KAN + SR destilado (PySR sobre ativacoes KAN)
  C. SR isolado (PySR treinado diretamente nos dados)

Analises geradas
----------------
1. Tabela de metricas consolidada (R2, RMSE, MAE, MAPE) — train/val/test
2. Graficos pred vs observado (scatter) — 3 modelos, conjunto teste
3. Graficos de residuos vs previstos — 3 modelos
4. Distribuicao dos residuos (histograma + curva normal)
5. Q-Q plot dos residuos (conjunto teste)
6. Residuos por padrao construtivo (boxplot) — 3 modelos
7. Perfis marginais: KAN vs KAN+SR vs SR isolado (12 features)
8. Sensibilidade / influencia das variaveis na equacao destilada
9. Comparacao de desempenho (barplot R2 e RMSE)

Entrada
-------
  modelos/predicoes.npz
  modelos/norm_target.json
  resultados_sr/formulas.json
  resultados_sr/metricas_sr.json
  resultados_sr_isolado/metricas_sr_isolado.json
  resultados_sr_isolado/predicoes_sr_isolado.npz
  resultados_sr_isolado/perfis/
  modelos/ativacoes_pysr/
  preprocessed/X_*.pt, y_*.pt, scaler_X.pkl, feature_names.json, splits_index.json
  base_modelagem_kan.csv

Saidas
------
  outputs/analise_comparativa/
    metricas_comparativas.json
    tabela_metricas.csv
    01_pred_vs_obs_teste.png
    02_residuos_vs_previsto.png
    03_distribuicao_residuos.png
    04_qq_plot_residuos.png
    05_residuos_por_padrao.png
    06_perfis_marginais_[feat].png
    07_sensibilidade_sr_destilado.png
    08_sensibilidade_sr_isolado.png
    09_comparacao_metricas.png
    10_residuos_temporais.png

Executar com:
    set KMP_DUPLICATE_LIB_OK=TRUE
    python 05_analise_comparativa.py
"""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend nao-interativo
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Configuracoes e caminhos
# ---------------------------------------------------------------------------

PREP_DIR  = Path("preprocessed")
MOD_DIR   = Path("modelos")
SR_DIR    = Path("resultados_sr")
SRI_DIR   = Path("resultados_sr_isolado")
ATIV_DIR  = MOD_DIR / "ativacoes_pysr"
OUT_DIR   = Path("outputs/analise_comparativa")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Estilo visual
COLORS = {
    "KAN":         "#2563EB",   # azul
    "KAN+SR":      "#16A34A",   # verde
    "SR_isolado":  "#DC2626",   # vermelho
}
ALPHA_SCATTER = 0.25
DPI = 150

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": DPI,
})

# ---------------------------------------------------------------------------
# 1. Carga dos dados
# ---------------------------------------------------------------------------

print("Carregando dados ...")

# -- Tensores --
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
    y_train = lt(PREP_DIR / "y_train.pt").ravel()
    y_val   = lt(PREP_DIR / "y_val.pt").ravel()
    y_test  = lt(PREP_DIR / "y_test.pt").ravel()
except Exception as e:
    print(f"ERRO ao carregar tensores: {e}"); sys.exit(1)

# Os tensores y_*.pt ja estao em escala ln(valor_m2) — sem desnormalizacao
# (a normalizacao [0,1] e aplicada internamente pelo KAN durante treino)
y_tr_ln = y_train
y_va_ln = y_val
y_te_ln = y_test

# -- Features --
with open(PREP_DIR / "feature_names.json", encoding="utf-8") as f:
    FEATURES = json.load(f)
with open(PREP_DIR / "scaler_X.pkl", "rb") as f:
    scaler = pickle.load(f)
data_min   = scaler.data_min_
data_range = scaler.data_range_

# -- Indices originais (para mapear para base_modelagem_kan.csv) --
with open(PREP_DIR / "splits_index.json", encoding="utf-8") as f:
    idx_splits = json.load(f)
idx_test = np.array(idx_splits["idx_test"])

# -- base_modelagem_kan.csv (para padrao_ordinal no test set) --
try:
    df_base = pd.read_csv("base_modelagem_kan.csv", sep=None, engine="python", usecols=["padrao_ordinal", "padrao_letra"])
    padrao_test = df_base.iloc[idx_test]["padrao_ordinal"].values
    padrao_letra_test = df_base.iloc[idx_test]["padrao_letra"].values
    print(f"  base_modelagem_kan.csv carregado — {len(df_base)} registros")
except Exception as e:
    print(f"  AVISO: nao foi possivel carregar padrao_ordinal: {e}")
    padrao_test = np.zeros(len(idx_test), dtype=int)
    padrao_letra_test = np.array(["N/A"] * len(idx_test))

# ---------------------------------------------------------------------------
# 2. Predicoes do KAN (ja em escala ln(valor_m2))
# ---------------------------------------------------------------------------

print("Carregando predicoes KAN ...")
with np.load(MOD_DIR / "predicoes.npz") as pk:
    kan = {
        "train": {"true": pk["y_train_true"].copy(), "pred": pk["y_train_pred"].copy()},
        "val":   {"true": pk["y_val_true"].copy(),   "pred": pk["y_val_pred"].copy()},
        "test":  {"true": pk["y_test_true"].copy(),  "pred": pk["y_test_pred"].copy()},
    }

# ---------------------------------------------------------------------------
# 3. Predicoes KAN+SR destilado
# ---------------------------------------------------------------------------

print("Carregando formulas KAN+SR destilado ...")
with open(SR_DIR / "formulas.json", encoding="utf-8") as f:
    formulas_dest = json.load(f)

# Reconstroi predicao KAN+SR a partir dos perfis + interpolacao
dest_perfis = {}
for i_f, feat in enumerate(FEATURES):
    p = ATIV_DIR / f"perfil_{i_f:02d}_{feat}.npz"
    if p.exists():
        d = np.load(p)
        dest_perfis[feat] = {"x": d["x_norm"], "y": d["y_ln"]}

def predict_kansr(X, y_true_ln):
    """Reconstroi predicao KAN+SR por interpolacao dos perfis."""
    with open(SR_DIR / "metricas_sr.json", encoding="utf-8") as f:
        m = json.load(f)
    C_dest = m.get("intercept", float(y_tr_ln.mean()))

    y_p = np.full(X.shape[0], C_dest)
    for feat, pdata in dest_perfis.items():
        i_f  = FEATURES.index(feat)
        x_g  = pdata["x"]
        y_g  = pdata["y"]
        # delta em relacao ao ponto central
        idx_ctr = np.argmin(np.abs(x_g - 0.5))
        delta_g = y_g - y_g[idx_ctr]
        y_p += np.interp(X[:, i_f], x_g, delta_g, left=delta_g[0], right=delta_g[-1])
    return y_p

kansr = {}
for sname, X_s, y_s_ln in [("train", X_train, y_tr_ln),
                             ("val",   X_val,   y_va_ln),
                             ("test",  X_test,  y_te_ln)]:
    y_p = predict_kansr(X_s, y_s_ln)
    kansr[sname] = {"true": y_s_ln, "pred": y_p}

# ---------------------------------------------------------------------------
# 4. Predicoes SR isolado
# ---------------------------------------------------------------------------

print("Carregando predicoes SR isolado ...")
sri_npz = SRI_DIR / "predicoes_sr_isolado.npz"
if not sri_npz.exists():
    print("  AVISO: predicoes_sr_isolado.npz nao encontrado.")
    print("  Execute 04_sr_isolado.py primeiro.")
    # Cria predicoes vazias para nao quebrar o restante do script
    sri = {
        "train": {"true": y_tr_ln, "pred": np.full_like(y_tr_ln, y_tr_ln.mean())},
        "val":   {"true": y_va_ln, "pred": np.full_like(y_va_ln, y_va_ln.mean())},
        "test":  {"true": y_te_ln, "pred": np.full_like(y_te_ln, y_te_ln.mean())},
    }
    sri_disponivel = False
else:
    with np.load(sri_npz) as ps:
        sri = {
            "train": {"true": ps["y_train_true"], "pred": ps["y_train_pred"]},
            "val":   {"true": ps["y_val_true"],   "pred": ps["y_val_pred"]},
            "test":  {"true": ps["y_test_true"],  "pred": ps["y_test_pred"]},
        }
    sri_disponivel = True

# ---------------------------------------------------------------------------
# 5. Funcoes de metrica
# ---------------------------------------------------------------------------

def metricas(y_true, y_pred):
    res   = y_pred - y_true
    ss_r  = np.sum(res**2)
    ss_t  = np.sum((y_true - y_true.mean())**2)
    r2    = float(1 - ss_r / max(ss_t, 1e-12))
    rmse  = float(np.sqrt(np.mean(res**2)))
    mae   = float(np.abs(res).mean())
    mape  = float(np.mean(np.abs(np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true)) * 100)
    return {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "n": len(y_true)}

# Tabela de metricas
tabela = []
modelos_dict = {"KAN": kan, "KAN+SR": kansr, "SR_isolado": sri}
for modelo, splits in modelos_dict.items():
    for split in ["train", "val", "test"]:
        m = metricas(splits[split]["true"], splits[split]["pred"])
        tabela.append({"modelo": modelo, "split": split, **m})

df_tab = pd.DataFrame(tabela)
df_tab.to_csv(OUT_DIR / "tabela_metricas.csv", index=False)

print("\n" + "="*70)
print("TABELA DE METRICAS COMPARATIVAS")
print("="*70)
print(df_tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------------------
# 6. Grafico 1 — Pred vs Observado (conjunto teste, 3 modelos)
# ---------------------------------------------------------------------------

print("\nGerando graficos ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Valores Previstos vs Observados — Conjunto Teste\n"
             "ln(Valor/m²) | ABNT NBR 14.653-2", fontsize=13, fontweight="bold")

for ax, (nome, splits), cor in zip(axes, modelos_dict.items(), COLORS.values()):
    yt = splits["test"]["true"]
    yp = splits["test"]["pred"]
    m  = metricas(yt, yp)
    lim = [min(yt.min(), yp.min()) - 0.1, max(yt.max(), yp.max()) + 0.1]

    ax.scatter(yt, yp, alpha=ALPHA_SCATTER, s=8, color=cor, rasterized=True)
    ax.plot(lim, lim, "k--", lw=1, label="perfeito")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observado")
    ax.set_ylabel("Previsto" if ax == axes[0] else "")
    ax.set_title(f"{nome}\nR²={m['r2']:.4f}  RMSE={m['rmse']:.4f}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(OUT_DIR / "01_pred_vs_obs_teste.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 7. Grafico 2 — Residuos vs Previsto (heteroscedasticidade)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Resíduos vs Valores Previstos — Conjunto Teste", fontsize=13, fontweight="bold")

for ax, (nome, splits), cor in zip(axes, modelos_dict.items(), COLORS.values()):
    yt  = splits["test"]["true"]
    yp  = splits["test"]["pred"]
    res = yt - yp

    ax.scatter(yp, res, alpha=ALPHA_SCATTER, s=8, color=cor, rasterized=True)
    ax.axhline(0, color="k", lw=1, ls="--")
    # Linha de tendencia (lowess simplificado: media movel)
    order = np.argsort(yp)
    window = max(1, len(yp) // 20)
    trend_x = np.convolve(yp[order], np.ones(window)/window, "valid")
    trend_y = np.convolve(res[order], np.ones(window)/window, "valid")
    ax.plot(trend_x, trend_y, color="black", lw=1.5, label="tendência")
    ax.set_xlabel("Previsto ln(V/m²)")
    ax.set_ylabel("Resíduo" if ax == axes[0] else "")
    ax.set_title(f"{nome}\nstd(res)={res.std():.4f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "02_residuos_vs_previsto.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 8. Grafico 3 — Distribuicao dos residuos (histograma + normal)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Distribuição dos Resíduos — Conjunto Teste", fontsize=13, fontweight="bold")

for ax, (nome, splits), cor in zip(axes, modelos_dict.items(), COLORS.values()):
    yt  = splits["test"]["true"]
    yp  = splits["test"]["pred"]
    res = yt - yp

    ax.hist(res, bins=50, density=True, color=cor, alpha=0.6, edgecolor="white", lw=0.3)
    xg = np.linspace(res.min(), res.max(), 200)
    ax.plot(xg, stats.norm.pdf(xg, res.mean(), res.std()), "k-", lw=1.5, label="normal")
    skew  = float(stats.skew(res))
    kurt  = float(stats.kurtosis(res))
    _, p_sw = stats.shapiro(res[:min(5000, len(res))])
    ax.set_xlabel("Resíduo")
    ax.set_ylabel("Densidade" if ax == axes[0] else "")
    ax.set_title(f"{nome}\nSkew={skew:.3f}  Kurt={kurt:.3f}\nShapiro p={p_sw:.4f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "03_distribuicao_residuos.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 9. Grafico 4 — Q-Q plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Q-Q Plot dos Resíduos Padronizados — Conjunto Teste",
             fontsize=13, fontweight="bold")

for ax, (nome, splits), cor in zip(axes, modelos_dict.items(), COLORS.values()):
    yt  = splits["test"]["true"]
    yp  = splits["test"]["pred"]
    res = (yt - yp)
    res_std = (res - res.mean()) / max(res.std(), 1e-12)

    qq = stats.probplot(res_std, dist="norm")
    ax.scatter(qq[0][0], qq[0][1], alpha=0.3, s=6, color=cor, rasterized=True)
    lim = max(abs(qq[0][0]).max(), abs(qq[0][1]).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlabel("Quantil teórico (Normal)")
    ax.set_ylabel("Quantil amostral" if ax == axes[0] else "")
    ax.set_title(f"{nome}")

plt.tight_layout()
plt.savefig(OUT_DIR / "04_qq_plot_residuos.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 10. Grafico 5 — Residuos por padrao construtivo
# ---------------------------------------------------------------------------

padroes_unicos = sorted(np.unique(padrao_test))
padrao_label   = {p: str(p) for p in padroes_unicos}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Resíduos por Padrão Construtivo — Conjunto Teste\n(1=popular … 6=fino)",
             fontsize=13, fontweight="bold")

for ax, (nome, splits), cor in zip(axes, modelos_dict.items(), COLORS.values()):
    yt  = splits["test"]["true"]
    yp  = splits["test"]["pred"]
    res = yt - yp

    data_bp  = [res[padrao_test == p] for p in padroes_unicos]
    bp = ax.boxplot(data_bp, labels=[str(p) for p in padroes_unicos],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(cor)
        patch.set_alpha(0.5)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Padrão construtivo")
    ax.set_ylabel("Resíduo ln(V/m²)" if ax == axes[0] else "")
    ax.set_title(f"{nome}")

plt.tight_layout()
plt.savefig(OUT_DIR / "05_residuos_por_padrao.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 11. Grafico 6 — Perfis marginais: KAN vs KAN+SR vs SR isolado
# ---------------------------------------------------------------------------

print("  Gerando perfis marginais (12 features) ...")
n_feat = len(FEATURES)
n_cols = 3
n_rows = (n_feat + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
fig.suptitle("Perfis Marginais: KAN vs KAN+SR vs SR Isolado\n(eixo x normalizado [0,1])",
             fontsize=13, fontweight="bold")
axes_flat = axes.flatten()

for idx_f, feat in enumerate(FEATURES):
    ax = axes_flat[idx_f]
    i_f = idx_f

    # KAN (perfil original das ativacoes)
    ativ_path = ATIV_DIR / f"perfil_{i_f:02d}_{feat}.npz"
    if ativ_path.exists():
        d = np.load(ativ_path)
        x_k = d["x_norm"]
        y_k = d["y_ln"]
        y_k_delta = y_k - y_k[np.argmin(np.abs(x_k - 0.5))]
        ax.plot(x_k, y_k_delta, color=COLORS["KAN"], lw=2, label="KAN", alpha=0.85)

    # KAN+SR (interpolacao do perfil via formula)
    if feat in dest_perfis:
        x_d = dest_perfis[feat]["x"]
        y_d = dest_perfis[feat]["y"]
        y_d_delta = y_d - y_d[np.argmin(np.abs(x_d - 0.5))]
        ax.plot(x_d, y_d_delta, color=COLORS["KAN+SR"], lw=1.5, ls="--",
                label="KAN+SR", alpha=0.85)

    # SR isolado (perfil do backfitting)
    sri_path = SRI_DIR / "perfis" / f"perfil_{i_f:02d}_{feat}.npz"
    if sri_path.exists():
        d2 = np.load(sri_path)
        x_si = d2["x_norm"]
        g_si = d2["g_norm"]
        ax.plot(x_si, g_si, color=COLORS["SR_isolado"], lw=1.5, ls=":",
                label="SR isolado", alpha=0.85)

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel("x normalizado", fontsize=8)
    if idx_f % n_cols == 0:
        ax.set_ylabel("Δ ln(V/m²)", fontsize=8)
    ax.tick_params(labelsize=7)

# Legenda global
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

# Remove eixos extras
for idx_extra in range(n_feat, len(axes_flat)):
    axes_flat[idx_extra].set_visible(False)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT_DIR / "06_perfis_marginais.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 12. Grafico 7 e 8 — Sensibilidade / influencia das variaveis
#     (variacao do output ao variar cada feature de 0 a 1 com demais na media)
# ---------------------------------------------------------------------------

def sensitivity_analysis(X_ref, perfis_dict, C_model, feature_list, title, fname):
    """
    Para cada feature: varia de 0 a 1 (normalizados) com as demais fixas na mediana.
    Calcula amplitude de variacao em ln(V/m2) — representa importancia.
    Plota barras ordenadas.
    """
    x_med = np.median(X_ref, axis=0)   # ponto de referencia (mediana)
    amplitudes = {}

    for i_f, feat in enumerate(feature_list):
        if feat not in perfis_dict:
            continue
        x_g = perfis_dict[feat]["x_norm"]
        g_g = perfis_dict[feat]["g_norm"]
        g_min = float(np.interp(0.0, x_g, g_g, left=g_g[0], right=g_g[-1]))
        g_max = float(np.interp(1.0, x_g, g_g, left=g_g[0], right=g_g[-1]))
        amplitudes[feat] = abs(g_max - g_min)

    if not amplitudes:
        return

    amp_sorted = dict(sorted(amplitudes.items(), key=lambda kv: kv[1], reverse=True))
    feats  = list(amp_sorted.keys())
    ampls  = list(amp_sorted.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = [COLORS["KAN+SR"] if "destilado" in fname else COLORS["SR_isolado"]] * len(feats)
    bars = ax.barh(feats[::-1], ampls[::-1], color=colors_bar, alpha=0.75)
    ax.set_xlabel("Amplitude Δ ln(V/m²)  [0→1 normalizado]", fontsize=11)
    ax.set_title(f"Influência das Variáveis — {title}\n"
                 "(variação no output ao mover a feature de 0 a 1)", fontsize=12)
    for bar, val in zip(bars, ampls[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close()


# KAN+SR destilado — perfis delta
dest_perfis_delta = {}
for feat, pdata in dest_perfis.items():
    x_d = pdata["x"]
    y_d = pdata["y"]
    y_ctr = y_d[np.argmin(np.abs(x_d - 0.5))]
    dest_perfis_delta[feat] = {"x_norm": x_d, "g_norm": y_d - y_ctr}

sensitivity_analysis(
    X_train, dest_perfis_delta, None, FEATURES,
    "KAN+SR Destilado", "07_sensibilidade_sr_destilado.png"
)

# SR isolado — perfis do backfitting
sri_perfis = {}
for i_f, feat in enumerate(FEATURES):
    p = SRI_DIR / "perfis" / f"perfil_{i_f:02d}_{feat}.npz"
    if p.exists():
        d = np.load(p)
        sri_perfis[feat] = {"x_norm": d["x_norm"], "g_norm": d["g_norm"]}

sensitivity_analysis(
    X_train, sri_perfis, None, FEATURES,
    "SR Isolado", "08_sensibilidade_sr_isolado.png"
)

# ---------------------------------------------------------------------------
# 13. Grafico 9 — Comparacao de metricas (barplot R2 e RMSE)
# ---------------------------------------------------------------------------

splits_plotar = ["train", "val", "test"]
modelos_nomes = list(modelos_dict.keys())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comparação de Desempenho — KAN vs KAN+SR vs SR Isolado",
             fontsize=13, fontweight="bold")

x = np.arange(len(splits_plotar))
width = 0.25

for ax, metrica, ylabel in zip(axes, ["r2", "rmse"], ["R²", "RMSE ln(V/m²)"]):
    for i_m, (nome, cor) in enumerate(COLORS.items()):
        vals = [metricas(modelos_dict[nome][s]["true"],
                         modelos_dict[nome][s]["pred"])[metrica]
                for s in splits_plotar]
        offset = (i_m - 1) * width
        bars = ax.bar(x + offset, vals, width, label=nome, color=cor, alpha=0.75)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits_plotar])
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend()
    if metrica == "r2":
        ax.axhline(0.64, color="red", lw=1, ls="--", label="Grau II (0.64)")
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig(OUT_DIR / "09_comparacao_metricas.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 14. Grafico 10 — Residuos ao longo do tempo (t_ordinal)
# ---------------------------------------------------------------------------

try:
    t_idx = FEATURES.index("t_ordinal")
    t_test_norm = X_test[:, t_idx]
    # Desnormaliza t_ordinal: x_orig = x_norm * range + min
    t_min  = data_min[t_idx]
    t_rng  = data_range[t_idx]
    t_test = t_test_norm * t_rng + t_min

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Resíduos ao Longo do Tempo (t_ordinal) — Conjunto Teste",
                 fontsize=13, fontweight="bold")

    for ax, (nome, splits), cor in zip(axes, modelos_dict.items(), COLORS.values()):
        yt  = splits["test"]["true"]
        yp  = splits["test"]["pred"]
        res = yt - yp
        ax.scatter(t_test, res, alpha=ALPHA_SCATTER, s=8, color=cor, rasterized=True)
        ax.axhline(0, color="k", lw=1, ls="--")
        # Media movel temporal
        order = np.argsort(t_test)
        w = max(1, len(t_test) // 15)
        tx = np.convolve(t_test[order], np.ones(w)/w, "valid")
        ry = np.convolve(res[order],    np.ones(w)/w, "valid")
        ax.plot(tx, ry, color="black", lw=1.5)
        ax.set_xlabel("t_ordinal (meses)")
        ax.set_ylabel("Resíduo" if ax == axes[0] else "")
        ax.set_title(f"{nome}")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_residuos_temporais.png", dpi=DPI, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"  AVISO: grafico temporal nao gerado: {e}")

# ---------------------------------------------------------------------------
# 15. JSON de metricas consolidado
# ---------------------------------------------------------------------------

metricas_consolidado = {}
for nome, splits in modelos_dict.items():
    metricas_consolidado[nome] = {}
    for split in ["train", "val", "test"]:
        metricas_consolidado[nome][split] = metricas(
            splits[split]["true"], splits[split]["pred"]
        )

# NBR 14.653-2 Grau II (base: conjunto teste)
for nome in metricas_consolidado:
    m_te = metricas_consolidado[nome]["test"]
    ok_r2  = m_te["r2"]   >= 0.64
    ok_err = m_te["mape"] <= 30.0
    metricas_consolidado[nome]["nbr14653_grau2_teste"] = {
        "ok_r2": ok_r2, "ok_mape": ok_err,
        "atende": bool(ok_r2 and ok_err),
    }

with open(OUT_DIR / "metricas_comparativas.json", "w", encoding="utf-8") as f:
    json.dump(metricas_consolidado, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 16. Resumo final
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print("RESUMO — METRICAS CONJUNTO TESTE")
print("="*70)
print(f"{'Modelo':<14}  {'R²':>8}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>8}  NBR Grau II")
print("-"*70)
for nome in modelos_dict:
    m = metricas_consolidado[nome]["test"]
    g2 = "OK" if metricas_consolidado[nome]["nbr14653_grau2_teste"]["atende"] else "NAO"
    print(f"{nome:<14}  {m['r2']:>8.4f}  {m['rmse']:>8.4f}  {m['mae']:>8.4f}  "
          f"{m['mape']:>7.2f}%  {g2}")

print(f"\nGraficos salvos em: {OUT_DIR.resolve()}")
print("  01_pred_vs_obs_teste.png")
print("  02_residuos_vs_previsto.png")
print("  03_distribuicao_residuos.png")
print("  04_qq_plot_residuos.png")
print("  05_residuos_por_padrao.png")
print("  06_perfis_marginais.png")
print("  07_sensibilidade_sr_destilado.png")
print("  08_sensibilidade_sr_isolado.png")
print("  09_comparacao_metricas.png")
print("  10_residuos_temporais.png")
print("  metricas_comparativas.json")
print("  tabela_metricas.csv")
print("\n05_analise_comparativa.py CONCLUIDO.")
