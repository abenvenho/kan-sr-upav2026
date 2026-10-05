"""
08_graficos_comparativos.py
============================
Gráficos comparativos das 4 abordagens finais:
  1. KAN
  2. KAN+SR Multivariado
  3. SR Isolado
  4. OLS com Box-Cox (stepwise AIC)

Gráficos gerados:
  A. Painel 4×1 — Valores Previstos vs Observados (conjunto de teste)
  B. Painel 4×1 — Resíduos vs Valores Previstos (conjunto de teste)
  C. Painel 4×1 — Distribuição dos resíduos com curva normal
  D. Painel 4×1 — Q-Q plot dos resíduos padronizados
  E. Painel 4×1 — Resíduos por padrão construtivo (boxplot)
  F. Barplot comparativo R² e RMSE (treino / val / teste)
  G. Tabela visual de métricas

Entrada:
  modelos/predicoes.npz            (KAN)
  resultados_sr_mv/predicoes_sr_mv.npz
  resultados_sr_isolado/predicoes_sr_isolado.npz
  resultados_ols/predicoes_ols.npz
  base_modelagem_kan.csv
  preprocessed/splits_index.json

Saída:
  outputs/comparativo_final/
    A_pred_vs_obs.png
    B_residuos_vs_previsto.png
    C_distribuicao_residuos.png
    D_qq_plot.png
    E_residuos_por_padrao.png
    F_comparacao_metricas.png

Executar com:
  set KMP_DUPLICATE_LIB_OK=TRUE
  python 08_graficos_comparativos.py
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

OUT_DIR = Path("outputs/comparativo_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI   = 150
ALPHA = 0.20
MSIZE = 4

MODELOS = {
    "KAN":         {"cor": "#2563EB", "marker": "o"},
    "KAN+SR MV":   {"cor": "#16A34A", "marker": "s"},
    "SR Isolado":  {"cor": "#DC2626", "marker": "^"},
    "OLS Box-Cox": {"cor": "#9333EA", "marker": "D"},
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 9,
    "figure.dpi": DPI,
})

# ---------------------------------------------------------------------------
# 1. Carrega predições
# ---------------------------------------------------------------------------
print("Carregando predições ...")

def load_split(npz_path, split):
    with np.load(npz_path) as d:
        return d[f"y_{split}_true"].copy(), d[f"y_{split}_pred"].copy()

PRED_FILES = {
    "KAN":         "modelos/predicoes.npz",
    "KAN+SR MV":   "resultados_sr_mv/predicoes_sr_mv.npz",
    "SR Isolado":  "resultados_sr_isolado/predicoes_sr_isolado.npz",
    "OLS Box-Cox": "resultados_ols/predicoes_ols.npz",
}

dados = {}
for nome, fpath in PRED_FILES.items():
    p = Path(fpath)
    if not p.exists():
        print(f"  AVISO: {fpath} não encontrado — pulando {nome}")
        continue
    dados[nome] = {}
    for split in ["train", "val", "test"]:
        try:
            yt, yp = load_split(p, split)
            dados[nome][split] = {"true": yt, "pred": yp}
        except Exception as e:
            print(f"  AVISO: {nome}/{split}: {e}")

modelos_disp = list(dados.keys())
print(f"  Modelos disponíveis: {modelos_disp}")

# Padrão construtivo no teste
with open("preprocessed/splits_index.json") as f:
    idx_splits = json.load(f)
idx_test = np.array(idx_splits["idx_test"])

try:
    df_base = pd.read_csv("base_modelagem_kan.csv", sep=None, engine="python",
                          usecols=["padrao_ordinal"])
    padrao_test = df_base.iloc[idx_test]["padrao_ordinal"].values
except Exception:
    padrao_test = np.ones(len(idx_test), dtype=int)
    print("  AVISO: padrao_ordinal não disponível")

padroes = sorted(np.unique(padrao_test))

# ---------------------------------------------------------------------------
# 2. Função de métricas
# ---------------------------------------------------------------------------
def metricas(yt, yp):
    res  = yp - yt
    ss_r = np.sum(res**2)
    ss_t = np.sum((yt - yt.mean())**2)
    return {
        "r2":   float(1 - ss_r / max(ss_t, 1e-12)),
        "rmse": float(np.sqrt(np.mean(res**2))),
        "mae":  float(np.abs(res).mean()),
        "mape": float(np.mean(np.abs(np.exp(yp) - np.exp(yt)) / np.exp(yt)) * 100),
    }

# ---------------------------------------------------------------------------
# 3. A — Pred vs Obs (4 painéis, conjunto de teste)
# ---------------------------------------------------------------------------
print("Gerando Figura A — Pred vs Obs ...")

n = len(modelos_disp)
fig, axes = plt.subplots(1, n, figsize=(4.5*n, 5), sharey=True)
if n == 1: axes = [axes]
fig.suptitle("Valores Previstos vs. Observados — Conjunto de Teste\nln(Valor/m²)",
             fontsize=13, fontweight="bold")

for ax, nome in zip(axes, modelos_disp):
    cfg = MODELOS.get(nome, {"cor":"gray","marker":"o"})
    yt  = dados[nome]["test"]["true"]
    yp  = dados[nome]["test"]["pred"]
    m   = metricas(yt, yp)
    lim = [min(yt.min(), yp.min()) - 0.15, max(yt.max(), yp.max()) + 0.15]

    ax.scatter(yt, yp, alpha=ALPHA, s=MSIZE, color=cfg["cor"],
               marker=cfg["marker"], rasterized=True)
    ax.plot(lim, lim, "k--", lw=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observado ln(V/m²)")
    ax.set_ylabel("Previsto" if ax == axes[0] else "")
    ax.set_title(f"{nome}\nR²={m['r2']:.4f}  RMSE={m['rmse']:.4f}")
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(OUT_DIR / "A_pred_vs_obs.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 4. B — Resíduos vs Previsto
# ---------------------------------------------------------------------------
print("Gerando Figura B — Resíduos vs Previsto ...")

fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5))
if n == 1: axes = [axes]
fig.suptitle("Resíduos vs. Valores Previstos — Conjunto de Teste", fontsize=13, fontweight="bold")

for ax, nome in zip(axes, modelos_disp):
    cfg = MODELOS.get(nome, {"cor":"gray","marker":"o"})
    yt  = dados[nome]["test"]["true"]
    yp  = dados[nome]["test"]["pred"]
    res = yt - yp

    ax.scatter(yp, res, alpha=ALPHA, s=MSIZE, color=cfg["cor"],
               marker=cfg["marker"], rasterized=True)
    ax.axhline(0, color="k", lw=1.2, ls="--")

    # Média móvel suave
    order = np.argsort(yp)
    w = max(1, len(yp) // 20)
    tx = np.convolve(yp[order], np.ones(w)/w, "valid")
    ry = np.convolve(res[order], np.ones(w)/w, "valid")
    ax.plot(tx, ry, color="black", lw=1.5, label="tendência")

    ax.set_xlabel("Previsto ln(V/m²)")
    ax.set_ylabel("Resíduo (obs−prev)" if ax == axes[0] else "")
    ax.set_title(f"{nome}\nstd={res.std():.4f}")

plt.tight_layout()
plt.savefig(OUT_DIR / "B_residuos_vs_previsto.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 5. C — Distribuição dos resíduos
# ---------------------------------------------------------------------------
print("Gerando Figura C — Distribuição dos resíduos ...")

fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5))
if n == 1: axes = [axes]
fig.suptitle("Distribuição dos Resíduos — Conjunto de Teste", fontsize=13, fontweight="bold")

for ax, nome in zip(axes, modelos_disp):
    cfg = MODELOS.get(nome, {"cor":"gray"})
    yt  = dados[nome]["test"]["true"]
    yp  = dados[nome]["test"]["pred"]
    res = yt - yp

    ax.hist(res, bins=50, density=True, color=cfg["cor"],
            alpha=0.6, edgecolor="white", lw=0.3)
    xg = np.linspace(res.min(), res.max(), 200)
    ax.plot(xg, stats.norm.pdf(xg, res.mean(), res.std()),
            "k-", lw=1.5, label="normal")
    _, sw_p = stats.shapiro(res[:5000])
    ax.set_xlabel("Resíduo")
    ax.set_ylabel("Densidade" if ax == axes[0] else "")
    ax.set_title(f"{nome}\nskew={stats.skew(res):.3f}  SW p={sw_p:.4f}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "C_distribuicao_residuos.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 6. D — Q-Q Plot
# ---------------------------------------------------------------------------
print("Gerando Figura D — Q-Q plot ...")

fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5))
if n == 1: axes = [axes]
fig.suptitle("Q-Q Plot dos Resíduos Padronizados — Conjunto de Teste",
             fontsize=13, fontweight="bold")

for ax, nome in zip(axes, modelos_disp):
    cfg = MODELOS.get(nome, {"cor":"gray"})
    yt  = dados[nome]["test"]["true"]
    yp  = dados[nome]["test"]["pred"]
    res = (yt - yp)
    res_std = (res - res.mean()) / max(res.std(), 1e-12)

    qq = stats.probplot(res_std, dist="norm")
    ax.scatter(qq[0][0], qq[0][1], alpha=0.3, s=4, color=cfg["cor"], rasterized=True)
    lim = max(abs(qq[0][0]).max(), abs(qq[0][1]).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2)
    ax.set_xlabel("Quantil teórico (Normal)")
    ax.set_ylabel("Quantil amostral" if ax == axes[0] else "")
    ax.set_title(f"{nome}")

plt.tight_layout()
plt.savefig(OUT_DIR / "D_qq_plot.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 7. E — Resíduos por padrão construtivo
# ---------------------------------------------------------------------------
print("Gerando Figura E — Resíduos por padrão ...")

fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5))
if n == 1: axes = [axes]
fig.suptitle("Resíduos por Padrão Construtivo — Conjunto de Teste\n(1=popular … 6=fino)",
             fontsize=13, fontweight="bold")

for ax, nome in zip(axes, modelos_disp):
    cfg = MODELOS.get(nome, {"cor":"gray"})
    yt  = dados[nome]["test"]["true"]
    yp  = dados[nome]["test"]["pred"]
    res = yt - yp

    data_bp = [res[padrao_test == p] for p in padroes]
    bp = ax.boxplot(data_bp, labels=[str(p) for p in padroes],
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(cfg["cor"])
        patch.set_alpha(0.5)
    ax.axhline(0, color="k", lw=1.2, ls="--")
    ax.set_xlabel("Padrão construtivo")
    ax.set_ylabel("Resíduo ln(V/m²)" if ax == axes[0] else "")
    ax.set_title(f"{nome}")

plt.tight_layout()
plt.savefig(OUT_DIR / "E_residuos_por_padrao.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 8. F — Comparação de métricas (R² e RMSE, todos os splits)
# ---------------------------------------------------------------------------
print("Gerando Figura F — Comparação de métricas ...")

splits = ["train", "val", "test"]
split_labels = ["Treino", "Val.", "Teste"]
x = np.arange(len(splits))
width = 0.2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comparação de Desempenho — 4 Abordagens", fontsize=13, fontweight="bold")

for ax_idx, (metric, ylabel) in enumerate([("r2","R²"),("rmse","RMSE ln(V/m²)")]):
    ax = axes[ax_idx]
    for i_m, nome in enumerate(modelos_disp):
        cfg = MODELOS.get(nome, {"cor":"gray"})
        vals = [metricas(dados[nome][s]["true"], dados[nome][s]["pred"])[metric]
                for s in splits]
        offset = (i_m - (len(modelos_disp)-1)/2) * width
        bars = ax.bar(x + offset, vals, width, label=nome,
                      color=cfg["cor"], alpha=0.78)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(fontsize=8)
    if metric == "r2":
        ax.axhline(0.64, color="red", lw=1.2, ls="--", alpha=0.7,
                   label="Grau II (0,64)")
        ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig(OUT_DIR / "F_comparacao_metricas.png", dpi=DPI, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 9. Tabela de métricas — salva CSV
# ---------------------------------------------------------------------------
rows = []
for nome in modelos_disp:
    for split in splits:
        m = metricas(dados[nome][split]["true"], dados[nome][split]["pred"])
        rows.append({"modelo": nome, "split": split, **m})

df_tab = pd.DataFrame(rows)
df_tab.to_csv(OUT_DIR / "metricas_comparativas_4modelos.csv", index=False)

print("\n" + "="*60)
print("RESUMO — CONJUNTO DE TESTE")
print("="*60)
print(f"{'Modelo':<14}  {'R²':>8}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>8}  Grau II")
print("-"*60)
for nome in modelos_disp:
    m = metricas(dados[nome]["test"]["true"], dados[nome]["test"]["pred"])
    g2 = "OK" if m["r2"] >= 0.64 and m["mape"] <= 30 else "NAO"
    print(f"{nome:<14}  {m['r2']:>8.4f}  {m['rmse']:>8.4f}  {m['mae']:>8.4f}  {m['mape']:>7.2f}%  {g2}")

print(f"\nGráficos em: {OUT_DIR.resolve()}")
print("  A_pred_vs_obs.png")
print("  B_residuos_vs_previsto.png")
print("  C_distribuicao_residuos.png")
print("  D_qq_plot.png")
print("  E_residuos_por_padrao.png")
print("  F_comparacao_metricas.png")
print("  metricas_comparativas_4modelos.csv")
print("\n08_graficos_comparativos.py CONCLUÍDO.")
