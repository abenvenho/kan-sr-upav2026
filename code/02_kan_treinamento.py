"""
02_kan_treinamento.py
=====================
Treinamento da Kolmogorov-Arnold Network (KAN)
Avaliacao de Apartamentos - Sao Paulo-SP
Referencia: ABNT NBR 14.653-2:2011 (Grau II)

Entrada : preprocessed/
Saidas  : modelos/
              kan_model_state.pt
              kan_checkpoint/
              metricas_kan.json
              historico_treino.json
              ativacoes_pysr/  (pares x,y por aresta para PySR)
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Configuracoes
# ---------------------------------------------------------------------------

PREPROCESSED_DIR = Path("preprocessed")
OUT_DIR          = Path("modelos")
ATIV_DIR         = OUT_DIR / "ativacoes_pysr"

OUT_DIR.mkdir(parents=True, exist_ok=True)
ATIV_DIR.mkdir(parents=True, exist_ok=True)

KAN_WIDTH    = [12, 6, 1]  # arquitetura (12 features apos expansao)
KAN_GRID     = 5            # grade inicial
KAN_GRID_EXT = 10           # grade apos extensao
KAN_K        = 3            # spline cubico
KAN_SEED     = 42

# Fases de treinamento
# Fase 1: ajuste puro (sem regularizacao) para convergencia inicial
STEPS_F1 = 500;  LR_F1 = 1e-3;  LAMB_F1 = 0.0;   LAMB_ENT_F1 = 0.0
# Fase 2: refinamento apos extensao de grade
STEPS_F2 = 300;  LR_F2 = 3e-4;  LAMB_F2 = 0.0;   LAMB_ENT_F2 = 0.0
# Fase 3: esparsidade suave (prepara para PySR)
STEPS_F3 = 150;  LR_F3 = 1e-4;  LAMB_F3 = 1e-5;  LAMB_ENT_F3 = 1.0

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Dispositivo: {device}")

# ---------------------------------------------------------------------------
# 2. Carga dos tensores
# ---------------------------------------------------------------------------

print("\nCarregando tensores pre-processados...")

def safe_load(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)

X_train = safe_load(PREPROCESSED_DIR / "X_train.pt").to(device)
X_val   = safe_load(PREPROCESSED_DIR / "X_val.pt").to(device)
X_test  = safe_load(PREPROCESSED_DIR / "X_test.pt").to(device)
y_train = safe_load(PREPROCESSED_DIR / "y_train.pt").to(device)
y_val   = safe_load(PREPROCESSED_DIR / "y_val.pt").to(device)
y_test  = safe_load(PREPROCESSED_DIR / "y_test.pt").to(device)

with open(PREPROCESSED_DIR / "feature_names.json", encoding="utf-8") as f:
    FEATURES = json.load(f)

print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")

# ---------------------------------------------------------------------------
# 3. Normalizacao do alvo para [0, 1]
#    KAN trabalha com splines em intervalo unitario; target nao-normalizado
#    faz a regularizacao dominar a loss de dados.
# ---------------------------------------------------------------------------

y_min = y_train.min().item()
y_max = y_train.max().item()
y_range = y_max - y_min

def norm_y(y):
    return (y - y_min) / y_range

def denorm_y(y_n):
    return y_n * y_range + y_min

y_train_n = norm_y(y_train)
y_val_n   = norm_y(y_val)
y_test_n  = norm_y(y_test)

print(f"  y_train  : [{y_train.min():.3f}, {y_train.max():.3f}]  (ln_valor_m2)")
print(f"  y_norm   : [{y_train_n.min():.3f}, {y_train_n.max():.3f}]  (normalizado)")

# Dataset pykan
dataset = {
    "train_input": X_train,
    "train_label": y_train_n.unsqueeze(1),
    "test_input":  X_val,
    "test_label":  y_val_n.unsqueeze(1),
}

# ---------------------------------------------------------------------------
# 4. Metricas (em escala original de ln_valor_m2)
# ---------------------------------------------------------------------------

def calc_rmse(y_pred_n, y_true):
    yp = denorm_y(y_pred_n.squeeze())
    return torch.sqrt(torch.mean((yp - y_true.squeeze()) ** 2)).item()

def calc_mae(y_pred_n, y_true):
    yp = denorm_y(y_pred_n.squeeze())
    return torch.mean(torch.abs(yp - y_true.squeeze())).item()

def calc_r2(y_pred_n, y_true):
    yp = denorm_y(y_pred_n.squeeze())
    yt = y_true.squeeze()
    ss_res = torch.sum((yt - yp) ** 2)
    ss_tot = torch.sum((yt - yt.mean()) ** 2)
    return (1.0 - ss_res / ss_tot).item()

# ---------------------------------------------------------------------------
# 5. Importacao do KAN
# ---------------------------------------------------------------------------

def instalar_pykan():
    import subprocess
    print("  Instalando pykan ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pykan", "--quiet"],
        stdout=subprocess.DEVNULL,
    )

KAN = None
for tentativa in range(2):
    try:
        from kan import KAN as _KAN
        KAN = _KAN
        print("Biblioteca KAN: pykan")
        break
    except ImportError:
        if tentativa == 0:
            instalar_pykan()

if KAN is None:
    try:
        from efficient_kan import KAN as _KAN
        KAN = _KAN
        print("Biblioteca KAN: efficient-kan (fallback)")
    except ImportError:
        raise RuntimeError("Instale pykan:  pip install pykan")

# ---------------------------------------------------------------------------
# 6. Construcao do modelo
# ---------------------------------------------------------------------------

print(f"\nArquitetura KAN : {KAN_WIDTH}  grid={KAN_GRID}  k={KAN_K}")

model = KAN(
    width=KAN_WIDTH,
    grid=KAN_GRID,
    k=KAN_K,
    seed=KAN_SEED,
    device=device,
)

# ---------------------------------------------------------------------------
# 7. FASE 1 — Ajuste puro (sem regularizacao)
# ---------------------------------------------------------------------------

print(f"\n{'='*55}")
print(f"FASE 1 - Ajuste puro ({STEPS_F1} steps, sem regularizacao)")
print("="*55)
t0 = time.time()
results1 = model.fit(
    dataset, opt="Adam", steps=STEPS_F1, lr=LR_F1,
    lamb=LAMB_F1, lamb_entropy=LAMB_ENT_F1,
)
t1 = time.time() - t0
with torch.no_grad():
    vp = model(X_val)
print(f"  Concluido em {t1:.1f}s  |  Val RMSE={calc_rmse(vp, y_val):.4f}  R2={calc_r2(vp, y_val):.4f}")

# ---------------------------------------------------------------------------
# 8. Extensao de grade
# ---------------------------------------------------------------------------

print(f"\nExtensao de grade: {KAN_GRID} -> {KAN_GRID_EXT}")
try:
    model = model.refine(KAN_GRID_EXT)
    print("  Grade estendida.")
except AttributeError:
    try:
        model.grid_extension(KAN_GRID_EXT, dataset["train_input"])
        print("  Grade estendida (grid_extension).")
    except Exception as e:
        print(f"  Extensao ignorada: {e}")

# ---------------------------------------------------------------------------
# 9. FASE 2 — Refinamento pos-extensao (sem regularizacao)
# ---------------------------------------------------------------------------

print(f"\n{'='*55}")
print(f"FASE 2 - Refinamento ({STEPS_F2} steps, sem regularizacao)")
print("="*55)
t0 = time.time()
results2 = model.fit(
    dataset, opt="Adam", steps=STEPS_F2, lr=LR_F2,
    lamb=LAMB_F2, lamb_entropy=LAMB_ENT_F2,
)
t2 = time.time() - t0
with torch.no_grad():
    vp = model(X_val)
print(f"  Concluido em {t2:.1f}s  |  Val RMSE={calc_rmse(vp, y_val):.4f}  R2={calc_r2(vp, y_val):.4f}")

# ---------------------------------------------------------------------------
# 10. FASE 3 — Esparsidade suave (prepara para PySR)
# ---------------------------------------------------------------------------

print(f"\n{'='*55}")
print(f"FASE 3 - Esparsidade ({STEPS_F3} steps, lamb={LAMB_F3})")
print("="*55)
t0 = time.time()
results3 = model.fit(
    dataset, opt="Adam", steps=STEPS_F3, lr=LR_F3,
    lamb=LAMB_F3, lamb_entropy=LAMB_ENT_F3,
)
t3 = time.time() - t0
t_total = t1 + t2 + t3

# ---------------------------------------------------------------------------
# 11. Avaliacao final
# ---------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    yp_train = model(X_train)
    yp_val   = model(X_val)
    yp_test  = model(X_test)

# Salvar predicoes para o relatorio (escala original ln_valor_m2)
np.savez(
    OUT_DIR / "predicoes.npz",
    y_train_true = y_train.cpu().numpy(),
    y_train_pred = denorm_y(yp_train).squeeze().cpu().detach().numpy(),
    y_val_true   = y_val.cpu().numpy(),
    y_val_pred   = denorm_y(yp_val).squeeze().cpu().detach().numpy(),
    y_test_true  = y_test.cpu().numpy(),
    y_test_pred  = denorm_y(yp_test).squeeze().cpu().detach().numpy(),
)
print("Predicoes salvas em modelos/predicoes.npz")

metricas = {}
print(f"\n{'='*55}")
print("METRICAS FINAIS (escala original ln_valor_m2)")
print("="*55)
for nome, (yp, yt) in [("treino", (yp_train, y_train)),
                        ("validacao", (yp_val, y_val)),
                        ("teste", (yp_test, y_test))]:
    m = {
        "rmse": round(calc_rmse(yp, yt), 6),
        "mae":  round(calc_mae(yp, yt), 6),
        "r2":   round(calc_r2(yp, yt), 6),
    }
    metricas[nome] = m
    print(f"  {nome.upper():12} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")

# NBR 14.653-2 Grau II
r2_val   = metricas["validacao"]["r2"]
rmse_val = metricas["validacao"]["rmse"]
erro_pct = (np.exp(rmse_val) - 1.0) * 100.0
ok_r2    = bool(r2_val   >= 0.64)
ok_ep    = bool(erro_pct <= 30.0)

print(f"\n  NBR 14.653-2 Grau II (validacao):")
print(f"    R2 >= 0.64        : {r2_val:.4f}  [{'OK' if ok_r2 else 'NAO ATENDE'}]")
print(f"    Erro padrao <= 30%: {erro_pct:.1f}%  [{'OK' if ok_ep else 'NAO ATENDE'}]")
print(f"    {'=> Modelo atende ao Grau II.' if (ok_r2 and ok_ep) else '=> Ajustar (ver discussao no artigo).'}")

# ---------------------------------------------------------------------------
# 12. Poda
# ---------------------------------------------------------------------------

print("\nAplicando poda (pruning) ...")
try:
    # MultKAN >= 0.3: prune() sem argumentos, ou prune_node()/prune_edge()
    model.prune()
    print("  Poda aplicada (model.prune()).")
except Exception:
    try:
        model.prune_node(threshold=1e-2)
        print("  Poda aplicada (model.prune_node()).")
    except Exception as e2:
        print(f"  Poda ignorada: {e2}")

# ---------------------------------------------------------------------------
# 13. Extracao de perfis marginais para PySR
#     Abordagem robusta a qualquer versao do pykan:
#     Para cada feature i, varia x_i em [0,1] mantendo os demais na media.
#     O script 03 (PySR) ajustara expressoes simbolicas a esses perfis.
# ---------------------------------------------------------------------------

print("\nExtraindo perfis marginais por feature ...")
N_POINTS = 300
n_perfis = 0
model.eval()
x_mean = X_train.mean(dim=0, keepdim=True)  # [1, n_features]

for i, feat_name in enumerate(FEATURES):
    try:
        x_probe = x_mean.repeat(N_POINTS, 1).clone()
        x_probe[:, i] = torch.linspace(0.0, 1.0, N_POINTS, device=device)
        with torch.no_grad():
            y_probe = model(x_probe).squeeze()
        # Desnormaliza saida para escala ln_valor_m2
        y_probe_orig = denorm_y(y_probe)
        np.savez(
            ATIV_DIR / f"perfil_{i:02d}_{feat_name}.npz",
            x_norm=x_probe[:, i].cpu().numpy(),
            y_ln=y_probe_orig.cpu().numpy(),
            feat_name=feat_name,
        )
        n_perfis += 1
    except Exception as ep:
        print(f"  Erro em {feat_name}: {ep}")

print(f"  {n_perfis} perfis salvos em {ATIV_DIR}/")

# ---------------------------------------------------------------------------
# 14. Persistencia
# ---------------------------------------------------------------------------

# State dict (para carregar o modelo depois)
torch.save(model.state_dict(), OUT_DIR / "kan_model_state.pt")

print(f"State dict salvo em {OUT_DIR}/kan_model_state.pt")

# Normalizacao do alvo (necessaria para o script 03)
norm_target = {"y_min": y_min, "y_max": y_max, "y_range": y_range}
with open(OUT_DIR / "norm_target.json", "w") as f:
    json.dump(norm_target, f, indent=2)

metricas["nbr14653_grau2"] = {
    "r2_validacao":       r2_val,
    "rmse_log_validacao": rmse_val,
    "erro_pct_mercado":   round(erro_pct, 2),
    "ok_r2":              ok_r2,
    "ok_erro_padrao":     ok_ep,
    "atende_grau2":       ok_r2 and ok_ep,
}
with open(OUT_DIR / "metricas_kan.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

def extrair_loss(res):
    return ([float(v) for v in res.get("train_loss", [])],
            [float(v) for v in res.get("test_loss", [])])

tr1, vl1 = extrair_loss(results1)
tr2, vl2 = extrair_loss(results2)
tr3, vl3 = extrair_loss(results3)
historico = {
    "fase1": {"train": tr1, "val": vl1, "tempo_s": round(t1,1)},
    "fase2": {"train": tr2, "val": vl2, "tempo_s": round(t2,1)},
    "fase3": {"train": tr3, "val": vl3, "tempo_s": round(t3,1)},
}
with open(OUT_DIR / "historico_treino.json", "w") as f:
    json.dump(historico, f, indent=2)

config_kan = {
    "width": KAN_WIDTH, "grid": KAN_GRID_EXT, "k": KAN_K,
    "seed": KAN_SEED, "features": FEATURES, "target": "ln_valor_m2",
    "y_norm": norm_target,
}
with open(OUT_DIR / "config_kan.json", "w", encoding="utf-8") as f:
    json.dump(config_kan, f, indent=2)

print(f"\nArtefatos em '{OUT_DIR}/':")
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {str(p.relative_to(OUT_DIR)):<40} {p.stat().st_size/1024:>7.1f} KB")

print(f"\nOK - Treinamento KAN concluido em {t_total:.0f}s.")
print(f"     R2_val={r2_val:.4f}  RMSE_val={rmse_val:.4f}  Erro={erro_pct:.1f}%")
with open(OUT_DIR / "config_kan.json", "w", encoding="utf-8") as f:
    json.dump(config_kan, f, indent=2)

print(f"\nArtefatos em '{OUT_DIR}/':")
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {str(p.relative_to(OUT_DIR)):<40} {p.stat().st_size/1024:>7.1f} KB")

print(f"\nOK - Treinamento KAN concluido em {t_total:.0f}s.")
print(f"     R2_val={r2_val:.4f}  RMSE_val={rmse_val:.4f}  Erro={erro_pct:.1f}%")
