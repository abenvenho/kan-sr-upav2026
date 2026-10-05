"""
10_mlp_treinamento.py
=====================
Redes MLP (Multi-Layer Perceptron) como referência comparativa.
Parcimônia: arquiteturas simples, comparáveis ao KAN [12→6→1].

Arquiteturas testadas:
  Grupo 1 — 1 camada oculta:
    MLP_1L_A: [12 → 16 → 1]   (225 parâmetros)
    MLP_1L_B: [12 → 24 → 1]   (361 parâmetros)

  Grupo 2 — 2 camadas ocultas:
    MLP_2L_A: [12 → 16 → 8 → 1]   (297 parâmetros)
    MLP_2L_B: [12 → 24 → 12 → 1]  (481 parâmetros)

Seleção: melhor de cada grupo por R² de validação.
Reporte final: melhor MLP_1L e melhor MLP_2L no conjunto de teste.

Treinamento:
  - Ativação: ReLU
  - Otimizador: Adam, lr=1e-3, weight_decay=1e-4
  - Scheduler: CosineAnnealingLR (T_max=200)
  - Early stopping: patience=60 (monitor val_loss)
  - Batch size: 512
  - Max epochs: 800
  - Semente: 42

Entrada:  preprocessed/  (X_*.pt, y_*.pt já em escala ln)
Saída:    resultados_mlp/
            predicoes_mlp_1L.npz
            predicoes_mlp_2L.npz
            metricas_mlp.json
            historico_treino_*.json
            modelos_mlp/  (pesos .pt)

Rodar com:
  set KMP_DUPLICATE_LIB_OK=TRUE
  python 10_mlp_treinamento.py
"""

import json
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PREP_DIR  = Path("preprocessed")
OUT_DIR   = Path("resultados_mlp")
MOD_DIR   = OUT_DIR / "modelos_mlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MOD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Device
# ---------------------------------------------------------------------------
device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if hasattr(torch.backends, "mps")
               and torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"Dispositivo: {device}")

# ---------------------------------------------------------------------------
# 2. Carrega tensores
# ---------------------------------------------------------------------------
print("\nCarregando tensores ...")

def lt(p):
    try:
        return torch.load(p, weights_only=True)
    except TypeError:
        return torch.load(p)

X_train = lt(PREP_DIR / "X_train.pt").float()
X_val   = lt(PREP_DIR / "X_val.pt").float()
X_test  = lt(PREP_DIR / "X_test.pt").float()
y_train = lt(PREP_DIR / "y_train.pt").float().view(-1, 1)
y_val   = lt(PREP_DIR / "y_val.pt").float().view(-1, 1)
y_test  = lt(PREP_DIR / "y_test.pt").float().view(-1, 1)

print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]  "
      f"mean={y_train.mean():.4f}")

N_IN = X_train.shape[1]

# ---------------------------------------------------------------------------
# 3. Arquiteturas MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP com camadas ocultas, ReLU e dropout opcional."""
    def __init__(self, layers, dropout=0.0):
        super().__init__()
        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:          # não aplica após última camada
                blocks.append(nn.ReLU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Grupos de arquiteturas
CONFIGS = {
    "MLP_1L_A": {"layers": [N_IN, 16, 1],       "dropout": 0.1, "grupo": "1L"},
    "MLP_1L_B": {"layers": [N_IN, 24, 1],        "dropout": 0.1, "grupo": "1L"},
    "MLP_2L_A": {"layers": [N_IN, 16, 8, 1],     "dropout": 0.1, "grupo": "2L"},
    "MLP_2L_B": {"layers": [N_IN, 24, 12, 1],    "dropout": 0.1, "grupo": "2L"},
}

# ---------------------------------------------------------------------------
# 4. Treinamento
# ---------------------------------------------------------------------------
BATCH_SIZE  = 512
MAX_EPOCHS  = 800
LR          = 1e-3
WEIGHT_DECAY= 1e-4
PATIENCE    = 60
T_MAX       = 200

train_ds = TensorDataset(X_train, y_train)
loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      generator=torch.Generator().manual_seed(SEED))

def treinar(nome, config):
    print(f"\n{'='*55}")
    print(f"Treinando {nome}  layers={config['layers']}  dropout={config['dropout']}")
    torch.manual_seed(SEED)
    model = MLP(config["layers"], dropout=config["dropout"]).to(device)
    print(f"  Parâmetros: {model.n_params():,}")

    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_MAX, eta_min=1e-5)
    crit  = nn.MSELoss()

    Xv = X_val.to(device); yv = y_val.to(device)
    Xt = X_train.to(device)

    best_val  = float("inf")
    best_state= None
    hist      = {"train_loss": [], "val_loss": [], "val_r2": []}
    no_improve= 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(Xb)
        ep_loss /= len(train_ds)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = crit(val_pred, yv).item()
            # R² validação
            ss_r = ((yv - val_pred)**2).sum().item()
            ss_t = ((yv - yv.mean())**2).sum().item()
            val_r2 = 1 - ss_r / max(ss_t, 1e-12)

        hist["train_loss"].append(round(ep_loss, 6))
        hist["val_loss"].append(round(val_loss, 6))
        hist["val_r2"].append(round(val_r2, 6))

        if val_loss < best_val:
            best_val   = val_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}  train={ep_loss:.5f}  val={val_loss:.5f}"
                  f"  R²_val={val_r2:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if no_improve >= PATIENCE:
            print(f"  Early stopping no epoch {epoch} (patience={PATIENCE})")
            break

    elapsed = time.time() - t0
    print(f"  Tempo: {elapsed:.1f}s  Melhor val_loss={best_val:.6f}")

    # Carrega melhores pesos
    model.load_state_dict(best_state)
    torch.save(best_state, MOD_DIR / f"{nome}_state.pt")

    with open(OUT_DIR / f"historico_{nome}.json", "w") as f:
        json.dump(hist, f)

    return model, hist, model.n_params()

# ---------------------------------------------------------------------------
# 5. Roda todos os configs
# ---------------------------------------------------------------------------
resultados = {}

for nome, cfg in CONFIGS.items():
    model, hist, n_par = treinar(nome, cfg)
    model.eval()
    with torch.no_grad():
        for split, X_s, y_s in [("train", X_train, y_train),
                                  ("val",   X_val,   y_val),
                                  ("test",  X_test,  y_test)]:
            yp = model(X_s.to(device)).cpu().numpy().ravel()
            yt = y_s.numpy().ravel()
            res = yp - yt
            ss_r = np.sum(res**2); ss_t = np.sum((yt-yt.mean())**2)
            r2   = float(1 - ss_r/max(ss_t, 1e-12))
            rmse = float(np.sqrt(np.mean(res**2)))
            mae  = float(np.abs(res).mean())
            mape = float(np.mean(np.abs(np.exp(yp)-np.exp(yt))/np.exp(yt))*100)
            if split not in resultados.get(nome, {}):
                resultados.setdefault(nome, {})[split] = {}
            resultados[nome][split] = {"r2":r2,"rmse":rmse,"mae":mae,"mape":mape}
            resultados[nome]["n_params"] = n_par
            resultados[nome]["layers"]   = cfg["layers"]
            resultados[nome]["grupo"]    = cfg["grupo"]

# ---------------------------------------------------------------------------
# 6. Seleção do melhor por grupo (R² validação)
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print("SELEÇÃO — MELHOR POR GRUPO (R² validação)")
print("="*55)

melhor = {}
for grupo in ["1L", "2L"]:
    candidatos = {n: r for n, r in resultados.items() if r["grupo"] == grupo}
    best_nome  = max(candidatos, key=lambda n: candidatos[n]["val"]["r2"])
    melhor[grupo] = best_nome
    print(f"\n  Grupo {grupo}:")
    for nome, r in candidatos.items():
        marca = " ← SELECIONADO" if nome == best_nome else ""
        print(f"    {nome}  R²_val={r['val']['r2']:.4f}  R²_te={r['test']['r2']:.4f}{marca}")

# ---------------------------------------------------------------------------
# 7. Predições finais dos modelos selecionados
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print("AVALIAÇÃO FINAL DOS MODELOS SELECIONADOS")
print("="*55)

metricas_finais = {}

for grupo, nome_sel in melhor.items():
    cfg = CONFIGS[nome_sel]
    torch.manual_seed(SEED)

    # Recarrega o modelo salvo com a mesma arquitetura do treino
    # model.eval() desativa o dropout na inferência — não precisa recriar sem ele
    model = MLP(cfg["layers"], dropout=cfg["dropout"]).to(device)
    state = torch.load(MOD_DIR / f"{nome_sel}_state.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"\n  {grupo} — {nome_sel}  {cfg['layers']}")
    splits_preds = {}
    splits_metr  = {}

    with torch.no_grad():
        for split, X_s, y_s in [("train", X_train, y_train),
                                  ("val",   X_val,   y_val),
                                  ("test",  X_test,  y_test)]:
            yp = model(X_s.to(device)).cpu().numpy().ravel()
            yt = y_s.numpy().ravel()
            res = yp - yt
            ss_r=np.sum(res**2); ss_t=np.sum((yt-yt.mean())**2)
            r2  =float(1-ss_r/max(ss_t,1e-12))
            rmse=float(np.sqrt(np.mean(res**2)))
            mae =float(np.abs(res).mean())
            mape=float(np.mean(np.abs(np.exp(yp)-np.exp(yt))/np.exp(yt))*100)
            splits_preds[split] = {"true": yt, "pred": yp}
            splits_metr[split]  = {"r2":r2,"rmse":rmse,"mae":mae,"mape":mape,"n":len(yt)}
            print(f"    {split.upper():<6}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%")

    ok_r2   = splits_metr["test"]["r2"]   >= 0.64
    ok_mape = splits_metr["test"]["mape"] <= 30.0
    print(f"    NBR Grau II (teste): {'ATENDE' if (ok_r2 and ok_mape) else 'NAO ATENDE'}")

    # Salva predições
    npz_path = OUT_DIR / f"predicoes_mlp_{grupo}.npz"
    np.savez(npz_path,
             y_train_true=splits_preds["train"]["true"],
             y_train_pred=splits_preds["train"]["pred"],
             y_val_true  =splits_preds["val"]["true"],
             y_val_pred  =splits_preds["val"]["pred"],
             y_test_true =splits_preds["test"]["true"],
             y_test_pred =splits_preds["test"]["pred"])

    metricas_finais[f"MLP_{grupo}"] = {
        "arquitetura_nome": nome_sel,
        "layers": cfg["layers"],
        "n_params": CONFIGS[nome_sel]["layers"],
        "metricas": splits_metr,
        "nbr14653_grau2_teste": {
            "ok_r2": ok_r2, "ok_mape": ok_mape, "atende": bool(ok_r2 and ok_mape)
        },
    }

# ---------------------------------------------------------------------------
# 8. Comparação com modelos anteriores
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print("RESUMO COMPARATIVO — CONJUNTO DE TESTE")
print("="*55)
print(f"{'Modelo':<20}  {'R²':>8}  {'RMSE':>8}  {'MAE':>8}  {'MAPE':>8}  Grau II")
print("-"*70)

referencias = {
    "KAN":            (0.7157, 0.2186, 0.1647, 16.7),
    "KAN+SR MV c19":  (0.6646, 0.2374, 0.1810, 18.3),
    "OLS Box-Cox":    (0.6607, 0.2388, 0.1799, 18.0),
    "SR Isolado":     (0.6077, 0.2567, 0.1934, 19.8),
}
for nome, (r2, rmse, mae, mape) in referencias.items():
    g2 = "SIM" if r2 >= 0.64 and mape <= 30 else "NAO"
    print(f"  {nome:<18}  {r2:>8.4f}  {rmse:>8.4f}  {mae:>8.4f}  {mape:>7.1f}%  {g2}")

print("-"*70)
for grupo_key, mdict in metricas_finais.items():
    m  = mdict["metricas"]["test"]
    g2 = "SIM" if mdict["nbr14653_grau2_teste"]["atende"] else "NAO"
    print(f"  {grupo_key} {mdict['arquitetura_nome']:<10}  "
          f"{m['r2']:>8.4f}  {m['rmse']:>8.4f}  {m['mae']:>8.4f}  "
          f"{m['mape']:>7.2f}%  {g2}")

# ---------------------------------------------------------------------------
# 9. Salva métricas JSON
# ---------------------------------------------------------------------------
with open(OUT_DIR / "metricas_mlp.json", "w", encoding="utf-8") as f:
    json.dump(metricas_finais, f, indent=2, ensure_ascii=False)

# Salva também resultados de todas as configs (para análise)
with open(OUT_DIR / "todos_configs.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"\nArquivos em: {OUT_DIR.resolve()}")
print("  predicoes_mlp_1L.npz  predicoes_mlp_2L.npz")
print("  metricas_mlp.json     todos_configs.json")
print("  modelos_mlp/          historico_*.json")
print("\n10_mlp_treinamento.py CONCLUÍDO.")
