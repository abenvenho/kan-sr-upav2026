# SPDX-License-Identifier: Apache-2.0
"""
Exporta os pesos da KAN treinada (models/kan_model_state.pt, formato PyTorch)
para models/kan_weights.npz e o normalizador MinMax (data/preprocessed/scaler_X.pkl)
para models/scaler_minmax.json - ambos lidos pelo aplicativo sem PyTorch nem pickle.

Uso (na raiz do repositorio):  python tools/exportar_pesos_kan.py
Requer: torch, numpy, scikit-learn
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sd = torch.load(ROOT / "models" / "kan_model_state.pt", weights_only=True, map_location="cpu")
cfg = json.loads((ROOT / "results" / "kan" / "config_kan.json").read_text(encoding="utf-8"))

# o ramo simbolico nao foi usado no modelo final; confere antes de omiti-lo
for key, val in sd.items():
    if key.startswith("symbolic_fun") and key.endswith("mask"):
        assert float(val.abs().sum()) == 0.0, f"{key} nao e nulo"
    if key.startswith(("node_bias", "subnode_bias")):
        assert float(val.abs().sum()) == 0.0, f"{key} nao e nulo"
    if key.startswith(("node_scale", "subnode_scale")):
        assert torch.allclose(val, torch.ones_like(val)), f"{key} nao e unitario"

out = {"k": np.int64(cfg["k"]), "n_layers": np.int64(len(cfg["width"]) - 1),
       "y_min": np.float64(cfg["y_norm"]["y_min"]), "y_range": np.float64(cfg["y_norm"]["y_range"])}
for l in range(int(out["n_layers"])):
    for name in ("grid", "coef", "mask", "scale_base", "scale_sp"):
        out[f"{name}_{l}"] = sd[f"act_fun.{l}.{name}"].numpy()

np.savez_compressed(ROOT / "models" / "kan_weights.npz", **out)
print("models/kan_weights.npz gravado")

import pickle  # noqa: E402

with open(ROOT / "data" / "preprocessed" / "scaler_X.pkl", "rb") as f:
    scaler = pickle.load(f)
features = json.loads((ROOT / "data" / "preprocessed" / "feature_names.json").read_text(encoding="utf-8"))
(ROOT / "models" / "scaler_minmax.json").write_text(json.dumps({
    "features": features,
    "data_min": scaler.data_min_.tolist(),
    "data_range": scaler.data_range_.tolist(),
    "data_max": scaler.data_max_.tolist(),
}, indent=2), encoding="utf-8")
print("models/scaler_minmax.json gravado")
