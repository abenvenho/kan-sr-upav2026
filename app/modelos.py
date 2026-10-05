# SPDX-License-Identifier: Apache-2.0
"""
Carga da base e previsao dos tres modelos executaveis do estudo:

  * KAN           - rede Kolmogorov-Arnold treinada (forward em NumPy, kan_numpy.py)
  * KAN+SR (c19)  - equacao destilada por regressao simbolica (results/sr_destilada)
  * OLS Box-Cox   - regressao linear de referencia (results/ols)

Todas as funcoes recebem um DataFrame com as colunas originais da base
(unidades reais) e devolvem ln(R$/m2). Os coeficientes e parametros sao lidos
dos arquivos de resultados do repositorio, nao copiados no codigo.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from kan_numpy import KANNumpy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
MOD = ROOT / "models"

TARGET = "ln_valor_m2"
FEATURES: list[str] = json.loads((DATA / "preprocessed" / "feature_names.json").read_text(encoding="utf-8"))
PADROES = ["A", "B", "C", "D", "E", "F"]          # padrao_ordinal 1..6
CEP_MIN, CEP_MAX = 1004010, 8490800               # normalizacao de cep_norm (base completa)
MODELOS = ["KAN", "KAN+SR", "OLS Box-Cox"]


# --------------------------------------------------------------------------- dados
@lru_cache(maxsize=1)
def base() -> pd.DataFrame:
    df = pd.read_csv(DATA / "base_modelagem_kan.csv")
    df["data_transacao"] = pd.to_datetime(df["data_transacao"])
    return df


@lru_cache(maxsize=1)
def scaler() -> dict:
    s = json.loads((MOD / "scaler_minmax.json").read_text(encoding="utf-8"))
    assert s["features"] == FEATURES
    return {"min": np.array(s["data_min"]), "range": np.array(s["data_range"]), "max": np.array(s["data_max"])}


def normalizar(df: pd.DataFrame) -> np.ndarray:
    """MinMax [0,1] com o mesmo ajuste (amostra de treino) usado no artigo."""
    s = scaler()
    return (df[FEATURES].to_numpy(dtype=float) - s["min"]) / s["range"]


def cep_para_norm(cep: int) -> float:
    return (cep - CEP_MIN) / (CEP_MAX - CEP_MIN)


def data_para_t(ano: int, mes: int) -> int:
    """Indice temporal t_ordinal: meses contados a partir de jan/2021 (jan/2021 = 0)."""
    return (ano - 2021) * 12 + mes - 1


# --------------------------------------------------------------------------- KAN
@lru_cache(maxsize=1)
def _kan() -> KANNumpy:
    return KANNumpy(MOD / "kan_weights.npz")


def prever_kan(df: pd.DataFrame) -> np.ndarray:
    return _kan().predict_ln_valor_m2(normalizar(df))


# --------------------------------------------------------------------------- KAN+SR c19
def prever_sr(df: pd.DataFrame) -> np.ndarray:
    """Equacao c19 (results/sr_destilada/formula_c19.txt), variaveis em MinMax [0,1]."""
    n = dict(zip(FEATURES, normalizar(df).T))
    return (
        (n["fator_obsolescencia"] - n["flag_subdeclaracao"]) ** 3 * n["padrao_ordinal"]
        + (n["valor_m2_construcao"] + n["area_terreno_m2"]) * 0.45851025
        + np.sqrt(np.clip(n["valor_m2_terreno"] * 1.969852, 0.0, None))
        + 7.6152616
    )


# --------------------------------------------------------------------------- OLS Box-Cox
@lru_cache(maxsize=1)
def _ols() -> tuple[pd.Series, dict]:
    coef = pd.read_csv(RES / "ols" / "coeficientes.csv").set_index("Feature")["Coeficiente"]
    lambdas = json.loads((RES / "ols" / "boxcox_lambdas.json").read_text(encoding="utf-8"))
    return coef, lambdas


def _boxcox(x, lam: float, shift: float = 0.0) -> np.ndarray:
    # mesma regra de 07_regressao_linear.py: |lambda| < 0,05 -> log
    x = np.clip(np.asarray(x, dtype=float) + shift, 1e-10, None)
    return np.log(x) if abs(lam) < 0.05 else (x**lam - 1.0) / lam


def prever_ols(df: pd.DataFrame) -> np.ndarray:
    coef, lambdas = _ols()
    y = np.full(len(df), float(coef["const"]))
    for nome, c in coef.items():
        if nome == "const":
            continue
        if nome.endswith("_bc"):
            v = nome[:-3]
            y += c * _boxcox(df[v], lambdas[v]["lambda"], lambdas[v]["shift"])
        else:
            y += c * df[nome].to_numpy(dtype=float)
    return y


PREVISORES = {"KAN": prever_kan, "KAN+SR": prever_sr, "OLS Box-Cox": prever_ols}


def prever(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({m: f(df) for m, f in PREVISORES.items()}, index=df.index)


# --------------------------------------------------------------------------- metricas
def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Mesmas definicoes de 05_analise_comparativa.py (MAPE em R$/m2)."""
    res = y_pred - y_true
    return {
        "R²": 1 - np.sum(res**2) / np.sum((y_true - y_true.mean()) ** 2),
        "RMSE (ln)": float(np.sqrt(np.mean(res**2))),
        "MAE (ln)": float(np.mean(np.abs(res))),
        "MAPE (%)": float(np.mean(np.abs(np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true)) * 100),
        "n": len(y_true),
    }


def metricas_publicadas() -> pd.DataFrame:
    return pd.read_csv(RES / "comparativo" / "metricas_comparativas_6modelos.csv")


def faixa_treino() -> pd.DataFrame:
    """Minimo e maximo de cada variavel na amostra de treino (dominio do modelo)."""
    tr = base().query("split == 'train'")[FEATURES]
    return pd.DataFrame({"min": tr.min(), "max": tr.max()})
