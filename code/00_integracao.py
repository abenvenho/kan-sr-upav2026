"""
00_integracao.py
================
Integracao ITBI-SP + IPTU-SP -> base_modelagem_kan.csv
Avaliacao de Apartamentos Residenciais - Sao Paulo-SP
Referencia: ABNT NBR 14.653-2:2011 (Grau II)

Coloque este script na mesma pasta dos dados e execute:
    cd "C:\\Users\\acben\\OneDrive\\ACADEMICO\\PAPERS\\KAN + REGRESSAO SIMBOLICA\\BASE DE DADOS\\IPTU"
    set KMP_DUPLICATE_LIB_OK=TRUE
    python 00_integracao.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Caminhos
# ---------------------------------------------------------------------------

BASE_DIR      = Path.cwd()
ITBI_PATH     = BASE_DIR / "itbi_apartamentos_reduzido.csv"
IPTU_H5_2022  = BASE_DIR / "IPTU_2022.hdf5"
IPTU_H5_2021  = BASE_DIR / "IPTU_2021.hdf5"
IPTU_CSV_FALL = BASE_DIR / "iptu_apartamentos_sp.csv"
OUT_PATH      = BASE_DIR / "base_modelagem_kan.csv"

N_AMOSTRA    = 15_000
RANDOM_STATE = 42
DATA_INICIO  = pd.Timestamp("2021-01-01")
DATA_FIM     = pd.Timestamp("2025-12-31")
PADROES_ACEITOS = {"A", "B", "C", "D", "E", "F"}
USOS_ACEITOS    = {20, 21, 85}

print(f"Diretorio de trabalho: {BASE_DIR}")

# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------

def extrair_letra_padrao(texto):
    if not isinstance(texto, str):
        return None
    partes = texto.strip().split()
    ultima = partes[-1].upper() if partes else None
    return ultima if ultima in PADROES_ACEITOS else None

def limpar_colunas(df):
    new_cols = []
    for c in df.columns:
        c2 = c.strip().lower()
        # Remove BOM em qualquer codificacao
        c2 = c2.replace("\xef\xbb\xbf", "").replace("﻿", "").replace("ï»¿", "")
        new_cols.append(c2)
    df.columns = new_cols
    return df

def t_ordinal_fn(data):
    ref = pd.Timestamp("2021-01-01")
    return ((data.dt.year - ref.year) * 12 + (data.dt.month - ref.month)).astype(int)

# ---------------------------------------------------------------------------
# Leitor HDF5 vetorizado (formato Vaex/PyTables)
# ---------------------------------------------------------------------------

def _hdf5_str_fixo(grp):
    """String de comprimento fixo via numpy frombuffer (vetorizado)."""
    data    = grp["data"][:]
    indices = grp["indices"][:].astype(np.int64)
    L = int(indices[1] - indices[0])
    raw = np.frombuffer(data.tobytes(), dtype="S" + str(L))
    return list(raw.astype(str))

def _hdf5_ultima_letra(grp):
    """Extrai APENAS a ultima letra de cada string (vetorizado)."""
    data    = grp["data"][:]
    indices = grp["indices"][:].astype(np.int64)
    has_content = indices[1:] > indices[:-1]
    end_idx     = np.where(has_content, indices[1:] - 1, 0)
    last_bytes  = data[end_idx]
    result      = np.where(has_content,
                           np.char.decode(last_bytes, "ascii"),
                           "")
    return list(result)

def _hdf5_num(grp):
    """Coluna numerica com mascara de nulos."""
    data = grp["data"][:].astype(float)
    if "mask" in grp:
        data[grp["mask"][:]] = np.nan
    return data

# Colunas a extrair do HDF5
_HDF5_COLS = {
    "NUMERO DO CONTRIBUINTE":       ("num_contribuinte_sql",  "str_fixo"),
    "TIPO DE PADRAO DA CONSTRUCAO": ("padrao_letra",          "ultima_letra"),
    "AREA CONSTRUIDA":              ("area_construida_m2",    "num"),
    "AREA DO TERRENO":              ("area_terreno_m2",       "num"),
    "FATOR DE OBSOLESCENCIA":       ("fator_obsolescencia",   "num"),
    "QUANTIDADE DE PAVIMENTOS":     ("qtd_pavimentos",        "num"),
    "VALOR DO M2 DE CONSTRUCAO":    ("valor_m2_construcao",   "num"),
    "VALOR DO M2 DO TERRENO":       ("valor_m2_terreno",      "num"),
    "ANO DA CONSTRUCAO CORRIGIDO":  ("ano_construcao",        "num"),
}

_LEITORES = {
    "str_fixo":     _hdf5_str_fixo,
    "ultima_letra": _hdf5_ultima_letra,
    "num":          _hdf5_num,
}

def _carregar_hdf5_vaex(path):
    import h5py
    rec = {}
    with h5py.File(str(path), "r") as f:
        cols = f["table/columns"]
        for hdf_nome, (proj_nome, tipo) in _HDF5_COLS.items():
            if hdf_nome not in cols:
                print(f"    aviso: coluna ausente no HDF5: {hdf_nome}")
                continue
            try:
                rec[proj_nome] = _LEITORES[tipo](cols[hdf_nome])
            except Exception as e:
                print(f"    aviso: erro em '{hdf_nome}': {e}")
    return pd.DataFrame(rec)

def carregar_iptu_hdf5(path_2022, path_2021):
    frames = []
    for path, ano in [(path_2022, 2022), (path_2021, 2021)]:
        if not path.exists():
            print(f"  HDF5 nao encontrado: {path.name}")
            continue
        print(f"  Carregando {path.name} ...")
        try:
            df = _carregar_hdf5_vaex(path)
            df["_fonte_ano"] = ano
            frames.append(df)
            print(f"    {len(df):,} registros, {len(df.columns)} colunas")
        except Exception as e:
            print(f"  Falha ({path.name}): {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def carregar_iptu(path_2022, path_2021, csv_fall):
    df = carregar_iptu_hdf5(path_2022, path_2021)
    if not df.empty:
        return df
    if not csv_fall.exists():
        raise FileNotFoundError("Nenhuma fonte IPTU encontrada.")
    print(f"  Usando fallback CSV: {csv_fall.name}")
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(csv_fall, sep=";", encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df["_fonte_ano"] = 2022
    return df

# ---------------------------------------------------------------------------
# ETAPA 1 - ITBI
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("ETAPA 1 - Carga do ITBI")
print("="*60)

if not ITBI_PATH.exists():
    raise FileNotFoundError(f"Arquivo ITBI nao encontrado: {ITBI_PATH}")

df_itbi = pd.read_csv(ITBI_PATH, sep=";", encoding="utf-8-sig", low_memory=False)
df_itbi = limpar_colunas(df_itbi)
print(f"  Shape: {df_itbi.shape}")

# Garante tipo Int64 na chave de join (ITBI le como int64 nativo)
if "num_contribuinte_sql" in df_itbi.columns:
    df_itbi["num_contribuinte_sql"] = pd.to_numeric(
        df_itbi["num_contribuinte_sql"], errors="coerce"
    ).astype("Int64")

df_itbi["data_transacao"] = pd.to_datetime(df_itbi["data_transacao"], errors="coerce")
for col in ["valor_declarado", "valor_venal_ref", "valor_mercado", "valor_financiado",
            "area_construida_m2", "area_terreno_m2", "proporcao_transmitida", "fracao_ideal"]:
    if col in df_itbi.columns:
        df_itbi[col] = pd.to_numeric(
            df_itbi[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

print("\nFiltrando ITBI ...")
n0 = len(df_itbi)
df_itbi = df_itbi[df_itbi["natureza_transacao"].str.contains("COMPRA E VENDA", na=False, case=False)]
print(f"  Compra e venda:    {len(df_itbi):>8,}  (-{n0-len(df_itbi):,})")

n1 = len(df_itbi)
df_itbi = df_itbi[df_itbi["data_transacao"].between(DATA_INICIO, DATA_FIM)]
print(f"  Periodo 2021-2025: {len(df_itbi):>8,}  (-{n1-len(df_itbi):,})")

n2 = len(df_itbi)
df_itbi = df_itbi[df_itbi["cod_uso_iptu"].isin(USOS_ACEITOS)]
print(f"  Uso apartamento:   {len(df_itbi):>8,}  (-{n2-len(df_itbi):,})")

n3 = len(df_itbi)
df_itbi = df_itbi[(df_itbi["valor_mercado"] > 0) & (df_itbi["area_construida_m2"] > 0)]
print(f"  Valores positivos: {len(df_itbi):>8,}  (-{n3-len(df_itbi):,})")

n4 = len(df_itbi)
prop = df_itbi["proporcao_transmitida"].copy()
if prop.median() < 2:
    prop = prop * 100
df_itbi["proporcao_transmitida"] = prop
df_itbi = df_itbi[df_itbi["proporcao_transmitida"] >= 50]
print(f"  Proporcao >= 50%:  {len(df_itbi):>8,}  (-{n4-len(df_itbi):,})")

n5 = len(df_itbi)
ln_vm2 = np.log(df_itbi["valor_mercado"] / df_itbi["area_construida_m2"])
q1, q3 = ln_vm2.quantile([0.25, 0.75])
iqr = q3 - q1
df_itbi = df_itbi[(ln_vm2 >= q1 - 3*iqr) & (ln_vm2 <= q3 + 3*iqr)]
print(f"  Sem outliers 3xIQR:{len(df_itbi):>8,}  (-{n5-len(df_itbi):,})")

# ---------------------------------------------------------------------------
# ETAPA 2 - IPTU
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("ETAPA 2 - Carga do IPTU")
print("="*60)

df_iptu = carregar_iptu(IPTU_H5_2022, IPTU_H5_2021, IPTU_CSV_FALL)
df_iptu = limpar_colunas(df_iptu)
print(f"  Shape: {df_iptu.shape}")

# Garante coluna padrao_letra
if "padrao_letra" not in df_iptu.columns:
    if "padrao_construtivo" in df_iptu.columns:
        df_iptu["padrao_letra"] = df_iptu["padrao_construtivo"].apply(extrair_letra_padrao)
    else:
        df_iptu["padrao_letra"] = None

df_iptu["padrao_letra"] = df_iptu["padrao_letra"].astype(str).str.upper().str.strip()
df_iptu = df_iptu[df_iptu["padrao_letra"].isin(PADROES_ACEITOS)].copy()
print(f"  Apos filtro padrao: {len(df_iptu):,}")

for col in ["area_construida_m2", "area_terreno_m2", "fator_obsolescencia",
            "qtd_pavimentos", "valor_m2_construcao", "valor_m2_terreno", "idade_imovel"]:
    if col in df_iptu.columns:
        df_iptu[col] = pd.to_numeric(
            df_iptu[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

# Converte chave IPTU: "0010030001-4" -> int64 10030001  (remove digito verificador)
chave = "num_contribuinte_sql"
if chave in df_iptu.columns:
    col_chave = df_iptu[chave].astype(str)
    if col_chave.str.contains("-", na=False).any():
        # Formato HDF5: "XXXXXXXXXX-Y" — remove so o traco, mantem digito verificador
        # Ex: "0824190054-6" -> "08241900546" -> int 8241900546  (igual ao ITBI)
        df_iptu[chave] = pd.to_numeric(
            col_chave.str.replace("-", "", regex=False), errors="coerce"
        ).astype("Int64")
    else:
        df_iptu[chave] = pd.to_numeric(col_chave, errors="coerce").astype("Int64")
    print(f"  num_contribuinte_sql sample: {df_iptu[chave].dropna().iloc[:3].tolist()}")

if chave in df_iptu.columns and "_fonte_ano" in df_iptu.columns:
    df_iptu = (
        df_iptu.sort_values("_fonte_ano", ascending=False)
               .drop_duplicates(subset=[chave], keep="first")
    )
    print(f"  Apos deduplicacao:  {len(df_iptu):,}")

# ---------------------------------------------------------------------------
# ETAPA 3 - Join ITBI x IPTU
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("ETAPA 3 - Join ITBI x IPTU")
print("="*60)

cols_iptu_join = [c for c in [
    "num_contribuinte_sql", "padrao_letra",
    "fator_obsolescencia", "qtd_pavimentos",
    "valor_m2_construcao", "valor_m2_terreno", "idade_imovel",
] if c in df_iptu.columns]

df_merged = df_itbi.merge(df_iptu[cols_iptu_join], on="num_contribuinte_sql", how="left")
print(f"  Shape apos join: {df_merged.shape}")
print(f"  Match IPTU:      {df_merged['padrao_letra'].notna().mean()*100:.1f}%")

df_merged = df_merged[df_merged["padrao_letra"].notna()].copy()
print(f"  Apos filtro match: {len(df_merged):,}")

# ---------------------------------------------------------------------------
# ETAPA 4 - Engenharia de variaveis
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("ETAPA 4 - Engenharia de variaveis")
print("="*60)

df = df_merged.copy()
PADRAO_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

df["valor_m2"]         = df["valor_mercado"] / df["area_construida_m2"]
df["ln_valor_m2"]      = np.log(df["valor_m2"])
df["ln_valor_mercado"] = np.log(df["valor_mercado"])
df["padrao_ordinal"]   = df["padrao_letra"].map(PADRAO_MAP)

df["cep_num"] = pd.to_numeric(
    df["cep"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8).replace("", np.nan),
    errors="coerce"
)
cep_min, cep_max = df["cep_num"].min(), df["cep_num"].max()
df["cep_norm"] = (df["cep_num"] - cep_min) / (cep_max - cep_min)

df["t_ordinal"] = t_ordinal_fn(df["data_transacao"])

df["valor_financiado_num"] = pd.to_numeric(
    df["valor_financiado"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
).fillna(0)
df["flag_financiamento"] = (df["valor_financiado_num"] > 0).astype(int)
df["prop_financiamento"] = (
    df["valor_financiado_num"] / df["valor_mercado"]
).clip(0, 1).fillna(0)

df["flag_subdeclaracao"] = pd.to_numeric(
    df.get("flag_subdeclaracao", 0), errors="coerce"
).fillna(0).astype(int)

df["ano"]       = df["data_transacao"].dt.year
df["mes"]       = df["data_transacao"].dt.month
df["trimestre"] = df["data_transacao"].dt.quarter
df["padrao_tipo"] = "Residencial Vertical"
df["flag_flat"]   = (df["cod_uso_iptu"] == 85).astype(int)

print("  Distribuicao por padrao:")
for ord_, cnt in df["padrao_ordinal"].value_counts().sort_index().items():
    letra = {v: k for k, v in PADRAO_MAP.items()}[ord_]
    print(f"    Padrao {letra}: {cnt:>6,}")

# ---------------------------------------------------------------------------
# ETAPA 5 - Amostragem estratificada por padrao_ordinal
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("ETAPA 5 - Amostragem estratificada")
print("="*60)

COLS_FINAIS = [
    "num_contribuinte_sql",
    "ln_valor_m2", "valor_m2", "ln_valor_mercado",
    "data_transacao", "ano", "mes", "trimestre", "t_ordinal",
    "cep", "cep_num", "cep_norm", "logradouro", "bairro",
    "area_construida_m2", "area_terreno_m2",
    "padrao_ordinal", "padrao_letra", "padrao_tipo", "cod_padrao_iptu",
    "flag_financiamento", "prop_financiamento", "valor_financiado_num",
    "flag_subdeclaracao", "valor_declarado", "valor_venal_ref",
    "natureza_transacao", "proporcao_transmitida", "cartorio",
    "cod_uso_iptu", "descricao_uso", "flag_flat",
    "fator_obsolescencia", "qtd_pavimentos",
    "valor_m2_construcao", "valor_m2_terreno", "idade_imovel",
]
cols_ok  = [c for c in COLS_FINAIS if c in df.columns]
df_final = df[cols_ok].copy()
print(f"  Shape antes da amostragem: {df_final.shape}")

n_disp = len(df_final)
n_alvo = min(N_AMOSTRA, n_disp)

if n_disp <= n_alvo:
    df_sample = df_final.copy()
else:
    df_sample = df_final.groupby("padrao_ordinal", group_keys=False).apply(
        lambda g: g.sample(n=max(1, round(n_alvo * len(g) / n_disp)), random_state=RANDOM_STATE)
    )
    if len(df_sample) > n_alvo:
        df_sample = df_sample.sample(n=n_alvo, random_state=RANDOM_STATE)
    elif len(df_sample) < n_alvo:
        extras = df_final.drop(df_sample.index).sample(
            n=n_alvo - len(df_sample), random_state=RANDOM_STATE
        )
        df_sample = pd.concat([df_sample, extras])

df_sample = df_sample.reset_index(drop=True)
print(f"\n  Amostra final: {len(df_sample):,} registros")
for ord_, n in df_sample["padrao_ordinal"].value_counts().sort_index().items():
    letra = {v: k for k, v in PADRAO_MAP.items()}[ord_]
    print(f"    Padrao {letra}: {n:>5} ({n/len(df_sample)*100:.1f}%)")

# ---------------------------------------------------------------------------
# ETAPA 6 - Verificacao e exportacao
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("ETAPA 6 - Verificacao e exportacao")
print("="*60)

VARS_MODELO = [
    "ln_valor_m2", "area_construida_m2", "padrao_ordinal", "cep_norm",
    "t_ordinal", "flag_financiamento", "prop_financiamento",
    "flag_subdeclaracao", "area_terreno_m2",
]
nan_check = df_sample[VARS_MODELO].isnull().sum()
if nan_check.any():
    print("  AVISO - NaN remanescentes:")
    print(nan_check[nan_check > 0].to_string())
else:
    print("  OK - Sem NaN nas variaveis do modelo.")

df_sample.to_csv(OUT_PATH, index=False, sep=";", encoding="utf-8-sig")
print(f"\nConcluido: {OUT_PATH}")
print(f"   Registros : {len(df_sample):,}")
print(f"   Variaveis : {len(df_sample.columns)}")
print(f"   Colunas   : {list(df_sample.columns)}")
