"""
================================================================================
CONVERSÃO E FILTRO — IPTU-SP (HDF5/Arrow offset buffer → CSV)  v9 — definitivo
================================================================================
COMO USAR:
    1. Coloque este script na mesma pasta que IPTU_2022.hdf5
    2. pip install pandas tables numpy
    3. python converter_iptu_hdf5.py
================================================================================
"""
import warnings, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import tables

warnings.filterwarnings("ignore", category=tables.NaturalNameWarning)

PASTA          = Path(__file__).parent
ARQUIVO_SAIDA  = PASTA / "iptu_apartamentos_sp.csv"
ARQUIVOS_HDF5  = ["IPTU_2022.hdf5", "IPTU_2021.hdf5"]
PALAVRAS_APART = r"APART|DUPLEX|TRIPLEX|FLAT|STUDIO|KITCHENETTE|COBERTURA"
N_AMOSTRA      = 15_000

COLUNAS_ALVO = [
    "NUMERO DO CONTRIBUINTE",
    "TIPO DE USO DO IMOVEL",
    "TIPO DE PADRAO DA CONSTRUCAO",
    "ANO DA CONSTRUCAO CORRIGIDO",
    "AREA CONSTRUIDA",
    "AREA DO TERRENO",
    "AREA OCUPADA",
    "FRACAO IDEAL",
    "FATOR DE OBSOLESCENCIA",
    "QUANTIDADE DE PAVIMENTOS",
    "VALOR DO M2 DE CONSTRUCAO",
    "VALOR DO M2 DO TERRENO",
    "NOME DE LOGRADOURO DO IMOVEL",
    "NUMERO DO IMOVEL",
    "COMPLEMENTO DO IMOVEL",
    "BAIRRO DO IMOVEL",
    "CEP DO IMOVEL",
    "CODLOG DO IMOVEL",
    "ANO DO EXERCICIO",
    "NUMERO DO CONDOMINIO",
    "REFERENCIA DO IMOVEL",
]

RENOMEAR = {
    "NUMERO DO CONTRIBUINTE"       : "num_contribuinte_sql",
    "TIPO DE USO DO IMOVEL"        : "tipo_uso",
    "TIPO DE PADRAO DA CONSTRUCAO" : "padrao_construtivo",
    "ANO DA CONSTRUCAO CORRIGIDO"  : "ano_construcao",
    "AREA CONSTRUIDA"              : "area_construida_m2",
    "AREA DO TERRENO"              : "area_terreno_m2",
    "AREA OCUPADA"                 : "area_ocupada_m2",
    "FRACAO IDEAL"                 : "fracao_ideal",
    "FATOR DE OBSOLESCENCIA"       : "fator_obsolescencia",
    "QUANTIDADE DE PAVIMENTOS"     : "qtd_pavimentos",
    "VALOR DO M2 DE CONSTRUCAO"    : "valor_m2_construcao",
    "VALOR DO M2 DO TERRENO"       : "valor_m2_terreno",
    "NOME DE LOGRADOURO DO IMOVEL" : "logradouro",
    "NUMERO DO IMOVEL"             : "numero_imovel",
    "COMPLEMENTO DO IMOVEL"        : "complemento",
    "BAIRRO DO IMOVEL"             : "bairro",
    "CEP DO IMOVEL"                : "cep",
    "CODLOG DO IMOVEL"             : "codlog",
    "ANO DO EXERCICIO"             : "ano_exercicio",
    "NUMERO DO CONDOMINIO"         : "num_condominio",
    "REFERENCIA DO IMOVEL"         : "referencia_imovel",
}

def normalizar(s):
    s = str(s).upper().strip()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def decodificar_arrow_string(data, indices):
    """
    Decodifica coluna de strings no formato Arrow offset buffer.
    
    - data    : array |S1 — buffer contínuo de bytes de todas as strings
    - indices : array int32 — offset (posição inicial) de cada string no buffer
    
    O fim de cada string é o início da próxima string diferente.
    Constrói um mapa offset→string e aplica vetorialmente.
    """
    # Buffer de bytes plano
    buf = data.view(np.uint8)

    # Offsets únicos ordenados para delimitar cada string
    offsets_unicos = np.unique(indices)
    # Adicionar sentinela no fim
    offsets_com_fim = np.append(offsets_unicos, len(buf))

    # Construir mapa: offset_inicio → string decodificada
    offset_map = {}
    for i, off in enumerate(offsets_unicos):
        fim   = int(offsets_com_fim[i + 1])
        chunk = buf[int(off):fim].tobytes()
        try:
            s = chunk.decode('latin-1').strip()
        except Exception:
            s = chunk.decode('utf-8', errors='replace').strip()
        offset_map[int(off)] = s

    # Vetorizar: mapear cada registro pelo seu offset
    resultado = np.array([offset_map.get(int(idx), "") for idx in indices])
    return resultado

def ler_coluna_arrow(f, col_name):
    """Lê coluna do HDF5/Arrow, suportando offset buffer e formato numérico."""
    grp   = f.get_node(f'/table/columns/{col_name}')
    filhos = list(grp._v_children.keys())
    data  = grp.data[:]

    if 'indices' in filhos:
        indices = grp.indices[:]

        # Detectar formato: offset buffer (|S1) vs dicionário categórico
        if data.dtype == np.dtype('|S1') or data.dtype.itemsize == 1:
            # Formato Arrow offset buffer — decodificar via mapa de offsets
            return decodificar_arrow_string(data, indices)
        else:
            # Formato dicionário categórico convencional
            dicionario = np.array([
                d.decode('latin-1').strip() if isinstance(d, bytes)
                else str(d).strip()
                for d in data
            ])
            idx_max = int(indices.max())
            if idx_max >= len(dicionario):
                indices = np.clip(indices - 1, 0, len(dicionario) - 1)
            return dicionario[indices]

    else:
        # Coluna numérica direta
        if data.dtype.kind in ('S', 'O'):
            data = np.array([
                d.decode('latin-1').strip() if isinstance(d, bytes)
                else str(d).strip()
                for d in data
            ])
        if 'mask' in filhos:
            mask = grp.mask[:]
            if data.dtype.kind in ('f', 'i', 'u'):
                data = data.astype(float)
                data[mask] = np.nan
            else:
                data = data.astype(object)
                data[mask] = None
        return data

def ler_hdf5_vaex(path):
    print(f"\n  Arquivo: {path.name} ({path.stat().st_size/1e6:.0f} MB)")
    print("  Lendo colunas (aguarde ~10-15 min)...")

    colunas_lidas = {}

    with tables.open_file(str(path), mode='r') as f:
        todas = list(f.get_node('/table/columns')._v_groups.keys())

        for i, col in enumerate(COLUNAS_ALVO, 1):
            if col not in todas:
                print(f"  [{i:02d}/{len(COLUNAS_ALVO)}] '{col}' — não encontrada")
                continue

            print(f"  [{i:02d}/{len(COLUNAS_ALVO)}] {col}...", end=" ", flush=True)
            try:
                dados = ler_coluna_arrow(f, col)
                colunas_lidas[col] = dados

                if col == "TIPO DE USO DO IMOVEL":
                    unicos = list(dict.fromkeys(dados[:500000]))[:10]
                    print(f"OK ({len(dados):,}) | valores: {unicos}")
                elif col == "TIPO DE PADRAO DA CONSTRUCAO":
                    unicos = list(dict.fromkeys(dados[:500000]))[:8]
                    print(f"OK ({len(dados):,}) | valores: {unicos}")
                else:
                    print(f"OK ({len(dados):,})")

            except Exception as e:
                print(f"ERRO: {type(e).__name__}: {str(e)[:100]}")

    # Alinhar tamanhos (truncar para o menor)
    if colunas_lidas:
        n_min = min(len(v) for v in colunas_lidas.values())
        n_max = max(len(v) for v in colunas_lidas.values())
        if n_min != n_max:
            print(f"\n  Alinhando: {n_min:,} (mín) a {n_max:,} (máx) → truncando para {n_min:,}")
            colunas_lidas = {k: v[:n_min] for k, v in colunas_lidas.items()}

    print(f"\n  Colunas lidas: {len(colunas_lidas)}/{len(COLUNAS_ALVO)}")
    df = pd.DataFrame(colunas_lidas)
    print(f"  DataFrame: {len(df):,} × {len(df.columns)}")
    return df

def amostrar_estratificado(df, n=N_AMOSTRA, seed=42):
    col = next((c for c in ["padrao_construtivo", "TIPO DE PADRAO DA CONSTRUCAO"]
                if c in df.columns), None)
    if col is None or len(df) <= n:
        return df.sample(min(n, len(df)), random_state=seed)

    contagens  = df[col].value_counts()
    proporcoes = (contagens / len(df) * n).round().astype(int)
    diff = n - proporcoes.sum()
    if diff != 0:
        proporcoes.iloc[0] += diff

    frames = []
    for estrato, qtd in proporcoes.items():
        grupo = df[df[col] == estrato]
        qtd = min(int(qtd), len(grupo))
        if qtd > 0:
            frames.append(grupo.sample(qtd, random_state=seed))

    amostra = pd.concat(frames, ignore_index=True)
    print(f"\n  Distribuição por padrão construtivo:")
    for v, c in amostra[col].value_counts().items():
        print(f"    {str(v):<35} {c:,} ({c/len(amostra)*100:.1f}%)")
    return amostra

def main():
    print("=" * 65)
    print("  IPTU-SP HDF5/Arrow → CSV  v9")
    print(f"  Alvo: {N_AMOSTRA:,} apartamentos (amostra estratificada)")
    print("=" * 65)

    path_escolhido = None
    for nome in ARQUIVOS_HDF5:
        p = PASTA / nome
        if p.exists():
            print(f"  ✓ {nome} ({p.stat().st_size/1e6:.0f} MB)")
            if path_escolhido is None:
                path_escolhido = p
        else:
            print(f"  ✗ {nome} — não encontrado")

    if not path_escolhido:
        print(f"\n  ERRO: Nenhum HDF5 encontrado em: {PASTA}")
        return

    df = ler_hdf5_vaex(path_escolhido)
    if df.empty:
        print("  ERRO: Nenhum dado lido.")
        return

    # Filtrar apartamentos
    col_uso = "TIPO DE USO DO IMOVEL"
    if col_uso in df.columns:
        df[col_uso] = df[col_uso].apply(
            lambda x: normalizar(str(x)) if x is not None else ""
        )
        mask = df[col_uso].str.contains(PALAVRAS_APART, na=False, regex=True)
        n_antes = len(df)
        df = df[mask].copy()
        print(f"\n  Filtro apartamentos: {n_antes:,} → {len(df):,}")
    else:
        print("\n  AVISO: coluna de tipo de uso ausente")

    if df.empty:
        print("  ERRO: Nenhum apartamento encontrado.")
        return

    # Renomear e calcular idade
    df = df.rename(columns={k: v for k, v in RENOMEAR.items() if k in df.columns})
    if "ano_construcao" in df.columns and "ano_exercicio" in df.columns:
        df["idade_imovel"] = (
            pd.to_numeric(df["ano_exercicio"], errors="coerce") -
            pd.to_numeric(df["ano_construcao"], errors="coerce")
        ).clip(lower=0)

    # Amostragem
    print(f"\n  Amostragem estratificada → {N_AMOSTRA:,}...")
    df = amostrar_estratificado(df, n=N_AMOSTRA)

    # Relatório
    print(f"\n  Nulidade na amostra final:")
    for col in df.columns:
        nn = df[col].isna().sum()
        print(f"    {col:<30} {nn/len(df)*100:5.1f}% nulos")

    # Exportar
    df.to_csv(str(ARQUIVO_SAIDA), index=False,
              encoding="utf-8-sig", sep=";", decimal=",")
    tam = ARQUIVO_SAIDA.stat().st_size / 1_000_000

    print(f"\n{'='*65}")
    print(f"  ✓ Arquivo : {ARQUIVO_SAIDA.name}")
    print(f"  ✓ Linhas  : {len(df):,} apartamentos")
    print(f"  ✓ Tamanho : {tam:.1f} MB")
    print(f"\n  Suba {ARQUIVO_SAIDA.name} no Claude para continuar.")
    print("=" * 65)

if __name__ == "__main__":
    main()
