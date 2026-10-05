"""
================================================================================
PRÉ-PROCESSAMENTO LOCAL — ITBI-SP  v3
Adaptado para o formato real das guias ITBI-SP com colunas enriquecidas
================================================================================
COMO USAR:
    1. Coloque este script na mesma pasta dos arquivos XLS/XLSX
    2. pip install pandas openpyxl xlrd
    3. python pre_processar_itbi_sp.py
================================================================================
"""

import unicodedata
from pathlib import Path
import pandas as pd

PASTA         = Path(__file__).parent
ARQUIVO_SAIDA = PASTA / "itbi_apartamentos_reduzido.csv"

# Abas a IGNORAR (metadados, não dados)
ABAS_IGNORAR = {
    "LEGENDA", "EXPLICACOES", "EXPLICAÇÕES",
    "TABELA DE USOS", "TABELA DE PADROES", "TABELA DE PADRÕES",
    "INSTRUCOES", "LEIAME", "README"
}

# Colunas a manter no output (nomes normalizados)
COLUNAS_SAIDA = {
    "N° DO CADASTRO (SQL)"                               : "num_contribuinte_sql",
    "NOME DO LOGRADOURO"                                 : "logradouro",
    "NUMERO"                                             : "numero_imovel",
    "COMPLEMENTO"                                        : "complemento",
    "BAIRRO"                                             : "bairro",
    "REFERENCIA"                                         : "referencia",
    "CEP"                                                : "cep",
    "NATUREZA DE TRANSACAO"                              : "natureza_transacao",
    "VALOR DE TRANSACAO (DECLARADO PELO CONTRIBUINTE)"   : "valor_declarado",
    "DATA DE TRANSACAO"                                  : "data_transacao",
    "VALOR VENAL DE REFERENCIA"                          : "valor_venal_ref",
    "PROPORCAO TRANSMITIDA (%)"                          : "proporcao_transmitida",
    "VALOR FINANCIADO"                                   : "valor_financiado",
    "CARTORIO DE REGISTRO"                               : "cartorio",
    "AREA DO TERRENO (M2)"                               : "area_terreno_m2",
    "FRACAO IDEAL"                                       : "fracao_ideal",
    "AREA CONSTRUIDA (M2)"                               : "area_construida_m2",
    "USO (IPTU)"                                         : "cod_uso_iptu",
    "DESCRICAO DO USO (IPTU)"                            : "descricao_uso",
    "PADRAO (IPTU)"                                      : "cod_padrao_iptu",
    "DESCRICAO DO PADRAO (IPTU)"                         : "descricao_padrao",
    "SITUACAO DO SQL"                                    : "situacao_sql",
}

def normalizar(s):
    s = str(s).upper().strip()
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")

def ler_arquivo(path):
    ext = path.suffix.lower()
    engines = ["openpyxl", "xlrd"] if ext != ".xls" else ["xlrd", "openpyxl"]

    for engine in engines:
        try:
            xls  = pd.ExcelFile(path, engine=engine)
            abas = xls.sheet_names

            # Filtrar abas de metadados
            abas_dados = [
                a for a in abas
                if normalizar(a) not in ABAS_IGNORAR
                and not any(x in normalizar(a) for x in
                            ["LEGENDA","EXPLICAC","TABELA","INSTRUC","LEIAME"])
            ]
            print(f"  Engine: {engine}")
            print(f"  Abas totais   : {abas}")
            print(f"  Abas de dados : {abas_dados}")

            frames = []
            for aba in abas_dados:
                try:
                    df = pd.read_excel(
                        path, sheet_name=aba, dtype=str, engine=engine
                    )
                    # Ignorar abas sem dados reais (menos de 10 linhas ou sem SQL)
                    df.columns = [normalizar(c) for c in df.columns]
                    col_sql = next(
                        (c for c in df.columns if "SQL" in c or "CADASTRO" in c), None
                    )
                    if col_sql and len(df) >= 10:
                        df["_aba"] = aba
                        frames.append(df)
                        print(f"    ✓ {aba}: {len(df):,} linhas")
                    else:
                        print(f"    ✗ {aba}: ignorada (sem coluna SQL ou poucos dados)")
                except Exception as e:
                    print(f"    ✗ {aba}: erro — {e}")

            if frames:
                return pd.concat(frames, ignore_index=True)

        except Exception as e:
            print(f"  Engine '{engine}' falhou: {e}")

    return pd.DataFrame()

def filtrar_e_selecionar(df, nome_arquivo):
    n_inicial = len(df)

    # ── Filtro 1: natureza da transação ──────────────────────────────────────
    col_nat = next(
        (c for c in df.columns if "NATUREZA" in c), None
    )
    if col_nat:
        df[col_nat] = df[col_nat].apply(normalizar)
        mask = df[col_nat].str.contains(
            "COMPRA|VENDA|COMPROMISSO", na=False, regex=True
        )
        df = df[mask].copy()
        print(f"  Filtro natureza     : {n_inicial:,} → {len(df):,}")
    else:
        print("  AVISO: coluna natureza não encontrada")

    # ── Filtro 2: tipo de uso — apartamentos ──────────────────────────────────
    # Tenta por descrição do uso primeiro, depois pelo código
    col_uso_desc = next(
        (c for c in df.columns if "DESCRICAO" in c and "USO" in c), None
    )
    col_uso_cod = next(
        (c for c in df.columns
         if "USO" in c and "IPTU" in c and "DESCRICAO" not in c), None
    )
    col_uso = col_uso_desc or col_uso_cod

    n_apos_nat = len(df)
    if col_uso:
        df[col_uso] = df[col_uso].apply(normalizar)
        # Mostrar valores únicos para diagnóstico
        print(f"  Coluna uso usada    : '{col_uso}'")
        print(f"  Valores únicos uso  : {list(df[col_uso].dropna().unique()[:10])}")

        mask = df[col_uso].str.contains(
            "APART|DUPLEX|TRIPLEX|FLAT|STUDIO|KITCHENETTE|COBERTURA|RESIDENCIAL VERTICAL",
            na=False, regex=True
        )
        df = df[mask].copy()
        print(f"  Filtro tipo uso     : {n_apos_nat:,} → {len(df):,}")
    else:
        print("  AVISO: coluna de uso não encontrada — sem filtro de tipo")

    # ── Selecionar e renomear colunas ─────────────────────────────────────────
    mapa = {k: v for k, v in COLUNAS_SAIDA.items() if k in df.columns}
    df_out = df[list(mapa.keys())].rename(columns=mapa).copy()
    df_out["_arquivo"] = nome_arquivo

    # Valor de mercado = maior entre declarado e venal de referência
    for col in ["valor_declarado", "valor_venal_ref"]:
        if col in df_out.columns:
            df_out[col] = (
                df_out[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\d.]", "", regex=True)
                .replace("", None)
                .astype(float)
            )
    if "valor_declarado" in df_out.columns and "valor_venal_ref" in df_out.columns:
        df_out["valor_mercado"] = df_out[
            ["valor_declarado", "valor_venal_ref"]
        ].max(axis=1)
        df_out["flag_subdeclaracao"] = (
            df_out["valor_declarado"] < df_out["valor_venal_ref"]
        ).astype(int)

    return df_out

def main():
    print("=" * 65)
    print("  PRÉ-PROCESSAMENTO ITBI-SP  v3")
    print("=" * 65)
    print(f"  Pasta: {PASTA}")

    arquivos = sorted(
        list(PASTA.glob("*.xls")) + list(PASTA.glob("*.xlsx")),
        key=lambda p: p.stat().st_size, reverse=True
    )
    arquivos = [a for a in arquivos if "itbi_apartamentos" not in a.name.lower()]

    if not arquivos:
        print("\n  ERRO: Nenhum arquivo .xls ou .xlsx encontrado.")
        return

    print(f"\n  Arquivos encontrados ({len(arquivos)}):")
    for a in arquivos:
        print(f"    {a.name} ({a.stat().st_size/1e6:.1f} MB)")

    frames = []
    for arquivo in arquivos:
        print(f"\n{'─'*55}")
        print(f"  Processando: {arquivo.name}")
        df = ler_arquivo(arquivo)
        if df.empty:
            print("  Sem dados lidos.")
            continue
        df_filtrado = filtrar_e_selecionar(df, arquivo.name)
        if not df_filtrado.empty:
            frames.append(df_filtrado)
            print(f"  Resultado: {len(df_filtrado):,} apartamentos")

    if not frames:
        print("\n  Nenhum dado encontrado após filtros.")
        return

    base = pd.concat(frames, ignore_index=True)

    # Deduplicar por SQL + valor + data
    chaves = [c for c in ["num_contribuinte_sql", "valor_declarado", "data_transacao"]
              if c in base.columns]
    if chaves:
        n_antes = len(base)
        base = base.drop_duplicates(subset=chaves, keep="first")
        print(f"\n  Deduplicação: {n_antes:,} → {len(base):,} (-{n_antes-len(base):,})")

    # Relatório final
    print(f"\n{'='*65}")
    print(f"  RESULTADO FINAL")
    print(f"{'='*65}")
    print(f"  Registros  : {len(base):,}")
    print(f"  Colunas    : {len(base.columns)}")
    print(f"\n  Colunas e nulidade:")
    for col in base.columns:
        nn  = base[col].isna().sum()
        pct = nn / len(base) * 100
        print(f"    {col:<35} {pct:5.1f}% nulos")

    base.to_csv(
        ARQUIVO_SAIDA, index=False,
        encoding="utf-8-sig", sep=";", decimal=","
    )
    tam = ARQUIVO_SAIDA.stat().st_size / 1_000_000
    print(f"\n  ✓ Arquivo : {ARQUIVO_SAIDA.name}")
    print(f"  ✓ Tamanho : {tam:.1f} MB")
    print(f"\n  Suba {ARQUIVO_SAIDA.name} no Claude para continuar.")
    print("=" * 65)

if __name__ == "__main__":
    main()
