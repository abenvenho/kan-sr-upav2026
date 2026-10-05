# KAN + Regressão Simbólica na avaliação de imóveis

**Português** · [Español](README.es.md) · [English](README.en.md)

[![Código: Apache 2.0](https://img.shields.io/badge/c%C3%B3digo-Apache%202.0-blue.svg)](LICENSE)
[![Dados e textos: CC BY 4.0](https://img.shields.io/badge/dados%20e%20textos-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY-4.0)

Dados, código, modelos treinados, artigo, apresentação e aplicativo interativo do trabalho

> **Modelagem híbrida para avaliação de imóveis com redes neurais Kolmogorov–Arnold e regressão simbólica**
> Agnaldo Calvi Benvenho — XL Congresso Panamericano de Avaliação, UPAV 2026, Madri.
> Eixo temático: inteligência artificial e novas tecnologias aplicadas à valoração.

Tudo o que é necessário para conferir, reproduzir e reutilizar o estudo está aqui: a base de 15.000
transações, os scripts na ordem em que foram executados, os pesos da rede treinada, as métricas e
figuras publicadas, o artigo completo e os slides da apresentação.

---

## O que foi feito

A avaliação de imóveis apoia-se historicamente em modelos lineares: interpretáveis, mas limitados
para representar as não linearidades do mercado. Redes neurais MLP captam essas não linearidades,
mas são "caixas-pretas" e não atendem à exigência da **ABNT NBR 14.653-2** de um modelo explícito e
justificável.

O estudo combina duas técnicas:

1. **Redes Kolmogorov–Arnold (KAN)** — redes neurais em que cada conexão aprende uma função
   univariada (B-spline), em vez de um peso; captam não linearidades com menos parâmetros e
   estrutura legível.
2. **Regressão simbólica (SR, PySR)** — busca por programação genética de uma expressão matemática
   fechada que reproduza a resposta da KAN. O resultado é uma **equação** que pode ser lida, auditada
   e aplicada em planilha.

Seis configurações foram comparadas na mesma partição 70/15/15, estratificada por padrão construtivo:
KAN, KAN+SR (equação destilada), OLS com transformações Box–Cox (referência clássica da engenharia de
avaliações), MLP de uma e de duas camadas ocultas, e SR aplicada diretamente aos dados, sem a KAN.

**Dados:** 15.000 transações de apartamentos no município de São Paulo, obtidas pelo cruzamento das
guias de ITBI pagas com o cadastro do IPTU. Variável dependente: ln do valor de transação por m² de
área construída; 12 variáveis explicativas.

## Principais resultados (amostra de teste, n = 2.250)

| Modelo | R² | RMSE (ln) | MAPE (%) | Forma |
|---|---:|---:|---:|---|
| **KAN** [12 → 6 → 1] | **0,716** | 0,219 | 16,7 | rede (B-splines) |
| **KAN+SR** (equação c19) | 0,665 | 0,237 | 18,3 | equação fechada, 6 variáveis |
| OLS Box–Cox | 0,661 | 0,239 | 18,0 | equação fechada, 10 variáveis |
| MLP 1 camada oculta | 0,638 | 0,247 | 18,5 | rede |
| SR isolada (sem KAN) | 0,608 | 0,257 | 19,8 | equação fechada |
| MLP 2 camadas ocultas | 0,600 | 0,259 | 18,5 | rede |

Tabela completa (treino, validação e teste): [`results/comparativo/metricas_comparativas_6modelos.csv`](results/comparativo/metricas_comparativas_6modelos.csv).

## Conclusões

- A **KAN** teve o melhor desempenho (R² = 0,716), superando o OLS e as MLP de profundidade semelhante.
  A comparação com as MLP sugere que a vantagem vem em boa parte das funções de aresta (B-splines),
  e não só da profundidade.
- A **destilação da KAN por regressão simbólica** produz uma equação fechada, compacta e auditável,
  que iguala ou supera levemente o OLS Box–Cox **sem fixar a forma funcional a priori** — e que é
  melhor do que a equação obtida por SR diretamente dos dados (0,665 contra 0,608).
- A equação destilada preserva cerca de 93% do R² da rede e reproduz a KAN com R² = 0,900.
- A abordagem KAN+SR concilia capacidade preditiva e transparência, requisito essencial na
  engenharia de avaliações, na perícia judicial e em aplicações regulatórias: a inteligência
  artificial atua a serviço — e não em substituição — do juízo profissional.

**Ressalvas registradas no artigo (Seções 3.4 e 6.4).** A vantagem da KAN+SR sobre o OLS é pequena
(ΔR² = 0,004) e ainda não tem quantificação formal de incerteza; o R² = 0,665 da equação destilada é
uma estimativa otimista, porque a expressão foi selecionada com inspeção da amostra de teste. O CEP
é uma proxy de localização de baixa granularidade, e a partição não controlou a presença do mesmo
edifício ou microárea em treino e teste. Recomendam-se: coordenadas geográficas no lugar do CEP,
intervalos de confiança de 80% para o enquadramento formal na NBR 14.653-2, estimadores robustos,
validação espacial e temporal, múltiplas sementes e bootstrap.

## Conteúdo do repositório

```
paper/          Artigo completo (PDF, em português, com resumo em português, espanhol e inglês)
presentation/   Apresentação do Congresso UPAV 2026, Madri (PDF e PPTX)
data/
  base_modelagem_kan.csv      15.000 × 35 — base de modelagem (UTF-8, separada por vírgulas)
  data_dictionary.csv         descrição, papel, tipo e faixa de cada coluna
  preprocessed/               nomes das variáveis, índices da partição, normalizador e tensores
code/           Scripts 00–10 na ordem de execução + conversores das fontes brutas (IPTU, ITBI)
models/         KAN treinada (state_dict PyTorch) e os mesmos pesos em NumPy (kan_weights.npz)
results/        Métricas, equações e coeficientes dos seis modelos
figures/        Figuras do artigo e gráficos de diagnóstico
app/            Aplicativo Streamlit
tools/          Utilitário de exportação dos pesos da KAN para NumPy
```

## Aplicativo interativo

O aplicativo recalcula ao vivo a **KAN**, a **equação KAN+SR** e o **OLS Box–Cox** a partir da base
e dos parâmetros publicados e reproduz as métricas do artigo até a 4ª casa decimal. Tem interface em
português, espanhol e inglês e cinco abas:

- **Sobre o estudo** — resumo, gráfico de R² dos seis modelos e equação destilada;
- **Base de dados** — filtros por padrão, ano e financiamento, distribuições e download do recorte;
- **Desempenho dos modelos** — métricas por amostra, observado × previsto e resíduos por padrão;
- **Curvas de efeito** — como cada modelo responde a cada variável, mantidas as demais na mediana;
- **Simulador** — estima o valor de um apartamento pelos três modelos e alerta quando há
  extrapolação para fora da faixa da amostra de treino.

A KAN roda em NumPy puro (`app/kan_numpy.py`), sem PyTorch: as previsões coincidem com as da rede
original em todos os 15.000 registros (diferença máxima de 1×10⁻⁶ em ln R$/m²).

**Rodar localmente** (Python 3.10 ou superior):

```bash
git clone https://github.com/abenvenho/kan-sr-upav2026.git
cd kan-sr-upav2026
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Publicar gratuitamente** no [Streamlit Community Cloud](https://share.streamlit.io): *Create app* →
repositório `abenvenho/kan-sr-upav2026`, branch `main`, arquivo `app/streamlit_app.py`.

## Reproduzir o estudo

```bash
pip install -r code/requirements.txt      # torch, pykan, pysr (instala o Julia), statsmodels...
cp data/base_modelagem_kan.csv code/
cd code
python 01_preprocessamento.py             # partição 70/15/15 e normalização -> preprocessed/
python 02_kan_treinamento.py              # treina a KAN -> modelos/
python 03_regressao_simbolica.py          # SR sobre as funções de ativação da KAN
python 04_sr_isolado.py                   # SR diretamente sobre os dados (sem KAN)
python 06_sr_destilado_multivariado.py    # SR multivariada sobre as previsões da KAN
python 09_selecionar_equacao_c19.py       # seleção da equação de complexidade 19
python 07_regressao_linear.py             # OLS Box–Cox stepwise (referência)
python 10_mlp_treinamento.py              # MLP de 1 e 2 camadas
python 05_analise_comparativa.py          # tabela comparativa
python 08_graficos_comparativos.py        # figuras
```

`00_integracao.py`, `pre_processar_itbi_sp.py` e `converter_iptu_hdf5.py` documentam a construção da
base a partir das fontes brutas (guias de ITBI da SMF-SP e cadastro do IPTU em HDF5/Vaex, CEM-USP);
não são necessários para reproduzir os modelos, pois a base final está em `data/`. A partição é
reproduzível pela semente (`RANDOM_STATE = 42`) e também está gravada em
`data/preprocessed/splits_index.json` e na coluna `split` da base.

## Equações

**KAN+SR (complexidade 19)** — variáveis normalizadas para [0, 1] pela amostra de treino:

```
ln(V/m²) = 7,6153 + P̃·(F̃ − S̃)³ + 0,4585·(Ṽc + Ãt) + √(1,9699·Ṽt)

P̃  = (padrao_ordinal − 1) / 5              F̃  = (fator_obsolescencia − 0,20) / 0,79
S̃  = flag_subdeclaracao                    Ṽc = (valor_m2_construcao − 1.291) / 2.499
Ãt = (area_terreno_m2 − 89) / 233.314      Ṽt = (valor_m2_terreno − 126) / 25.914
```

**OLS Box–Cox** — forma expandida, em unidades originais, com as transformações Box–Cox
(x^λ − 1)/λ escritas por extenso (é a forma implementada em `code/07` e no aplicativo; reproduz
R² = 0,661 no teste):

```
ln(V/m²) = 5,4830 + 0,2909·ln(valor_m2_terreno) + 0,4621·fator_obsolescencia^1,8727
         + 0,001025·valor_m2_construcao^0,8141 − 0,9286·area_terreno_m2^(−0,1643)
         − 0,01313·qtd_pavimentos^0,7307 − 0,2741·flag_subdeclaracao + 0,1211·cep_norm
         + 0,002514·t_ordinal + 0,0306·flag_financiamento + 0,0230·padrao_ordinal
```

Coeficientes completos, erros-padrão e valores-p: [`results/ols/coeficientes.csv`](results/ols/coeficientes.csv);
parâmetros λ: [`results/ols/boxcox_lambdas.json`](results/ols/boxcox_lambdas.json).

## Dados: origem, privacidade e limitações

- **Fontes.** Guias de ITBI pagas (Secretaria Municipal da Fazenda de São Paulo, dados abertos) e
  cadastro do IPTU do Município de São Paulo (compilação CEM-USP). São registros públicos; a base é
  um extrato derivado, filtrado e enriquecido.
- **Privacidade.** Foram removidos os identificadores diretos do imóvel (número do contribuinte/SQL e
  logradouro). Bairro e CEP permanecem porque o CEP normalizado é variável do modelo.
- **Escopo.** Apartamentos residenciais, município de São Paulo, transações de 2021 a 2025 (quase
  todas em 2024 e 2025), valores declarados de transmissão.
- **Localização.** A geocodificação completa não foi viável; `cep_norm` é a proxy de localização —
  principal limitação metodológica.
- **Subdeclaração.** `flag_subdeclaracao` marca valores declarados abaixo do valor venal de
  referência, distorção conhecida do ITBI, modelada explicitamente em vez de filtrada.

## Licenças

| Conteúdo | Licença |
|---|---|
| Código (`code/`, `app/`, `tools/`) | [Apache License 2.0](LICENSE) |
| Dados, modelos, resultados e figuras (`data/`, `models/`, `results/`, `figures/`) | [CC BY 4.0](LICENSE-CC-BY-4.0) |
| Artigo e apresentação (`paper/`, `presentation/`) | [CC BY 4.0](LICENSE-CC-BY-4.0) |

Ambas permitem uso livre, inclusive comercial, com atribuição. Veja também [`NOTICE`](NOTICE).

## Como citar

> BENVENHO, A. C. Modelagem híbrida para avaliação de imóveis com redes neurais Kolmogorov–Arnold e
> regressão simbólica. In: XL Congresso Panamericano de Avaliação — UPAV 2026, Madri, 2026.

Metadados de citação em [`CITATION.cff`](CITATION.cff) (o GitHub oferece "Cite this repository" no
menu lateral). A mesma base está publicada no
[Kaggle](https://www.kaggle.com/datasets/agnaldocalvibenvenho/kan-e-regresso-simblica).

## Diferenças em relação à versão do Kaggle

- `code/05`, `07`, `08` e `09` liam a base com separador `;`; agora detectam o separador
  automaticamente e funcionam com o CSV publicado (separado por vírgulas).
- `code/07` rotulava como `log` transformações Box–Cox com λ negativo e como `sqrt` as com λ ≈ 0,5
  no arquivo de texto da fórmula; o rótulo agora corresponde ao cálculo efetivo.
  `results/ols/formula_ols.txt` foi regenerado. Coeficientes, λ e métricas não mudaram.
- Novos: `app/`, `tools/`, `models/kan_weights.npz` e `models/scaler_minmax.json` (normalizador em
  JSON, para leitura sem `pickle`).

## Autor

**Agnaldo Calvi Benvenho** — Engenheiro mecânico (POLI-USP), mestre em Engenharia de Avaliações
(Universidad Politécnica de Valencia), BP Avaliações e Perícias.
