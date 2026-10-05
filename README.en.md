# KAN + Symbolic Regression for real-estate appraisal

[Português](README.md) · [Español](README.es.md) · **English**

[![Code: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
[![Data and texts: CC BY 4.0](https://img.shields.io/badge/data%20and%20texts-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY-4.0)

Data, code, trained models, paper, slides and interactive app for the paper

> **Modelagem híbrida para avaliação de imóveis com redes neurais Kolmogorov–Arnold e regressão simbólica**
> (Hybrid modelling for real-estate appraisal with Kolmogorov–Arnold neural networks and symbolic regression)
> Agnaldo Calvi Benvenho — 40th Pan-American Valuation Congress, UPAV 2026, Madrid.
> Track: artificial intelligence and new technologies applied to valuation.

Everything needed to check, reproduce and reuse the study is here: the 15,000-transaction dataset,
the scripts in execution order, the trained network weights, the published metrics and figures, the
full paper and the conference slides.

---

## What was done

Real-estate appraisal has historically relied on linear models: interpretable, but limited in
representing market non-linearities. MLP neural networks capture those non-linearities but are
"black boxes" and fail the requirement of the Brazilian appraisal standard **ABNT NBR 14.653-2** for an
explicit, justifiable model.

The study combines two techniques:

1. **Kolmogorov–Arnold Networks (KAN)**: neural networks in which each connection learns a
   univariate function (a B-spline) instead of a weight; they capture non-linearities with fewer
   parameters and a readable structure.
2. **Symbolic regression (SR, PySR)**: a genetic-programming search for a closed-form mathematical
   expression that reproduces the KAN's response. The result is an **equation** that can be read,
   audited and applied in a spreadsheet.

Six configurations were compared on the same 70/15/15 split, stratified by construction standard:
KAN, KAN+SR (distilled equation), OLS with Box–Cox transforms (the classical benchmark in appraisal
engineering), MLPs with one and two hidden layers, and SR run directly on the data, without the KAN.

**Data:** 15,000 apartment transactions in the city of São Paulo, Brazil, built by joining paid
real-estate transfer tax returns (ITBI) with the property tax cadastre (IPTU). Target: ln of
transaction price per m² of built area; 12 explanatory variables.

## Main results (test split, n = 2,250)

| Model | R² | RMSE (ln) | MAPE (%) | Form |
|---|---:|---:|---:|---|
| **KAN** [12 → 6 → 1] | **0.716** | 0.219 | 16.7 | network (B-splines) |
| **KAN+SR** (equation c19) | 0.665 | 0.237 | 18.3 | closed form, 6 variables |
| OLS Box–Cox | 0.661 | 0.239 | 18.0 | closed form, 10 variables |
| MLP, 1 hidden layer | 0.638 | 0.247 | 18.5 | network |
| SR alone (no KAN) | 0.608 | 0.257 | 19.8 | closed form |
| MLP, 2 hidden layers | 0.600 | 0.259 | 18.5 | network |

Full table (train, validation and test): [`results/comparativo/metricas_comparativas_6modelos.csv`](results/comparativo/metricas_comparativas_6modelos.csv).

## Conclusions

- The **KAN** performed best (R² = 0.716), outperforming OLS and MLPs of similar depth. The comparison
  with the MLPs suggests the advantage comes largely from the edge functions (B-splines), not only
  from depth.
- **Distilling the KAN with symbolic regression** yields a closed-form, compact and auditable equation
  that matches or slightly outperforms OLS Box–Cox **without fixing the functional form in advance**,
  and is better than the equation SR finds directly from the data (0.665 vs 0.608).
- The distilled equation keeps about 93% of the network's R² and reproduces the KAN with R² = 0.900.
- The KAN+SR approach reconciles predictive power and transparency, an essential requirement in
  appraisal engineering, forensic expert work and regulatory applications: artificial intelligence
  serves professional judgement rather than replacing it.

**Caveats stated in the paper (Sections 3.4 and 6.4).** The KAN+SR advantage over OLS is small
(ΔR² = 0.004) and not yet backed by formal uncertainty quantification; the distilled equation's
R² = 0.665 is an optimistic estimate, because the expression was selected with the test split in
view. The postcode (CEP) is a coarse location proxy, and the split did not control for the same
building or micro-area appearing in both train and test. Recommended next steps: geographic
coordinates instead of CEP, 80% confidence intervals for formal NBR 14.653-2 grading, robust
estimators, spatial and temporal validation, multiple seeds and bootstrap.

## Repository contents

```
paper/          Full paper (PDF, in Portuguese, with abstracts in Portuguese, Spanish and English)
presentation/   UPAV 2026 Congress slides, Madrid (PDF and PPTX)
data/
  base_modelagem_kan.csv      15,000 × 35 — modelling dataset (UTF-8, comma-separated)
  data_dictionary.csv         description, role, type and range of each column (in Portuguese)
  preprocessed/               feature names, split indices, scaler and tensors
code/           Scripts 00–10 in execution order + raw-source converters (IPTU, ITBI)
models/         Trained KAN (PyTorch state_dict) and the same weights in NumPy (kan_weights.npz)
results/        Metrics, equations and coefficients of all six models
figures/        Paper figures and diagnostic plots
app/            Streamlit app
tools/          Utility that exports the KAN weights to NumPy
```

## Interactive app

The app recomputes the **KAN**, the **KAN+SR equation** and **OLS Box–Cox** live from the dataset and
the published parameters, reproducing the paper's metrics to the 4th decimal place. It has a
Portuguese, Spanish and English interface and five tabs:

- **About the study**: summary, R² chart of all six models and the distilled equation;
- **Dataset**: filters by standard, year and financing, distributions and download of the selection;
- **Model performance**: metrics per split, observed × predicted and residuals by standard;
- **Effect curves**: how each model responds to each variable, with the others held at the median;
- **Simulator**: estimates an apartment's value with all three models and warns when inputs fall
  outside the training range (extrapolation).

The KAN runs in pure NumPy (`app/kan_numpy.py`), without PyTorch: its predictions match the original
network on all 15,000 records (maximum difference 1×10⁻⁶ in ln R$/m²).

**Run locally** (Python 3.10 or later):

```bash
git clone https://github.com/abenvenho/kan-sr-upav2026.git
cd kan-sr-upav2026
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Publish for free** on [Streamlit Community Cloud](https://share.streamlit.io): *Create app* →
repository `abenvenho/kan-sr-upav2026`, branch `main`, file `app/streamlit_app.py`.

## Reproducing the study

```bash
pip install -r code/requirements.txt      # torch, pykan, pysr (installs Julia), statsmodels...
cp data/base_modelagem_kan.csv code/
cd code
python 01_preprocessamento.py             # 70/15/15 split and scaling -> preprocessed/
python 02_kan_treinamento.py              # trains the KAN -> modelos/
python 03_regressao_simbolica.py          # SR over the KAN activation functions
python 04_sr_isolado.py                   # SR directly on the data (no KAN)
python 06_sr_destilado_multivariado.py    # multivariate SR over the KAN predictions
python 09_selecionar_equacao_c19.py       # selects the complexity-19 equation
python 07_regressao_linear.py             # stepwise OLS Box–Cox (benchmark)
python 10_mlp_treinamento.py              # 1- and 2-layer MLPs
python 05_analise_comparativa.py          # comparison table
python 08_graficos_comparativos.py        # figures
```

`00_integracao.py`, `pre_processar_itbi_sp.py` and `converter_iptu_hdf5.py` document how the dataset
was built from the raw sources; they are not needed to reproduce the models, since the final dataset
is in `data/`. The split is reproducible from the seed (`RANDOM_STATE = 42`) and is also stored in
`data/preprocessed/splits_index.json` and in the dataset's `split` column. Script comments and
messages are in Portuguese.

## Equations

**KAN+SR (complexity 19)**: variables min-max scaled to [0, 1] on the training split:

```
ln(V/m²) = 7.6153 + P̃·(F̃ − S̃)³ + 0.4585·(Ṽc + Ãt) + √(1.9699·Ṽt)

P̃  = (padrao_ordinal − 1) / 5              F̃  = (fator_obsolescencia − 0.20) / 0.79
S̃  = flag_subdeclaracao                    Ṽc = (valor_m2_construcao − 1,291) / 2,499
Ãt = (area_terreno_m2 − 89) / 233,314      Ṽt = (valor_m2_terreno − 126) / 25,914
```

**OLS Box–Cox**: expanded form, in original units, with the Box–Cox transforms (x^λ − 1)/λ written out
(this is the form implemented in `code/07` and in the app; it reproduces test R² = 0.661):

```
ln(V/m²) = 5.4830 + 0.2909·ln(valor_m2_terreno) + 0.4621·fator_obsolescencia^1.8727
         + 0.001025·valor_m2_construcao^0.8141 − 0.9286·area_terreno_m2^(−0.1643)
         − 0.01313·qtd_pavimentos^0.7307 − 0.2741·flag_subdeclaracao + 0.1211·cep_norm
         + 0.002514·t_ordinal + 0.0306·flag_financiamento + 0.0230·padrao_ordinal
```

Full coefficients, standard errors and p-values: [`results/ols/coeficientes.csv`](results/ols/coeficientes.csv);
λ parameters: [`results/ols/boxcox_lambdas.json`](results/ols/boxcox_lambdas.json).

## Data: provenance, privacy and limitations

- **Sources.** Paid ITBI transfer-tax returns (São Paulo Municipal Finance Department, open data) and
  the São Paulo IPTU property-tax cadastre (CEM-USP compilation). Both are public records; the dataset
  is a derived, filtered and enriched extract.
- **Privacy.** Direct property identifiers (taxpayer/SQL number and street address) were removed.
  Neighbourhood and CEP remain because the normalised CEP is a model feature.
- **Scope.** Residential apartments, city of São Paulo, transactions from 2021 to 2025 (almost all in
  2024 and 2025), declared transfer values rather than asking prices.
- **Location.** Full geocoding was not feasible; `cep_norm` is the location proxy and the main
  methodological limitation.
- **Under-declaration.** `flag_subdeclaracao` flags declared values below the municipal reference
  value, a known distortion in ITBI data, modelled explicitly rather than filtered out.

## Licences

| Content | Licence |
|---|---|
| Code (`code/`, `app/`, `tools/`) | [Apache License 2.0](LICENSE) |
| Data, models, results and figures (`data/`, `models/`, `results/`, `figures/`) | [CC BY 4.0](LICENSE-CC-BY-4.0) |
| Paper and slides (`paper/`, `presentation/`) | [CC BY 4.0](LICENSE-CC-BY-4.0) |

Both allow free use, including commercial use, with attribution. See also [`NOTICE`](NOTICE).

## How to cite

> BENVENHO, A. C. Modelagem híbrida para avaliação de imóveis com redes neurais Kolmogorov–Arnold e
> regressão simbólica. In: XL Congreso Panamericano de Valuación — UPAV 2026, Madrid, 2026.

Citation metadata in [`CITATION.cff`](CITATION.cff) (GitHub shows "Cite this repository" in the
sidebar). The same dataset is published on
[Kaggle](https://www.kaggle.com/datasets/agnaldocalvibenvenho/kan-e-regresso-simblica).

## Differences from the Kaggle release

- `code/05`, `07`, `08` and `09` read the dataset with a `;` separator; they now detect the separator
  automatically and work with the published (comma-separated) CSV.
- `code/07` labelled Box–Cox transforms with negative λ as `log`, and those with λ ≈ 0.5 as `sqrt`, in
  the formula text file; the label now matches the actual computation. `results/ols/formula_ols.txt`
  was regenerated. Coefficients, λ values and metrics are unchanged.
- New: `app/`, `tools/`, `models/kan_weights.npz` and `models/scaler_minmax.json` (the scaler as JSON,
  readable without `pickle`).

## Author

**Agnaldo Calvi Benvenho**: mechanical engineer (POLI-USP), MSc in Appraisal Engineering (Universidad
Politécnica de Valencia), BP Avaliações e Perícias.
