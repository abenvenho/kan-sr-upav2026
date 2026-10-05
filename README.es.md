# KAN + Regresión Simbólica en la valuación de inmuebles

[Português](README.md) · **Español** · [English](README.en.md)

[![Código: Apache 2.0](https://img.shields.io/badge/c%C3%B3digo-Apache%202.0-blue.svg)](LICENSE)
[![Datos y textos: CC BY 4.0](https://img.shields.io/badge/datos%20y%20textos-CC%20BY%204.0-lightgrey.svg)](LICENSE-CC-BY-4.0)

Datos, código, modelos entrenados, artículo, presentación y aplicación interactiva del trabajo

> **Modelagem híbrida para avaliação de imóveis com redes neurais Kolmogorov–Arnold e regressão simbólica**
> (Modelización híbrida para la valuación de inmuebles con redes neuronales de Kolmogórov–Arnold y regresión simbólica)
> Agnaldo Calvi Benvenho — XL Congreso Panamericano de Valuación, UPAV 2026, Madrid.
> Eje temático: inteligencia artificial y nuevas tecnologías aplicadas a la valuación.

Todo lo necesario para verificar, reproducir y reutilizar el estudio está aquí: la base de 15.000
transacciones, los scripts en el orden en que se ejecutaron, los pesos de la red entrenada, las
métricas y figuras publicadas, el artículo completo y las diapositivas de la presentación.

---

## Qué se hizo

La valuación de inmuebles se ha apoyado históricamente en modelos lineales: interpretables, pero
limitados para representar las no linealidades del mercado. Las redes neuronales MLP captan esas no
linealidades, pero son "cajas negras" y no cumplen la exigencia de la norma brasileña
**ABNT NBR 14.653-2** de un modelo explícito y justificable.

El estudio combina dos técnicas:

1. **Redes de Kolmogórov–Arnold (KAN)**: redes neuronales en las que cada conexión aprende una
   función univariada (B-spline) en lugar de un peso; captan no linealidades con menos parámetros y
   una estructura legible.
2. **Regresión simbólica (SR, PySR)**: búsqueda, mediante programación genética, de una expresión
   matemática cerrada que reproduzca la respuesta de la KAN. El resultado es una **ecuación** que puede
   leerse, auditarse y aplicarse en una hoja de cálculo.

Se compararon seis configuraciones en la misma partición 70/15/15, estratificada por estándar
constructivo: KAN, KAN+SR (ecuación destilada), OLS con transformaciones Box–Cox (referencia clásica
de la ingeniería de valuaciones), MLP de una y de dos capas ocultas, y SR aplicada directamente a los
datos, sin la KAN.

**Datos:** 15.000 transacciones de departamentos en el municipio de São Paulo (Brasil), obtenidas
cruzando las declaraciones pagadas del impuesto de transmisión (ITBI) con el catastro del impuesto
predial (IPTU). Variable dependiente: ln del valor de transacción por m² de área construida;
12 variables explicativas.

## Principales resultados (muestra de prueba, n = 2.250)

| Modelo | R² | RMSE (ln) | MAPE (%) | Forma |
|---|---:|---:|---:|---|
| **KAN** [12 → 6 → 1] | **0,716** | 0,219 | 16,7 | red (B-splines) |
| **KAN+SR** (ecuación c19) | 0,665 | 0,237 | 18,3 | ecuación cerrada, 6 variables |
| OLS Box–Cox | 0,661 | 0,239 | 18,0 | ecuación cerrada, 10 variables |
| MLP 1 capa oculta | 0,638 | 0,247 | 18,5 | red |
| SR aislada (sin KAN) | 0,608 | 0,257 | 19,8 | ecuación cerrada |
| MLP 2 capas ocultas | 0,600 | 0,259 | 18,5 | red |

Tabla completa (entrenamiento, validación y prueba): [`results/comparativo/metricas_comparativas_6modelos.csv`](results/comparativo/metricas_comparativas_6modelos.csv).

## Conclusiones

- La **KAN** obtuvo el mejor desempeño (R² = 0,716), superando al OLS y a las MLP de profundidad
  similar. La comparación con las MLP sugiere que la ventaja proviene en buena medida de las funciones
  de arista (B-splines), y no solo de la profundidad.
- La **destilación de la KAN mediante regresión simbólica** produce una ecuación cerrada, compacta y
  auditable, que iguala o supera levemente al OLS Box–Cox **sin fijar a priori la forma funcional**, y
  que es mejor que la ecuación obtenida por SR directamente de los datos (0,665 frente a 0,608).
- La ecuación destilada conserva cerca del 93% del R² de la red y reproduce la KAN con R² = 0,900.
- El enfoque KAN+SR concilia capacidad predictiva y transparencia, requisito esencial en la ingeniería
  de valuaciones, la pericia judicial y las aplicaciones regulatorias: la inteligencia artificial
  actúa al servicio del juicio profesional, no en su sustitución.

**Salvedades registradas en el artículo (Secciones 3.4 y 6.4).** La ventaja de KAN+SR sobre el OLS es
pequeña (ΔR² = 0,004) y aún no tiene cuantificación formal de incertidumbre; el R² = 0,665 de la
ecuación destilada es una estimación optimista, porque la expresión se seleccionó con inspección de la
muestra de prueba. El código postal (CEP) es una aproximación de localización de baja granularidad, y
la partición no controló la presencia del mismo edificio o microárea en entrenamiento y prueba. Se
recomiendan: coordenadas geográficas en lugar del CEP, intervalos de confianza del 80% para el
encuadre formal en la NBR 14.653-2, estimadores robustos, validación espacial y temporal, múltiples
semillas y bootstrap.

## Contenido del repositorio

```
paper/          Artículo completo (PDF, en portugués, con resumen en portugués, español e inglés)
presentation/   Presentación del Congreso UPAV 2026, Madrid (PDF y PPTX)
data/
  base_modelagem_kan.csv      15.000 × 35 — base de modelización (UTF-8, separada por comas)
  data_dictionary.csv         descripción, rol, tipo y rango de cada columna (en portugués)
  preprocessed/               nombres de variables, índices de la partición, normalizador y tensores
code/           Scripts 00–10 en orden de ejecución + conversores de las fuentes brutas (IPTU, ITBI)
models/         KAN entrenada (state_dict de PyTorch) y los mismos pesos en NumPy (kan_weights.npz)
results/        Métricas, ecuaciones y coeficientes de los seis modelos
figures/        Figuras del artículo y gráficos de diagnóstico
app/            Aplicación Streamlit
tools/          Utilidad de exportación de los pesos de la KAN a NumPy
```

## Aplicación interactiva

La aplicación recalcula en vivo la **KAN**, la **ecuación KAN+SR** y el **OLS Box–Cox** a partir de la
base y de los parámetros publicados, y reproduce las métricas del artículo hasta el 4.º decimal. Tiene
interfaz en portugués, español e inglés y cinco pestañas:

- **Sobre el estudio**: resumen, gráfico de R² de los seis modelos y ecuación destilada;
- **Base de datos**: filtros por estándar, año y financiamiento, distribuciones y descarga de la selección;
- **Desempeño de los modelos**: métricas por muestra, observado × previsto y residuos por estándar;
- **Curvas de efecto**: cómo responde cada modelo a cada variable, con las demás en la mediana;
- **Simulador**: estima el valor de un departamento con los tres modelos y advierte cuando hay
  extrapolación fuera del rango de la muestra de entrenamiento.

La KAN se ejecuta en NumPy puro (`app/kan_numpy.py`), sin PyTorch: sus predicciones coinciden con las
de la red original en los 15.000 registros (diferencia máxima de 1×10⁻⁶ en ln R$/m²).

**Ejecutar localmente** (Python 3.10 o superior):

```bash
git clone https://github.com/abenvenho/kan-sr-upav2026.git
cd kan-sr-upav2026
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Publicar gratis** en [Streamlit Community Cloud](https://share.streamlit.io): *Create app* →
repositorio `abenvenho/kan-sr-upav2026`, rama `main`, archivo `app/streamlit_app.py`.

## Reproducir el estudio

```bash
pip install -r code/requirements.txt      # torch, pykan, pysr (instala Julia), statsmodels...
cp data/base_modelagem_kan.csv code/
cd code
python 01_preprocessamento.py             # partición 70/15/15 y normalización -> preprocessed/
python 02_kan_treinamento.py              # entrena la KAN -> modelos/
python 03_regressao_simbolica.py          # SR sobre las funciones de activación de la KAN
python 04_sr_isolado.py                   # SR directamente sobre los datos (sin KAN)
python 06_sr_destilado_multivariado.py    # SR multivariada sobre las predicciones de la KAN
python 09_selecionar_equacao_c19.py       # selección de la ecuación de complejidad 19
python 07_regressao_linear.py             # OLS Box–Cox stepwise (referencia)
python 10_mlp_treinamento.py              # MLP de 1 y 2 capas
python 05_analise_comparativa.py          # tabla comparativa
python 08_graficos_comparativos.py        # figuras
```

`00_integracao.py`, `pre_processar_itbi_sp.py` y `converter_iptu_hdf5.py` documentan la construcción
de la base a partir de las fuentes brutas; no son necesarios para reproducir los modelos, porque la
base final está en `data/`. La partición es reproducible con la semilla (`RANDOM_STATE = 42`) y además
está grabada en `data/preprocessed/splits_index.json` y en la columna `split` de la base. Los scripts y
sus mensajes están en portugués.

## Ecuaciones

**KAN+SR (complejidad 19)**: variables normalizadas a [0, 1] con la muestra de entrenamiento:

```
ln(V/m²) = 7,6153 + P̃·(F̃ − S̃)³ + 0,4585·(Ṽc + Ãt) + √(1,9699·Ṽt)

P̃  = (padrao_ordinal − 1) / 5              F̃  = (fator_obsolescencia − 0,20) / 0,79
S̃  = flag_subdeclaracao                    Ṽc = (valor_m2_construcao − 1.291) / 2.499
Ãt = (area_terreno_m2 − 89) / 233.314      Ṽt = (valor_m2_terreno − 126) / 25.914
```

**OLS Box–Cox**: forma expandida, en unidades originales, con las transformaciones Box–Cox
(x^λ − 1)/λ escritas por extenso (es la forma implementada en `code/07` y en la aplicación; reproduce
R² = 0,661 en prueba):

```
ln(V/m²) = 5,4830 + 0,2909·ln(valor_m2_terreno) + 0,4621·fator_obsolescencia^1,8727
         + 0,001025·valor_m2_construcao^0,8141 − 0,9286·area_terreno_m2^(−0,1643)
         − 0,01313·qtd_pavimentos^0,7307 − 0,2741·flag_subdeclaracao + 0,1211·cep_norm
         + 0,002514·t_ordinal + 0,0306·flag_financiamento + 0,0230·padrao_ordinal
```

Coeficientes completos, errores estándar y valores p: [`results/ols/coeficientes.csv`](results/ols/coeficientes.csv);
parámetros λ: [`results/ols/boxcox_lambdas.json`](results/ols/boxcox_lambdas.json).

## Datos: origen, privacidad y limitaciones

- **Fuentes.** Declaraciones pagadas de ITBI (Secretaría Municipal de Hacienda de São Paulo, datos
  abiertos) y catastro del IPTU del Municipio de São Paulo (compilación CEM-USP). Son registros
  públicos; la base es un extracto derivado, filtrado y enriquecido.
- **Privacidad.** Se eliminaron los identificadores directos del inmueble (número de contribuyente/SQL
  y dirección). Barrio y CEP permanecen porque el CEP normalizado es variable del modelo.
- **Alcance.** Departamentos residenciales, municipio de São Paulo, transacciones de 2021 a 2025 (casi
  todas en 2024 y 2025), valores declarados de transmisión.
- **Localización.** La geocodificación completa no fue viable; `cep_norm` es la aproximación de
  localización, principal limitación metodológica.
- **Subdeclaración.** `flag_subdeclaracao` marca valores declarados inferiores al valor catastral de
  referencia, distorsión conocida del ITBI, modelada explícitamente en lugar de filtrada.

## Licencias

| Contenido | Licencia |
|---|---|
| Código (`code/`, `app/`, `tools/`) | [Apache License 2.0](LICENSE) |
| Datos, modelos, resultados y figuras (`data/`, `models/`, `results/`, `figures/`) | [CC BY 4.0](LICENSE-CC-BY-4.0) |
| Artículo y presentación (`paper/`, `presentation/`) | [CC BY 4.0](LICENSE-CC-BY-4.0) |

Ambas permiten el uso libre, incluso comercial, con atribución. Véase también [`NOTICE`](NOTICE).

## Cómo citar

> BENVENHO, A. C. Modelagem híbrida para avaliação de imóveis com redes neurais Kolmogorov–Arnold e
> regressão simbólica. En: XL Congreso Panamericano de Valuación — UPAV 2026, Madrid, 2026.

Metadatos de citación en [`CITATION.cff`](CITATION.cff) (GitHub ofrece "Cite this repository" en el
menú lateral). La misma base está publicada en
[Kaggle](https://www.kaggle.com/datasets/agnaldocalvibenvenho/kan-e-regresso-simblica).

## Diferencias respecto de la versión de Kaggle

- `code/05`, `07`, `08` y `09` leían la base con separador `;`; ahora detectan el separador
  automáticamente y funcionan con el CSV publicado (separado por comas).
- `code/07` etiquetaba como `log` las transformaciones Box–Cox con λ negativo y como `sqrt` las de
  λ ≈ 0,5 en el archivo de texto de la fórmula; la etiqueta ahora corresponde al cálculo efectivo.
  `results/ols/formula_ols.txt` fue regenerado. Coeficientes, λ y métricas no cambiaron.
- Nuevos: `app/`, `tools/`, `models/kan_weights.npz` y `models/scaler_minmax.json` (normalizador en
  JSON, para lectura sin `pickle`).

## Autor

**Agnaldo Calvi Benvenho**: ingeniero mecánico (POLI-USP), máster en Ingeniería de Valuaciones
(Universidad Politécnica de Valencia), BP Avaliações e Perícias.
