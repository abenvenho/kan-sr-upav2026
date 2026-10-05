# SPDX-License-Identifier: Apache-2.0
"""Textos da interface em portugues, espanhol e ingles."""

IDIOMAS = {"Português": "pt", "Español": "es", "English": "en"}

T = {
    # ------------------------------------------------------------------ geral
    "titulo": {
        "pt": "KAN + Regressão Simbólica na avaliação de imóveis",
        "es": "KAN + Regresión Simbólica en la valuación de inmuebles",
        "en": "KAN + Symbolic Regression for real-estate appraisal",
    },
    "subtitulo": {
        "pt": "Apartamentos no município de São Paulo · 15.000 transações ITBI/IPTU · Congresso UPAV 2026, Madri",
        "es": "Departamentos en el municipio de São Paulo · 15.000 transacciones ITBI/IPTU · Congreso UPAV 2026, Madrid",
        "en": "Apartments in the city of São Paulo · 15,000 ITBI/IPTU transactions · UPAV 2026 Congress, Madrid",
    },
    "idioma": {"pt": "Idioma", "es": "Idioma", "en": "Language"},
    "abas": {
        "pt": ["Sobre o estudo", "Base de dados", "Desempenho dos modelos", "Curvas de efeito", "Simulador"],
        "es": ["Sobre el estudio", "Base de datos", "Desempeño de los modelos", "Curvas de efecto", "Simulador"],
        "en": ["About the study", "Dataset", "Model performance", "Effect curves", "Simulator"],
    },
    "rodape": {
        "pt": "Código: Apache 2.0 · Dados, artigo e apresentação: CC BY 4.0 · Benvenho, A. C. (2026)",
        "es": "Código: Apache 2.0 · Datos, artículo y presentación: CC BY 4.0 · Benvenho, A. C. (2026)",
        "en": "Code: Apache 2.0 · Data, paper and slides: CC BY 4.0 · Benvenho, A. C. (2026)",
    },
    # ------------------------------------------------------------------ sobre
    "sobre_md": {
        "pt": """
**Pergunta.** Uma rede neural capta melhor a estrutura não linear do mercado imobiliário do que a
regressão linear clássica — mas não pode ser apresentada em um laudo, porque a ABNT NBR 14.653-2
exige um modelo explícito e justificável. É possível ter as duas coisas?

**Método.** (1) Uma **rede Kolmogorov–Arnold (KAN)** é ajustada aos dados; (2) sua resposta é
**destilada por regressão simbólica (SR)** em uma equação fechada, que pode ser lida, auditada e
reproduzida em planilha. A referência de comparação é a **regressão linear com transformações
Box–Cox (OLS)**, prática corrente na engenharia de avaliações, além de duas redes MLP e da regressão
simbólica aplicada diretamente aos dados, sem a KAN.

**Resultado (amostra de teste, n = 2.250).** A KAN obtém R² = 0,716. A equação destilada (KAN+SR)
preserva cerca de 93% desse desempenho (R² = 0,665) usando apenas 6 variáveis e supera ligeiramente
o OLS Box–Cox (R² = 0,661). A SR aplicada sem a KAN fica em R² = 0,608: destilar a rede rende uma
equação melhor do que buscar a equação diretamente nos dados.

*Ressalvas do próprio artigo (Seções 3.4 e 6.4):* a vantagem da KAN+SR sobre o OLS é pequena
(ΔR² = 0,004) e ainda sem quantificação de incerteza; o R² da equação destilada é uma estimativa
otimista, porque a expressão foi escolhida com inspeção da amostra de teste.

**Este aplicativo** recalcula ao vivo a KAN, a equação KAN+SR e o OLS Box–Cox a partir da base e dos
parâmetros publicados no repositório, e reproduz os números do artigo.
""",
        "es": """
**Pregunta.** Una red neuronal capta mejor la estructura no lineal del mercado inmobiliario que la
regresión lineal clásica, pero no puede presentarse en un informe de valuación, porque la norma
brasileña ABNT NBR 14.653-2 exige un modelo explícito y justificable. ¿Es posible tener ambas cosas?

**Método.** (1) Se ajusta una **red Kolmogorov–Arnold (KAN)** a los datos; (2) su respuesta se
**destila mediante regresión simbólica (SR)** en una ecuación cerrada, que puede leerse, auditarse y
reproducirse en una hoja de cálculo. La referencia de comparación es la **regresión lineal con
transformaciones Box–Cox (OLS)**, práctica habitual en la ingeniería de valuaciones, además de dos
redes MLP y de la regresión simbólica aplicada directamente a los datos, sin la KAN.

**Resultado (muestra de prueba, n = 2.250).** La KAN alcanza R² = 0,716. La ecuación destilada
(KAN+SR) conserva cerca del 93% de ese desempeño (R² = 0,665) con solo 6 variables y supera
ligeramente al OLS Box–Cox (R² = 0,661). La SR aplicada sin la KAN queda en R² = 0,608: destilar la
red produce una ecuación mejor que buscarla directamente en los datos.

*Salvedades del propio artículo (Secciones 3.4 y 6.4):* la ventaja de KAN+SR sobre el OLS es
pequeña (ΔR² = 0,004) y aún sin cuantificación de incertidumbre; el R² de la ecuación destilada es
una estimación optimista, porque la expresión se eligió con inspección de la muestra de prueba.

**Esta aplicación** recalcula en vivo la KAN, la ecuación KAN+SR y el OLS Box–Cox a partir de la base
y de los parámetros publicados en el repositorio, y reproduce las cifras del artículo.
""",
        "en": """
**Question.** A neural network captures the non-linear structure of the housing market better than
classical linear regression, but it cannot be presented in an appraisal report: the Brazilian
standard ABNT NBR 14.653-2 requires an explicit, justifiable model. Can we have both?

**Method.** (1) A **Kolmogorov–Arnold Network (KAN)** is fitted to the data; (2) its response is
**distilled by symbolic regression (SR)** into a closed-form equation that can be read, audited and
reproduced in a spreadsheet. The benchmark is **linear regression with Box–Cox transforms (OLS)**,
standard practice in Brazilian appraisal engineering, alongside two MLP networks and symbolic
regression run directly on the data, without the KAN.

**Result (test split, n = 2,250).** The KAN reaches R² = 0.716. The distilled equation (KAN+SR)
keeps about 93% of that performance (R² = 0.665) with only 6 variables and slightly outperforms OLS
Box–Cox (R² = 0.661). SR without the KAN reaches R² = 0.608: distilling the network yields a better
equation than searching for one directly in the data.

*Caveats stated in the paper (Sections 3.4 and 6.4):* the KAN+SR advantage over OLS is small
(ΔR² = 0.004) and not yet backed by uncertainty quantification; the distilled equation's R² is an
optimistic estimate, because the expression was selected with the test split in view.

**This app** recomputes the KAN, the KAN+SR equation and OLS Box–Cox live from the dataset and the
parameters published in the repository, reproducing the paper's figures.
""",
    },
    "equacao": {"pt": "Equação destilada KAN+SR (complexidade 19)", "es": "Ecuación destilada KAN+SR (complejidad 19)", "en": "Distilled KAN+SR equation (complexity 19)"},
    "equacao_nota": {
        "pt": "Cada variável com til é normalizada para [0, 1] pela amostra de treino: x̃ = (x − mín) / (máx − mín). Coeficientes da normalização em results/sr_destilada/formula_c19.txt.",
        "es": "Cada variable con tilde se normaliza a [0, 1] con la muestra de entrenamiento: x̃ = (x − mín) / (máx − mín). Coeficientes de la normalización en results/sr_destilada/formula_c19.txt.",
        "en": "Each variable with a tilde is min-max scaled to [0, 1] on the training split: x̃ = (x − min) / (max − min). Scaling constants in results/sr_destilada/formula_c19.txt.",
    },
    "materiais": {"pt": "Materiais", "es": "Materiales", "en": "Materials"},
    "materiais_md": {
        "pt": "- [Artigo completo (PDF, em português, resumo trilíngue)]({repo}/blob/main/paper/Artigo_KAN_SR_UPAV_2026.pdf)\n- [Apresentação — Madri, 2026 (PDF)]({repo}/blob/main/presentation/Apresentacao_KAN_SR_UPAV_Madrid_2026.pdf)\n- [Repositório com dados e scripts]({repo})",
        "es": "- [Artículo completo (PDF, en portugués, resumen trilingüe)]({repo}/blob/main/paper/Artigo_KAN_SR_UPAV_2026.pdf)\n- [Presentación — Madrid, 2026 (PDF)]({repo}/blob/main/presentation/Apresentacao_KAN_SR_UPAV_Madrid_2026.pdf)\n- [Repositorio con datos y scripts]({repo})",
        "en": "- [Full paper (PDF, in Portuguese, trilingual abstract)]({repo}/blob/main/paper/Artigo_KAN_SR_UPAV_2026.pdf)\n- [Slides — Madrid, 2026 (PDF)]({repo}/blob/main/presentation/Apresentacao_KAN_SR_UPAV_Madrid_2026.pdf)\n- [Repository with data and scripts]({repo})",
    },
    # ------------------------------------------------------------------ base
    "filtros": {"pt": "Filtros", "es": "Filtros", "en": "Filters"},
    "padrao": {"pt": "Padrão construtivo", "es": "Estándar constructivo", "en": "Construction standard"},
    "ano": {"pt": "Ano", "es": "Año", "en": "Year"},
    "financiamento": {"pt": "Financiamento", "es": "Financiamiento", "en": "Financing"},
    "todos": {"pt": "Todos", "es": "Todos", "en": "All"},
    "com": {"pt": "Com financiamento", "es": "Con financiamiento", "en": "Financed"},
    "sem": {"pt": "Sem financiamento", "es": "Sin financiamiento", "en": "Not financed"},
    "n_reg": {"pt": "Transações", "es": "Transacciones", "en": "Transactions"},
    "mediana_m2": {"pt": "Mediana R$/m²", "es": "Mediana R$/m²", "en": "Median R$/m²"},
    "area_med": {"pt": "Área mediana (m²)", "es": "Área mediana (m²)", "en": "Median area (m²)"},
    "subdecl": {"pt": "Subdeclaração", "es": "Subdeclaración", "en": "Under-declared"},
    "hist_titulo": {"pt": "Distribuição do valor unitário (R$/m²)", "es": "Distribución del valor unitario (R$/m²)", "en": "Unit value distribution (R$/m²)"},
    "box_titulo": {"pt": "Valor unitário por padrão construtivo", "es": "Valor unitario por estándar constructivo", "en": "Unit value by construction standard"},
    "disp_titulo": {"pt": "Valor unitário × valor do terreno na Planta Genérica de Valores", "es": "Valor unitario × valor del terreno en la Planta Genérica de Valores", "en": "Unit value × land value in the municipal value map (PGV)"},
    "baixar_csv": {"pt": "Baixar recorte (CSV)", "es": "Descargar selección (CSV)", "en": "Download selection (CSV)"},
    "dicionario": {"pt": "Dicionário de dados", "es": "Diccionario de datos", "en": "Data dictionary"},
    "contagem": {"pt": "Transações", "es": "Transacciones", "en": "Transactions"},
    # ------------------------------------------------------------------ desempenho
    "amostra": {"pt": "Amostra", "es": "Muestra", "en": "Split"},
    "amostras": {
        "pt": {"test": "Teste (15%)", "val": "Validação (15%)", "train": "Treino (70%)"},
        "es": {"test": "Prueba (15%)", "val": "Validación (15%)", "train": "Entrenamiento (70%)"},
        "en": {"test": "Test (15%)", "val": "Validation (15%)", "train": "Train (70%)"},
    },
    "recalc": {
        "pt": "Métricas recalculadas agora, a partir da base e dos parâmetros publicados",
        "es": "Métricas recalculadas ahora, a partir de la base y de los parámetros publicados",
        "en": "Metrics recomputed now from the dataset and the published parameters",
    },
    "publicadas": {
        "pt": "Os seis modelos do artigo (valores publicados em results/comparativo)",
        "es": "Los seis modelos del artículo (valores publicados en results/comparativo)",
        "en": "All six models in the paper (published values in results/comparativo)",
    },
    "publicadas_nota": {
        "pt": "MLP e SR isolada não são recalculadas aqui: seus pesos/equações não fazem parte do aplicativo. Os scripts 04 e 10 as reproduzem.",
        "es": "MLP y SR aislada no se recalculan aquí: sus pesos/ecuaciones no forman parte de la aplicación. Los scripts 04 y 10 las reproducen.",
        "en": "MLP and standalone SR are not recomputed here: their weights/equations are not part of the app. Scripts 04 and 10 reproduce them.",
    },
    "obs_prev": {"pt": "Observado × previsto (ln R$/m²)", "es": "Observado × previsto (ln R$/m²)", "en": "Observed × predicted (ln R$/m²)"},
    "observado": {"pt": "Observado", "es": "Observado", "en": "Observed"},
    "previsto": {"pt": "Previsto", "es": "Previsto", "en": "Predicted"},
    "residuo": {"pt": "Resíduo (ln)", "es": "Residuo (ln)", "en": "Residual (ln)"},
    "res_padrao": {"pt": "Resíduos por padrão construtivo", "es": "Residuos por estándar constructivo", "en": "Residuals by construction standard"},
    "modelo": {"pt": "Modelo", "es": "Modelo", "en": "Model"},
    # ------------------------------------------------------------------ curvas
    "curvas_intro": {
        "pt": "Varie uma variável ao longo da faixa observada (percentis 1 a 99), mantendo as demais na mediana da base. É a forma mais direta de ver o que cada modelo aprendeu — e quanto a equação destilada se afasta da rede.",
        "es": "Varíe una variable a lo largo del rango observado (percentiles 1 a 99), manteniendo las demás en la mediana de la base. Es la forma más directa de ver lo que aprendió cada modelo y cuánto se aleja la ecuación destilada de la red.",
        "en": "Sweep one variable across its observed range (1st–99th percentile) with the others held at the dataset median. It is the most direct way to see what each model learned, and how far the distilled equation departs from the network.",
    },
    "variavel": {"pt": "Variável", "es": "Variable", "en": "Variable"},
    "valor_prev": {"pt": "Valor previsto (R$/m²)", "es": "Valor previsto (R$/m²)", "en": "Predicted value (R$/m²)"},
    "nao_usa": {
        "pt": "Não entra na equação de: {m}. A curva desses modelos é plana por construção.",
        "es": "No entra en la ecuación de: {m}. La curva de esos modelos es plana por construcción.",
        "en": "Not used by: {m}. Their curves are flat by construction.",
    },
    # ------------------------------------------------------------------ simulador
    "sim_intro": {
        "pt": "Informe as características de um apartamento e compare o valor estimado pelos três modelos. Uso didático: não substitui um laudo de avaliação.",
        "es": "Ingrese las características de un departamento y compare el valor estimado por los tres modelos. Uso didáctico: no sustituye un informe de valuación.",
        "en": "Enter the features of an apartment and compare the value estimated by the three models. For teaching only: it does not replace an appraisal report.",
    },
    "partida": {"pt": "Ponto de partida", "es": "Punto de partida", "en": "Starting point"},
    "partida_med": {"pt": "Mediana da base", "es": "Mediana de la base", "en": "Dataset median"},
    "partida_reg": {"pt": "Um registro da amostra de teste", "es": "Un registro de la muestra de prueba", "en": "A record from the test split"},
    "registro": {"pt": "Registro de teste nº", "es": "Registro de prueba n.º", "en": "Test record #"},
    "imovel": {"pt": "Imóvel", "es": "Inmueble", "en": "Property"},
    "edificio": {"pt": "Edifício e terreno", "es": "Edificio y terreno", "en": "Building and plot"},
    "transacao": {"pt": "Transação", "es": "Transacción", "en": "Transaction"},
    "area_constr": {"pt": "Área construída (m²)", "es": "Área construida (m²)", "en": "Built area (m²)"},
    "area_terr": {"pt": "Área do terreno do condomínio (m²)", "es": "Área del terreno del condominio (m²)", "en": "Condominium plot area (m²)"},
    "pav": {"pt": "Pavimentos do edifício", "es": "Pisos del edificio", "en": "Building storeys"},
    "obsol": {"pt": "Fator de obsolescência (IPTU)", "es": "Factor de obsolescencia (IPTU)", "en": "Obsolescence factor (IPTU)"},
    "vt": {"pt": "Valor do m² de terreno na PGV (R$)", "es": "Valor del m² de terreno en la PGV (R$)", "en": "PGV land value per m² (R$)"},
    "vc": {"pt": "Valor do m² de construção na PGV (R$)", "es": "Valor del m² de construcción en la PGV (R$)", "en": "PGV construction value per m² (R$)"},
    "cep": {"pt": "CEP (8 dígitos)", "es": "CEP (código postal, 8 dígitos)", "en": "CEP postcode (8 digits)"},
    "mes": {"pt": "Mês", "es": "Mes", "en": "Month"},
    "financiado": {"pt": "Transação financiada", "es": "Transacción financiada", "en": "Financed transaction"},
    "prop_fin": {"pt": "Proporção financiada", "es": "Proporción financiada", "en": "Financed share"},
    "subdecl_q": {
        "pt": "Subdeclaração no ITBI",
        "es": "Subdeclaración en el ITBI",
        "en": "Under-declared ITBI value",
    },
    "subdecl_h": {
        "pt": "Valor declarado abaixo do valor venal de referência. Para estimar valor de mercado, deixe desmarcado. A variável existe para que o modelo não confunda subdeclaração com preço baixo.",
        "es": "Valor declarado inferior al valor venal de referencia. Para estimar valor de mercado, déjelo sin marcar. La variable existe para que el modelo no confunda subdeclaración con precio bajo.",
        "en": "Declared value below the municipal reference value. Leave unchecked to estimate market value. The variable exists so the model does not mistake under-declaration for a low price.",
    },
    "cep_inval": {"pt": "CEP inválido — usando a mediana da base.", "es": "CEP inválido: se usa la mediana de la base.", "en": "Invalid CEP — using the dataset median."},
    "por_m2": {"pt": "R$/m²", "es": "R$/m²", "en": "R$/m²"},
    "total": {"pt": "Valor total", "es": "Valor total", "en": "Total value"},
    "observado_reg": {"pt": "Valor observado neste registro", "es": "Valor observado en este registro", "en": "Observed value for this record"},
    "fora": {
        "pt": "Fora da faixa da amostra de treino (extrapolação): {v}. Nenhum dos modelos foi ajustado nessa região.",
        "es": "Fuera del rango de la muestra de entrenamiento (extrapolación): {v}. Ningún modelo fue ajustado en esa región.",
        "en": "Outside the training range (extrapolation): {v}. None of the models was fitted in that region.",
    },
}

NOMES_VAR = {
    "area_construida_m2": {"pt": "Área construída (m²)", "es": "Área construida (m²)", "en": "Built area (m²)"},
    "padrao_ordinal": {"pt": "Padrão construtivo (1=A … 6=F)", "es": "Estándar constructivo (1=A … 6=F)", "en": "Construction standard (1=A … 6=F)"},
    "cep_norm": {"pt": "CEP normalizado", "es": "CEP normalizado", "en": "Normalised postcode"},
    "t_ordinal": {"pt": "Mês da transação (índice)", "es": "Mes de la transacción (índice)", "en": "Transaction month (index)"},
    "flag_financiamento": {"pt": "Financiamento (0/1)", "es": "Financiamiento (0/1)", "en": "Financing (0/1)"},
    "prop_financiamento": {"pt": "Proporção financiada", "es": "Proporción financiada", "en": "Financed share"},
    "flag_subdeclaracao": {"pt": "Subdeclaração (0/1)", "es": "Subdeclaración (0/1)", "en": "Under-declaration (0/1)"},
    "area_terreno_m2": {"pt": "Área do terreno (m²)", "es": "Área del terreno (m²)", "en": "Plot area (m²)"},
    "valor_m2_terreno": {"pt": "Valor m² terreno PGV (R$)", "es": "Valor m² terreno PGV (R$)", "en": "PGV land value (R$/m²)"},
    "valor_m2_construcao": {"pt": "Valor m² construção PGV (R$)", "es": "Valor m² construcción PGV (R$)", "en": "PGV construction value (R$/m²)"},
    "qtd_pavimentos": {"pt": "Pavimentos", "es": "Pisos", "en": "Storeys"},
    "fator_obsolescencia": {"pt": "Fator de obsolescência", "es": "Factor de obsolescencia", "en": "Obsolescence factor"},
}
