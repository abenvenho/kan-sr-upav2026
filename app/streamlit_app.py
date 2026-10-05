# SPDX-License-Identifier: Apache-2.0
"""
Aplicativo interativo do estudo
"Modelagem hibrida para avaliacao de imoveis com redes neurais Kolmogorov-Arnold
e regressao simbolica" (Benvenho, Congresso UPAV 2026, Madri).

Executar na raiz do repositorio:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelos as M  # noqa: E402
from i18n import IDIOMAS, NOMES_VAR, T  # noqa: E402

REPO = "https://github.com/abenvenho/kan-sr-upav2026"

# cores fixas por modelo (paleta categorica validada, slots 1-3; a cor segue o modelo)
COR = {"KAN": "#2a78d6", "KAN+SR": "#eb6834", "OLS Box-Cox": "#1baf7a"}
COR_NEUTRA = "#8a8984"

# variaveis usadas por cada modelo (KAN: todas as 12)
USA = {
    "KAN": set(M.FEATURES),
    "KAN+SR": {"valor_m2_terreno", "valor_m2_construcao", "area_terreno_m2",
               "fator_obsolescencia", "flag_subdeclaracao", "padrao_ordinal"},
    "OLS Box-Cox": set(M.FEATURES) - {"area_construida_m2", "prop_financiamento"},
}

st.set_page_config(page_title="KAN + SR · UPAV 2026", page_icon="🏢", layout="wide")


# --------------------------------------------------------------------------- utilidades
@st.cache_data(show_spinner=False)
def dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = M.base()
    return df, M.prever(df)


def br(x: float, casas: int = 0, lang: str = "pt") -> str:
    """Formata numero no padrao do idioma (pt/es: 1.234,5 · en: 1,234.5)."""
    s = f"{x:,.{casas}f}"
    return s if lang == "en" else s.replace(",", "§").replace(".", ",").replace("§", ".")


def num(casas: int):
    return lambda v: br(v, casas, lang)


def md(texto: str) -> str:
    """Escapa o cifrao de "R$" para o Markdown do Streamlit nao abrir modo matematico."""
    return texto.replace("$", "\\$")


def estilo(fig: go.Figure, altura: int = 380) -> go.Figure:
    fig.update_layout(
        separators=".," if lang == "en" else ",.",
        height=altura, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(namelength=-1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridwidth=0.5, zeroline=False)
    return fig


# --------------------------------------------------------------------------- cabecalho
with st.sidebar:
    lang = IDIOMAS[st.radio(T["idioma"]["pt"] + " / Language", list(IDIOMAS), horizontal=False)]
    st.caption(T["rodape"][lang])
    st.markdown(f"[GitHub]({REPO})")


def t(chave: str) -> str:
    return T[chave][lang]


st.title(t("titulo"))
st.caption(t("subtitulo"))

df, prev = dados()
abas = st.tabs(T["abas"][lang])


# =========================================================================== 1. Sobre
with abas[0]:
    col_txt, col_lado = st.columns([3, 2], gap="large")
    with col_txt:
        st.markdown(t("sobre_md"))
    with col_lado:
        pub = M.metricas_publicadas().query("split == 'test'").copy()
        pub["modelo"] = pub["modelo"].replace({"KAN+SR MV c19": "KAN+SR"})
        pub = pub.sort_values("r2")
        cores = [COR.get(m, COR_NEUTRA) for m in pub["modelo"]]
        fig = go.Figure(go.Bar(
            x=pub["r2"], y=pub["modelo"], orientation="h", marker_color=cores,
            text=[br(v, 3, lang) for v in pub["r2"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: R² = %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(title="R² · " + T["amostras"][lang]["test"], xaxis_range=[0, 0.8], bargap=0.35)
        st.plotly_chart(estilo(fig, 320), width="stretch")

        st.subheader(t("equacao"))
        st.latex(r"\begin{aligned}\ln\!\left(\tfrac{V}{m^2}\right) = {} & 7{,}6153"
                 r" + \tilde{P}\,\bigl(\tilde{F}_{ob}-\tilde{S}\bigr)^{3} \\"
                 r" & + 0{,}4585\,\bigl(\tilde{V}_{c}+\tilde{A}_{t}\bigr) \\"
                 r" & + \sqrt{1{,}9699\,\tilde{V}_{t}}\end{aligned}")
        legenda = {
            r"\tilde{P}": "padrao_ordinal", r"\tilde{F}_{ob}": "fator_obsolescencia",
            r"\tilde{S}": "flag_subdeclaracao", r"\tilde{V}_{c}": "valor_m2_construcao",
            r"\tilde{A}_{t}": "area_terreno_m2", r"\tilde{V}_{t}": "valor_m2_terreno",
        }
        st.markdown("  \n".join(f"${s}$ — {md(NOMES_VAR[v][lang])}" for s, v in legenda.items()))
        st.caption(t("equacao_nota"))

        st.subheader(t("materiais"))
        st.markdown(t("materiais_md").format(repo=REPO))


# =========================================================================== 2. Base
with abas[1]:
    f1, f2, f3 = st.columns(3)
    padroes = f1.multiselect(t("padrao"), M.PADROES, default=M.PADROES)
    anos = sorted(df["ano"].unique())
    sel_anos = f2.multiselect(t("ano"), anos, default=anos)
    fin = f3.radio(t("financiamento"), [t("todos"), t("com"), t("sem")], horizontal=True)

    sub = df[df["padrao_letra"].isin(padroes) & df["ano"].isin(sel_anos)]
    if fin == t("com"):
        sub = sub[sub["flag_financiamento"] == 1]
    elif fin == t("sem"):
        sub = sub[sub["flag_financiamento"] == 0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("n_reg"), br(len(sub), 0, lang))
    if len(sub):
        k2.metric(t("mediana_m2"), "R$ " + br(sub["valor_m2"].median(), 0, lang))
        k3.metric(t("area_med"), br(sub["area_construida_m2"].median(), 0, lang))
        k4.metric(t("subdecl"), br(100 * sub["flag_subdeclaracao"].mean(), 1, lang) + " %")

        c1, c2 = st.columns(2)
        fig = go.Figure(go.Histogram(
            x=sub["valor_m2"].clip(upper=sub["valor_m2"].quantile(0.995)), nbinsx=60,
            marker=dict(color=COR["KAN"], line=dict(color="rgba(0,0,0,0)", width=0)),
            hovertemplate="R$ %{x}/m²<br>%{y}<extra></extra>",
        ))
        fig.update_layout(title=t("hist_titulo"), bargap=0.08, xaxis_title="R$/m²", yaxis_title=t("contagem"))
        c1.plotly_chart(estilo(fig), width="stretch")

        fig = go.Figure()
        for p in [p for p in M.PADROES if p in padroes]:
            fig.add_trace(go.Box(y=sub.loc[sub["padrao_letra"] == p, "valor_m2"], name=p,
                                 marker_color=COR["KAN"], boxpoints=False, showlegend=False))
        fig.update_layout(title=t("box_titulo"), yaxis_title="R$/m²", xaxis_title=t("padrao"))
        c2.plotly_chart(estilo(fig), width="stretch")

        amostra = sub.sample(min(len(sub), 4000), random_state=0)
        fig = go.Figure(go.Scattergl(
            x=amostra["valor_m2_terreno"], y=amostra["valor_m2"], mode="markers",
            marker=dict(size=5, color=COR["KAN"], opacity=0.35),
            customdata=amostra[["padrao_letra", "bairro"]].fillna("—"),
            hovertemplate="PGV: R$ %{x:,.0f}/m²<br>R$ %{y:,.0f}/m²<br>%{customdata[0]} · %{customdata[1]}<extra></extra>",
        ))
        fig.update_layout(title=t("disp_titulo"), xaxis_title=NOMES_VAR["valor_m2_terreno"][lang],
                          yaxis_title="R$/m²", xaxis_type="log", yaxis_type="log")
        st.plotly_chart(estilo(fig, 420), width="stretch")

        st.download_button(t("baixar_csv"), sub.to_csv(index=False).encode("utf-8"),
                           "base_modelagem_kan_recorte.csv", "text/csv")
    with st.expander(t("dicionario")):
        st.dataframe(pd.read_csv(M.DATA / "data_dictionary.csv"), width="stretch", hide_index=True)


# =========================================================================== 3. Desempenho
with abas[2]:
    nomes_amostra = T["amostras"][lang]
    split = st.radio(t("amostra"), list(nomes_amostra), format_func=nomes_amostra.get, horizontal=True)
    msk = (df["split"] == split).to_numpy()
    y = df.loc[msk, M.TARGET].to_numpy()

    st.markdown(f"**{t('recalc')}**")
    linhas = [{t("modelo"): m, **M.metricas(y, prev.loc[msk, m].to_numpy())} for m in M.MODELOS]
    tab = pd.DataFrame(linhas)
    st.dataframe(tab.style.format({"R²": num(4), "RMSE (ln)": num(4), "MAE (ln)": num(4),
                                   "MAPE (%)": num(2), "n": num(0)}),
                 width="stretch", hide_index=True)

    with st.expander(t("publicadas")):
        pub = M.metricas_publicadas().query("split == @split").drop(columns="split")
        pub.columns = [t("modelo"), "R²", "RMSE (ln)", "MAPE (%)"]
        st.dataframe(pub.style.format({"R²": num(4), "RMSE (ln)": num(4), "MAPE (%)": num(2)}),
                     width="stretch", hide_index=True)
        st.caption(t("publicadas_nota"))

    lo, hi = np.quantile(y, [0.001, 0.999])
    fig = make_subplots(rows=1, cols=3, subplot_titles=M.MODELOS, shared_yaxes=True, horizontal_spacing=0.03)
    for i, m in enumerate(M.MODELOS, start=1):
        fig.add_trace(go.Scattergl(
            x=prev.loc[msk, m], y=y, mode="markers", name=m, showlegend=False,
            marker=dict(size=4, color=COR[m], opacity=0.35),
            hovertemplate=f"{t('previsto')}: %{{x:.3f}}<br>{t('observado')}: %{{y:.3f}}<extra>{m}</extra>",
        ), row=1, col=i)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=COR_NEUTRA, width=1, dash="dot")), row=1, col=i)
        fig.update_xaxes(title_text=t("previsto"), range=[lo, hi], row=1, col=i)
    fig.update_yaxes(title_text=t("observado"), range=[lo, hi], row=1, col=1)
    fig.update_layout(title=t("obs_prev"))
    estilo(fig, 430).update_layout(margin=dict(t=80))
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    letras = df.loc[msk, "padrao_letra"]
    for m in M.MODELOS:
        fig.add_trace(go.Box(x=letras, y=prev.loc[msk, m].to_numpy() - y, name=m, marker_color=COR[m],
                             boxpoints=False))
    fig.add_hline(y=0, line=dict(color=COR_NEUTRA, width=1, dash="dot"))
    fig.update_layout(title=t("res_padrao"), boxmode="group", yaxis_title=t("residuo"),
                      xaxis=dict(title=t("padrao"), categoryorder="array", categoryarray=M.PADROES))
    st.plotly_chart(estilo(fig, 400), width="stretch")


# =========================================================================== 4. Curvas
with abas[3]:
    st.markdown(t("curvas_intro"))
    var = st.selectbox(t("variavel"), M.FEATURES, index=M.FEATURES.index("valor_m2_terreno"),
                       format_func=lambda v: NOMES_VAR[v][lang])
    mediana = df[M.FEATURES].median()
    if df[var].nunique() <= 6:
        grade = np.sort(df[var].unique())
    else:
        grade = np.linspace(*df[var].quantile([0.01, 0.99]), 80)
    cen = pd.DataFrame([mediana] * len(grade))
    cen[var] = grade
    pc = M.prever(cen)

    fig = go.Figure()
    for m in M.MODELOS:
        fig.add_trace(go.Scatter(x=grade, y=np.exp(pc[m]), name=m, mode="lines+markers" if len(grade) <= 6 else "lines",
                                 line=dict(color=COR[m], width=2), marker=dict(size=8),
                                 hovertemplate=f"{m}: R$ %{{y:,.0f}}/m²<extra></extra>"))
    fig.update_layout(xaxis_title=NOMES_VAR[var][lang], yaxis_title=t("valor_prev"), hovermode="x unified")
    st.plotly_chart(estilo(fig, 440), width="stretch")
    nao = [m for m in M.MODELOS if var not in USA[m]]
    if nao:
        st.caption(t("nao_usa").format(m=", ".join(nao)))


# =========================================================================== 5. Simulador
with abas[4]:
    st.markdown(t("sim_intro"))
    teste = df[df["split"] == "test"].reset_index(drop=True)
    c_a, c_b = st.columns([2, 1])
    partida = c_a.radio(t("partida"), [t("partida_med"), t("partida_reg")], horizontal=True)
    if partida == t("partida_reg"):
        n_reg = c_b.number_input(t("registro"), 0, len(teste) - 1, 0, step=1)
        base_reg = teste.loc[int(n_reg)]
        semente = f"r{int(n_reg)}"
    else:
        base_reg = df[M.FEATURES].median()
        base_reg["cep"] = int(df["cep"].median())
        semente = "med"

    # um sufixo de chave por ponto de partida recria os campos com novos valores iniciais
    def chave(nome: str) -> str:
        return f"{nome}_{semente}"

    t_ini = int(base_reg["t_ordinal"])
    ano_ini, mes_ini = 2021 + t_ini // 12, t_ini % 12 + 1

    g1, g2, g3 = st.columns(3, gap="large")
    with g1:
        st.markdown(f"**{t('imovel')}**")
        padrao = st.select_slider(t("padrao"), M.PADROES, value=M.PADROES[int(base_reg["padrao_ordinal"]) - 1], key=chave("p"))
        area_c = st.number_input(t("area_constr"), 1, 2000, int(base_reg["area_construida_m2"]), key=chave("ac"))
        cep_txt = st.text_input(t("cep"), f"{int(base_reg['cep']):08d}", max_chars=9, key=chave("cep"))
    with g2:
        st.markdown(f"**{t('edificio')}**")
        area_t = st.number_input(t("area_terr"), 50, 300000, int(base_reg["area_terreno_m2"]), key=chave("at"))
        pav = st.number_input(t("pav"), 1, 60, int(base_reg["qtd_pavimentos"]), key=chave("pv"))
        obs = st.slider(t("obsol"), 0.20, 0.99, float(round(base_reg["fator_obsolescencia"], 2)), 0.01, key=chave("ob"))
        vt = st.number_input(t("vt"), 50, 40000, int(base_reg["valor_m2_terreno"]), step=50, key=chave("vt"))
        vc = st.number_input(t("vc"), 500, 6000, int(base_reg["valor_m2_construcao"]), step=50, key=chave("vc"))
    with g3:
        st.markdown(f"**{t('transacao')}**")
        ano = st.number_input(t("ano"), 2021, 2026, ano_ini, key=chave("an"))
        mes = st.number_input(t("mes"), 1, 12, mes_ini, key=chave("ms"))
        financiado = st.checkbox(t("financiado"), bool(base_reg["flag_financiamento"]), key=chave("fi"))
        prop = st.slider(t("prop_fin"), 0.0, 1.0, float(base_reg["prop_financiamento"]), 0.01,
                         disabled=not financiado, key=chave("pf"))
        subd = st.checkbox(t("subdecl_q"), bool(base_reg["flag_subdeclaracao"]), help=t("subdecl_h"), key=chave("sd"))

    digitos = "".join(ch for ch in cep_txt if ch.isdigit())
    if len(digitos) == 8:
        cep_norm = M.cep_para_norm(int(digitos))
    else:
        st.warning(t("cep_inval"))
        cep_norm = float(df["cep_norm"].median())

    x = pd.DataFrame([{
        "area_construida_m2": area_c, "padrao_ordinal": M.PADROES.index(padrao) + 1, "cep_norm": cep_norm,
        "t_ordinal": M.data_para_t(int(ano), int(mes)), "flag_financiamento": int(financiado),
        "prop_financiamento": prop if financiado else 0.0, "flag_subdeclaracao": int(subd),
        "area_terreno_m2": area_t, "valor_m2_terreno": vt, "valor_m2_construcao": vc,
        "qtd_pavimentos": pav, "fator_obsolescencia": obs,
    }])
    r = M.prever(x).iloc[0]

    st.divider()
    cols = st.columns(4 if partida == t("partida_reg") else 3)
    for col, m in zip(cols, M.MODELOS):
        v = float(np.exp(r[m]))
        col.markdown(f"<span style='color:{COR[m]};font-size:1.4em'>■</span> **{m}**", unsafe_allow_html=True)
        col.metric(t("por_m2"), "R$ " + br(v, 0, lang))
        col.caption(md(f"{t('total')}: R$ {br(v * area_c, 0, lang)}"))
    if partida == t("partida_reg"):
        cols[3].markdown(f"**{t('observado_reg')}**")
        cols[3].metric(t("por_m2"), "R$ " + br(float(base_reg["valor_m2"]), 0, lang))
        cols[3].caption(md(f"{t('total')}: R$ {br(float(base_reg['valor_m2']) * float(base_reg['area_construida_m2']), 0, lang)}"))

    faixa = M.faixa_treino()
    fora = [NOMES_VAR[v][lang] for v in M.FEATURES
            if not (faixa.loc[v, "min"] <= x.loc[0, v] <= faixa.loc[v, "max"])]
    if fora:
        st.warning(md(t("fora").format(v="; ".join(fora))))

st.divider()
st.caption(t("rodape"))
