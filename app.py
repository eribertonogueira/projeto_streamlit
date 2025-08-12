
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from pathlib import Path

st.set_page_config(page_title="Dashboard de Telemetria", page_icon="🚚", layout="wide")

NEG = "NEGATIVO"
POS = "POSITIVO"
COLOR_MAP = {NEG: "#ef4444", POS: "#22c55e"}
BASE_TEMPLATE = "plotly_white"

def style_fig(fig, title=None):
    fig.update_layout(
        template=BASE_TEMPLATE,
        font=dict(family="Inter, Roboto, Segoe UI, sans-serif", size=13),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60 if title else 40, b=10),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, y=0.95, xanchor="left", font=dict(size=18, color="#0f172a")))
    return fig

#Detecta o tipo de separador
def _detectar_sep(f):
    if hasattr(f, "read") and not isinstance(f, (str, bytes, Path)):
        try:
            pos = f.tell()
        except Exception:
            pos = None
        chunk = f.read(8192)
        try:
            if pos is not None: f.seek(pos)
        except Exception:
            pass
        text = chunk.decode("utf-8", "ignore")
    else:
        with open(f, "rb") as fh:
            text = fh.read(8192).decode("utf-8", "ignore")
    counts = {";": text.count(";"), ",": text.count(","), "\t": text.count("\t")}
    sep = max(counts, key=counts.get)
    return sep if counts[sep] > 0 else ","



@st.cache_data(show_spinner=True)
def carregar_dados(fonte):
    sep = _detectar_sep(fonte)
    if hasattr(fonte, "seek"):
        try: fonte.seek(0)
        except Exception: pass

    df = pd.read_csv(fonte, sep=sep, engine="c", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    for c in ["veiculo","modelo_veiculo","motorista","empresa","evento","tipo_evento"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "data" in df.columns:
        s = df["data"].astype(str).str.strip()
        if s.str.contains("/").any():
            df["data"] = pd.to_datetime(s, errors="coerce", dayfirst=True)
        else:
            df["data"] = pd.to_datetime(s, errors="coerce")
        df = df.dropna(subset=["data"])
        df["dia"] = df["data"].dt.floor("D")
        df["hora"] = df["data"].dt.hour
        df["dow"] = df["data"].dt.dayofweek  # 0=Seg

    for c in ["rpm","velocidade","latitude","longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for cand in ["% pedal_acelerador","pedal_acelerador","acelerador","pedal"]:
        if cand in df.columns:
            df["pedal_acelerador"] = pd.to_numeric(df[cand], errors="coerce")
            break

    if "tipo_evento" in df.columns:
        df["tipo_evento"] = df["tipo_evento"].str.upper().replace({"NEGATIVO": NEG, "POSITIVO": POS})

    if "quantidade" not in df.columns:
        df["quantidade"] = 1

    return df


#Carrega os dados
df = carregar_dados('data/dados.csv')

st.title("🚚 Dashboard de Telemetria")
st.caption("Dashboard de eventos de telemetria veícular, com dados fornecidos pela CAN do veículo.")


st.sidebar.header("Filtros")
if "dia" not in df.columns:
    st.error("Coluna de data inválida. Verifique 'data' no CSV.")
    st.stop()

min_d, max_d = df["dia"].min().date(), df["dia"].max().date()
ini_default = max_d
fim_default = max_d
periodo = st.sidebar.date_input("Período", (ini_default, fim_default), min_value=min_d, max_value=max_d)
if isinstance(periodo, (list, tuple)) and len(periodo) == 2:
    ini, fim = periodo
else:
    ini, fim = ini_default, fim_default
if ini > fim:
    ini, fim = fim, ini

dt_ini = pd.Timestamp(ini)
dt_fim = pd.Timestamp(fim) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

empresas = sorted(df["empresa"].dropna().unique().tolist()) if "empresa" in df.columns else []
tipos = sorted(df["tipo_evento"].dropna().unique().tolist()) if "tipo_evento" in df.columns else []
eventos = sorted(df["evento"].dropna().unique().tolist()) if "evento" in df.columns else []
veics = sorted(df["veiculo"].dropna().unique().tolist()) if "veiculo" in df.columns else []

sel_emp = st.sidebar.multiselect("Empresa", empresas, default=empresas or None)
sel_tipo = st.sidebar.multiselect("Tipo", tipos, default=tipos or None)
sel_evt = st.sidebar.multiselect("Evento", eventos, default=eventos or None)
sel_veic = st.sidebar.multiselect("Veículo (opcional)", veics, default=[] if len(veics)>40 else veics)

mask = (df["data"] >= dt_ini) & (df["data"] <= dt_fim)
if sel_emp:  mask &= df["empresa"].isin(sel_emp)
if sel_tipo: mask &= df["tipo_evento"].isin(sel_tipo)
if sel_evt:  mask &= df["evento"].isin(sel_evt)
if sel_veic: mask &= df["veiculo"].isin(sel_veic)

f = df.loc[mask].copy()
if f.empty:
    st.warning("Sem dados para os filtros escolhidos.")
    st.stop()


#KPIs
k1, k2, k3, k4, k5 = st.columns(5)
total = int(f["quantidade"].sum())
neg = int(f.loc[f.get("tipo_evento","") == NEG, "quantidade"].sum()) if "tipo_evento" in f else 0
pos = int(f.loc[f.get("tipo_evento","") == POS, "quantidade"].sum()) if "tipo_evento" in f else 0
pct_neg = (100*neg/total) if total else 0
veic_unique = f["veiculo"].nunique() if "veiculo" in f else 0

k1.metric("Eventos (Σ)", f"{total:,}".replace(",", "."))
k2.metric("Negativos (Σ)", f"{neg:,}".replace(",", "."))
k3.metric("Positivos (Σ)", f"{pos:,}".replace(",", "."))
k4.metric("% Negativos", f"{pct_neg:.1f}%")
k5.metric("Veículos únicos", f"{veic_unique:,}".replace(",", "."))

st.divider()


#Top veículos NEG
l1a, l1b = st.columns((2,1))

gran = l1a.segmented_control("Granularidade", options=["Dia","Hora"], default="Hora")
suavizar = l1a.toggle("Suavizar curvas", value=True, help="Usa 'spline' para linhas mais suaves")
mm_win = l1a.slider("Média móvel (dias)", 0, 14, 7, help="0 desativa a média móvel", key="mmwin")

if gran == "Hora":
    grp = (f.set_index("data").groupby("tipo_evento")["quantidade"].resample("1H").sum().reset_index())
    grp = grp.rename(columns={"data":"ts"}) if "data" in grp.columns else grp
else:
    grp = (f.groupby(["dia","tipo_evento"], as_index=False)["quantidade"].sum()
             .rename(columns={"dia":"ts"}))

mode = "lines" if not suavizar else "lines"
fig_ts = go.Figure()
if "tipo_evento" in grp.columns:
    for t in [NEG, POS]:
        df_t = grp[grp["tipo_evento"] == t].sort_values("ts")
        if df_t.empty: 
            continue
        y = df_t["quantidade"]
        fig_ts.add_trace(go.Scatter(
            x=df_t["ts"], y=y, mode=mode, name=t, line=dict(color=COLOR_MAP[t], width=3, shape="spline" if suavizar else "linear"),
            fill="tozeroy", opacity=0.25
        ))
        if mm_win and gran == "Dia":
            mm = y.rolling(mm_win).mean()
            fig_ts.add_trace(go.Scatter(
                x=df_t["ts"], y=mm, mode="lines", name=f"MM{mm_win} {t}",
                line=dict(color=COLOR_MAP[t], width=2, dash="dot"), showlegend=True
            ))
else:
    df_t = grp.sort_values("ts")
    fig_ts.add_trace(go.Scatter(x=df_t["ts"], y=df_t["quantidade"], mode=mode, name="Total",
                                line=dict(color="#3b82f6", width=3, shape="spline" if suavizar else "linear"),
                                fill="tozeroy", opacity=0.25))

style_fig(fig_ts, "Eventos ao longo do tempo")
l1a.plotly_chart(fig_ts, use_container_width=True)

if "veiculo" in f.columns and "tipo_evento" in f.columns:
    top_neg = (f.loc[f["tipo_evento"]==NEG]
                 .groupby("veiculo", as_index=False)["quantidade"].sum()
                 .sort_values("quantidade", ascending=False).head(15))
    fig_top_neg = px.bar(top_neg, x="quantidade", y="veiculo", orientation="h",
                         color_discrete_sequence=[COLOR_MAP[NEG]], text_auto=True,
                         labels={"quantidade":"Negativos (Σ)","veiculo":"Veículo"})
    fig_top_neg.update_traces(textposition="outside")
    style_fig(fig_top_neg, "Top 15 veículos (eventos negativos)")
    l1b.plotly_chart(fig_top_neg, use_container_width=True)

st.divider()

#RPM x Velocidade
l2a, l2b = st.columns((2,1))
if {"rpm","velocidade"}.issubset(f.columns):
    size_dim = l2a.toggle("Tamanho pelo pedal do acelerador", value=False)
    scat = f.dropna(subset=["rpm","velocidade"]).copy()
    if size_dim and "pedal_acelerador" in scat.columns:
        fig_sc = px.scatter(
            scat, x="velocidade", y="rpm", color="tipo_evento" if "tipo_evento" in scat else None,
            color_discrete_map=COLOR_MAP, size="pedal_acelerador", size_max=16,
            labels={"velocidade":"Velocidade (km/h)","rpm":"RPM","tipo_evento":"Tipo","pedal_acelerador":"Pedal (%)"},
            opacity=0.65
        )
    else:
        fig_sc = px.scatter(
            scat, x="velocidade", y="rpm", color="tipo_evento" if "tipo_evento" in scat else None,
            color_discrete_map=COLOR_MAP, labels={"velocidade":"Velocidade (km/h)","rpm":"RPM","tipo_evento":"Tipo"},
            opacity=0.7
        )
    style_fig(fig_sc, "Relação RPM × Velocidade")
    l2a.plotly_chart(fig_sc, use_container_width=True)

if "empresa" in f.columns and "tipo_evento" in f.columns:
    modo_pct = l2b.toggle("Mostrar em % por empresa", value=True)
    emp = (f.groupby(["empresa","tipo_evento"], as_index=False)["quantidade"].sum())
    if modo_pct:
        tot = emp.groupby("empresa")["quantidade"].transform("sum")
        emp["pct"] = 100 * emp["quantidade"] / tot
        fig_emp = px.bar(emp, x="empresa", y="pct", color="tipo_evento",
                         color_discrete_map=COLOR_MAP, barmode="stack",
                         labels={"empresa":"Empresa","pct":"% de eventos","tipo_evento":"Tipo"},
                         text=emp["pct"].round(1).astype(str) + "%")
    else:
        fig_emp = px.bar(emp, x="empresa", y="quantidade", color="tipo_evento",
                         color_discrete_map=COLOR_MAP, barmode="stack",
                         labels={"empresa":"Empresa","quantidade":"Eventos","tipo_evento":"Tipo"},
                         text_auto=True)
    style_fig(fig_emp, "Eventos por empresa e tipo")
    l2b.plotly_chart(fig_emp, use_container_width=True)

st.divider()


#velocidade + Heatmap NEG
l3a, l3b = st.columns((2,1))
if "velocidade" in f.columns and "tipo_evento" in f.columns:
    vio = f.dropna(subset=["velocidade"]).copy()
    fig_vio = px.violin(vio, x="tipo_evento", y="velocidade", color="tipo_evento",
                        color_discrete_map=COLOR_MAP, box=True, points="outliers",
                        labels={"tipo_evento":"Tipo","velocidade":"Velocidade (km/h)"})
    style_fig(fig_vio, "Distribuição de velocidade por tipo (violin + box)")
    l3a.plotly_chart(fig_vio, use_container_width=True)

if {"hora","dow","tipo_evento"}.issubset(f.columns):
    neg_only = f[f["tipo_evento"]==NEG].copy()
    if not neg_only.empty:
        pivot = neg_only.pivot_table(index="dow", columns="hora", values="quantidade", aggfunc="sum").fillna(0)
        pivot = pivot.reindex([0,1,2,3,4,5,6])  # seg-dom
        fig_heat = px.imshow(pivot, aspect="auto", color_continuous_scale=[(0,"#fee2e2"), (1,"#b91c1c")],
                             labels=dict(x="Hora do dia", y="Dia da semana", color="Negativos (Σ)"))
        fig_heat.update_yaxes(ticktext=["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"], tickvals=list(range(7)))
        style_fig(fig_heat, "Heatmap de negativos (Hora × Dia)")
        l3b.plotly_chart(fig_heat, use_container_width=True)

st.divider()


#Mapa
l4a, l4b = st.columns((2,1))

if {"latitude","longitude"}.issubset(f.columns):
    locs = f.dropna(subset=["latitude","longitude"]).copy()
    if len(locs) > 6000:
        locs = locs.sample(6000, random_state=0)

    fig_map = px.scatter_mapbox(
        locs, lat="latitude", lon="longitude",
        color="tipo_evento" if "tipo_evento" in locs else None,
        color_discrete_map=COLOR_MAP,
        hover_data=["veiculo","empresa","evento"] if "evento" in locs.columns else ["veiculo","empresa"],
        zoom=11, height=460
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
    l4a.plotly_chart(style_fig(fig_map), use_container_width=True)


#Ranking
if "evento" in f.columns:
    rank_evt = (f.groupby(["evento","tipo_evento"], as_index=False)["quantidade"].sum()
                  .sort_values("quantidade", ascending=False).head(20))
    fig_evt = px.bar(rank_evt, x="quantidade", y="evento", color="tipo_evento",
                     orientation="h", color_discrete_map=COLOR_MAP,
                     labels={"quantidade":"Eventos (Σ)","evento":"Evento","tipo_evento":"Tipo"})
    fig_evt.update_traces(texttemplate="%{x}", textposition="outside")
    style_fig(fig_evt, "Top 20 eventos")
    l4b.plotly_chart(fig_evt, use_container_width=True)

st.divider()


#Download
with st.expander("Ver dados filtrados"):
    show = f.copy()
    show["data"] = show["data"].dt.strftime("%d/%m/%Y %H:%M:%S")
    st.dataframe(show, use_container_width=True, height=380)

csv_bytes = f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Baixar CSV filtrado", data=csv_bytes, file_name="telemetria_filtrado.csv", mime="text/csv")

st.caption("© 2025 — Dashboard de Telemetria (v4). Vermelho=Negativo, Verde=Positivo.")
