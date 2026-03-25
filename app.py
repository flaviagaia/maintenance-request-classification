from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CONFUSION_MATRIX_PATH, METRICS_PATH, MODEL_PATH, PROCESSED_DATA_PATH, SUMMARY_PATH
from src.modeling import load_best_model, predict_request
from src.pipeline import run_pipeline


st.set_page_config(page_title="Maintenance Request Classification", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(34, 197, 94, 0.10), transparent 26%),
            #07111f;
        color: #e5eef9;
    }
    .hero {
        background: rgba(10, 18, 32, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 22px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 18px 40px rgba(2, 6, 23, 0.35);
    }
    .hero h1, .hero p { color: #e5eef9; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Maintenance Request Classification</h1>
        <p>Classificação de solicitações operacionais inspirada em roteamento de manutenção, usando dados públicos do NYC 311.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Atualizar pipeline e reprocessar"):
    run_pipeline(refresh_download=False)

def _artifacts_ready() -> bool:
    if not all(path.exists() for path in [PROCESSED_DATA_PATH, METRICS_PATH, MODEL_PATH, SUMMARY_PATH]):
        return False
    try:
        pd.read_csv(PROCESSED_DATA_PATH, nrows=5)
        pd.read_csv(METRICS_PATH, nrows=5)
        json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


if not _artifacts_ready():
    run_pipeline(refresh_download=False)

dataset = pd.read_csv(PROCESSED_DATA_PATH)
metrics = pd.read_csv(METRICS_PATH)
summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
model = load_best_model()

metric_cols = st.columns(4)
metric_cols[0].metric("Solicitações", int(summary["rows"]))
metric_cols[1].metric("Grupos de manutenção", int(summary["maintenance_groups"]))
metric_cols[2].metric("Melhor modelo", summary["best_model"])
metric_cols[3].metric("Macro F1", f"{summary['best_macro_f1']:.3f}")

tab_predict, tab_metrics, tab_data = st.tabs(["Simular Chamado", "Métricas", "Base de Dados"])

with tab_predict:
    st.subheader("Teste o roteamento da solicitação")
    descriptor = st.text_area("Descrição do problema", value="pothole in front of school entrance causing traffic risk")
    borough = st.selectbox("Borough", sorted(dataset["borough"].dropna().unique().tolist()))
    location_type = st.selectbox("Tipo de localização", sorted(dataset["location_type"].fillna("unknown").unique().tolist()))
    street_name = st.text_input("Rua ou referência", value="Main Avenue")
    agency = st.selectbox("Agência", sorted(dataset["agency"].dropna().unique().tolist()))
    status = st.selectbox("Status do chamado", sorted(dataset["status"].fillna("Open").unique().tolist()))

    if st.button("Prever grupo de manutenção", use_container_width=True):
        feature_text = " | ".join([descriptor, location_type, street_name, borough, agency, status]).lower()
        prediction = predict_request(model, feature_text)
        st.success(f"Grupo previsto: {prediction}")

with tab_metrics:
    st.subheader("Comparação de modelos")
    st.caption("Este benchmark usa campos bastante informativos do NYC 311, então o desempenho alto deve ser interpretado como um cenário de roteamento com forte sinal operacional, não como benchmark universal.")
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(metrics, x="model", y="macro_f1", color="model", title="Macro F1 por modelo"),
        use_container_width=True,
    )
    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion matrix do melhor modelo", use_container_width=True)

with tab_data:
    st.subheader("Amostra pública processada")
    st.plotly_chart(
        px.histogram(
            dataset,
            x="maintenance_group",
            color="maintenance_group",
            title="Distribuição dos grupos de manutenção",
        ),
        use_container_width=True,
    )
    st.dataframe(
        dataset[
            [
                "sample_id",
                "descriptor",
                "borough",
                "location_type",
                "street_name",
                "maintenance_group",
            ]
        ].head(50),
        use_container_width=True,
        hide_index=True,
    )
