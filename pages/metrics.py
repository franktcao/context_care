from pathlib import Path
from typing import Literal
from langchain_community.utils.math import cosine_similarity

import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def get_sessions() -> pd.DataFrame:
    dpath = Path.cwd() / "data" / "sessions"
    fpaths = dpath.glob("*.csv")

    dfs = [pd.read_csv(fpath, index_col=0) for fpath in fpaths]

    return pd.concat(dfs)


def get_metrics(df: pd.DataFrame, feedback_filler: Literal[0, 1, None]) -> pd.DataFrame:
    user_feedback = df["user_feedback"]
    total_feedback = len(df)
    if feedback_filler is None:
        total_feedback = len(user_feedback.dropna())
        positive_feedback = df["user_feedback"].sum()
    else:
        positive_feedback = user_feedback.fillna(feedback_filler).sum()
    metrics = [
        ("User satisfaction", positive_feedback / total_feedback, "{:.1%}"),
        ("Average response time (s)", df["response_time_sec"].mean(), "{:.2f}"),
    ]

    results = pd.DataFrame(metrics, columns=["metric", "value", "format"]).set_index(
        "metric"
    )

    return results


def get_golden_set_metrics() -> pd.DataFrame:
    metrics = [
        ("Gold Set Average Similarity*", 0.606, "{:.2f}"),
    ]

    results = pd.DataFrame(metrics, columns=["metric", "value", "format"]).set_index(
        "metric"
    )

    return results


def display_metrics(df: pd.DataFrame) -> None:
    columns = st.columns(len(df))
    for i_col, (name, row) in enumerate(df.iterrows()):
        value = row["value"]
        format_str = row["format"]
        with columns[i_col]:
            st.metric(name, format_str.format(value))


with st.sidebar:
    st.write("# Config")
    feedback_mapping = {1: "Positive", 0: "Negative", None: "Skip"}
    treat_null_feedback = st.radio(
        "Treat null feedback",
        [1, 0, None],
        format_func=lambda x: feedback_mapping[x],
        help="If no feedback is provided, treat it as positive, negative, or skip it for metrics",
    )

df = get_sessions()
if len(df) == 0:
    st.write("Please use the [chatbot]() to generate chat history to")
    st.stop()

with st.expander("Sessions"):
    st.write(df)

st.write("## Top Line Metrics")
metrics = get_metrics(df, treat_null_feedback)
golden_set_metrics = get_golden_set_metrics()
metrics = pd.concat([metrics, golden_set_metrics], axis="rows")
display_metrics(metrics)
with st.expander("Golden Set Metrics*"):
    st.write(
        """
        The golden set metrics are an evaluation of the LLM's performance on
        questions where the answers are already known. Ideally there would be
        *many* questions that are both relevant and irrelevant to the article
        from subject matter experts.

        Please see the notebook `notebooks/03_gold_set_eval.ipynb` for more
        detail.
        """
    )

st.divider()

st.write("## Metrics by Session")
metrics_by_session = (
    df.groupby("session_id", group_keys=True)
    .apply(lambda x: get_metrics(x, treat_null_feedback))
    .reset_index()
)

columns = st.columns(metrics_by_session["metric"].nunique())
for i_col, (metric_name, df_metric) in enumerate(metrics_by_session.groupby("metric")):
    with columns[i_col]:
        fig = px.histogram(
            df_metric, x="metric", y="value", color="session_id", barmode="group"
        )
        st.plotly_chart(fig)

for session_id, df_session in metrics_by_session.groupby("session_id"):
    st.write(f"### Session ID: `{session_id}`")
    display_metrics(df_session.set_index("metric"))
