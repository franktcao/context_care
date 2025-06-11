from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st


@st.cache_data
def get_sessions() -> pd.DataFrame:
    dpath = Path.cwd() / "data" / "sessions"
    fpaths = dpath.glob("*.csv")

    dfs = [pd.read_csv(fpath, index_col=0) for fpath in fpaths]

    return pd.concat(dfs)


def get_metrics(
    df: pd.DataFrame, feedback_filler: Literal[0, 1, None], beta: float = 1
) -> pd.DataFrame:
    user_feedback = df["user_feedback"]
    total_feedback = len(df)
    if feedback_filler is None:
        total_feedback = len(user_feedback.dropna())
        positive_feedback = df["user_feedback"].sum()
    else:
        positive_feedback = user_feedback.fillna(feedback_filler).sum()
    TP = 1
    FN = 1
    FP = 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_beta = (1 + beta**2) * precision * recall / (precision * beta**2 + recall)
    accuracy = TP / (TP + FN + FP)
    metrics = [
        ("User satisfaction", positive_feedback / total_feedback, "{:.1%}"),
        ("Average response time (s)", df["response_time_sec"].mean(), "{:.2f}"),
        ("Precision", precision, "{:.2f}"),
        ("Recall", recall, "{:.2f}"),
        (f"f_{beta}", f_beta, "{:.2f}"),
        ("Accuracy", accuracy, "{:.2f}"),
    ]

    results = pd.DataFrame(metrics, columns=["metric", "value", "format"]).set_index(
        "metric"
    )

    return results


def display_metrics(df: pd.DataFrame) -> None:
    columns = st.columns(len(df))
    for i_col, (name, row) in enumerate(df.iterrows()):
        value = row["value"]
        format = row["format"]
        with columns[i_col]:
            st.metric(name, format.format(value))


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
display_metrics(metrics)


st.divider()
st.write("## Metrics by Session")
# session_id = st.selectbox("Session ID", df["session_id"])
for session_id, df_session in df.groupby("session_id"):
    st.write(f"### Session ID: `{session_id}`")
    metrics = get_metrics(df_session, treat_null_feedback)
    display_metrics(metrics)
