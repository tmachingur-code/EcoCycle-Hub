
from datetime import datetime

import pandas as pd
import streamlit as st


KEY = "ecocycle_history"


def initialise_history():

    if KEY not in st.session_state:
        st.session_state[KEY] = []


def add_scan(source_name, result):

    initialise_history()

    probabilities = result["probabilities"]

    record = {
        "Timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Source": source_name,

        "Model": "MobileNetV2",

        "Material": result["material"],

        "Confidence (%)": round(
            result["confidence"] * 100, 2
        ),

        "PET (%)": round(
            probabilities["PET"] * 100, 2
        ),

        "HDPEM (%)": round(
            probabilities["HDPEM"] * 100, 2
        ),

        "OTHER (%)": round(
            probabilities["OTHER"] * 100, 2
        ),
    }

    st.session_state[KEY].append(record)


def history_frame():

    initialise_history()

    columns = [
        "Timestamp",
        "Source",
        "Model",
        "Material",
        "Confidence (%)",
        "PET (%)",
        "HDPEM (%)",
        "OTHER (%)",
    ]

    return pd.DataFrame(
        st.session_state[KEY],
        columns=columns
    )
