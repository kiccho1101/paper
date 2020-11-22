import streamlit as st
import sys
from pathlib import Path

root_path = str(Path(__file__).parents[1])
sys.path.append(root_path)

from Web_Optimization.click_rate_bayes import click_rate_bayes

mode = st.selectbox("mode", ["Bayes Inference"])

if mode == "Bayes Inference":
    click_rate_bayes()