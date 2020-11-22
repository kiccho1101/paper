import streamlit as st
import sys
from pathlib import Path

root_path = str(Path(__file__).parents[1])
sys.path.append(root_path)

from Web_Optimization.header import header
from Web_Optimization.click_rate_bayes import click_rate_bayes
from Web_Optimization.ab_testing import ab_testing


st.set_page_config(page_title="Web Optimzation")
st.set_option("deprecation.showPyplotGlobalUse", False)
header()
mode = st.selectbox("mode", ["Bayes Inference", "A/B Testing"])

if mode == "Bayes Inference":
    click_rate_bayes()
if mode == "A/B Testing":
    ab_testing()