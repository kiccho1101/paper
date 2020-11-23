import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pymc3 as pm

from typing import List


def ab_testing_comb():
    img_str: str = st.text_input("image", value="0, 0, 1, 1").replace(" ", "")
    btn_str: str = st.text_input("button", value="0, 1, 0, 1").replace(" ", "")
    N_str: str = st.text_input("N", value="434, 382, 394, 88").replace(" ", "")
    a_str: str = st.text_input("a", value="8, 17, 10, 4").replace(" ", "")
    img: List[int] = [int(s) for s in img_str.split(",")]
    btn: List[int] = [int(s) for s in btn_str.split(",")]
    N: List[int] = [int(s) for s in N_str.split(",")]
    a: List[int] = [int(s) for s in a_str.split(",")]
    if not len(img) == len(btn) == len(N) == len(a):
        st.error("Lists must have the same lengths")
    st.table(
        pd.DataFrame(
            {
                "trial": [chr(65 + i) for i in range(len(img))],
                "img": img,
                "btn": btn,
                "N": N,
                "a": a,
                "a/N(%)": np.array(a) / np.array(N) * 100,
            }
        ).set_index("trial")
    )
    button = st.button("Run MCMC")
    if button:
        with pm.Model() as model:
            with st.spinner("Running MCMC..."):
                alpha = pm.Normal("alpha", mu=0, sigma=10)
                beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
                gamma = pm.Normal("gamma", mu=0, sigma=10)
                comb = alpha + beta[0] * img + beta[1] * btn + gamma * img * btn
                theta = pm.Deterministic("theta", 1 / (1 + pm.math.exp(-comb)))
                obs = pm.Binomial("obs", p=theta, n=N, observed=a)
                trace = pm.sample(5000, chains=2, tune=1000, cores=1)
            pm.traceplot(trace)
            st.pyplot()

            waic = pm.waic(trace, model)
            st.subheader("WAIC: {}".format(waic.p_waic))

            summary = pm.summary(trace, hdi_prob=0.95)
            st.table(summary)

            st.subheader("Forest Plot")
            pm.forestplot(trace, combined=True, hdi_prob=0.95)
            st.pyplot()
