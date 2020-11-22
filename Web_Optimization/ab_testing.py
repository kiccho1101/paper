import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pymc3 as pm


def ab_testing():
    trials_str: str = st.text_input("N_A, a_A, N_B, a_B", value="30, 5, 40, 3").replace(
        " ", ""
    )
    N_A, a_A, N_B, a_B = [int(s) for s in trials_str.split(",")]
    with pm.Model() as model:
        with st.spinner("Running MCMC..."):
            theta = pm.Uniform("theta", lower=0, upper=1, shape=2)
            obs = pm.Binomial("obs", p=theta, n=[N_A, N_B], observed=[a_A, a_B])
            trace = pm.sample(5000, chains=2, tune=1000, cores=1)

        st.balloons()
        sns.distplot(trace["theta"][:, 0], label="A", color="red", bins=100, kde=True)
        sns.distplot(trace["theta"][:, 1], label="B", color="blue", bins=100, kde=True)
        plt.legend()
        st.pyplot()

        summary = pm.summary(trace, hdi_prob=0.95)
        st.table(summary)
        diff = (trace["theta"][:, 1] - trace["theta"][:, 0] > 0).mean() * 100
        st.subheader("p(A - B) > 0: {}%".format(diff if diff > 50 else 100 - diff))
