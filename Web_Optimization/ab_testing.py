import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pymc3 as pm
from itertools import combinations

from typing import List


def ab_testing():
    N_str: str = st.text_input("N", value="434, 382, 394, 88").replace(" ", "")
    a_str: str = st.text_input("a", value="8, 17, 10, 4").replace(" ", "")
    N: List[int] = [int(s) for s in N_str.split(",")]
    a: List[int] = [int(s) for s in a_str.split(",")]
    if not len(N) == len(a):
        st.error("Lists must have the same lengths")
    st.table(
        pd.DataFrame(
            {
                "trial": [chr(65 + i) for i in range(len(N))],
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
                theta = pm.Uniform("theta", lower=0, upper=1, shape=len(N))
                obs = pm.Binomial("obs", p=theta, n=N, observed=a)
                trace = pm.sample(5000, chains=2, tune=1000, cores=1)
            for i in range(len(N)):
                sns.distplot(
                    trace["theta"][:, i], label=chr(65 + i), bins=100, kde=True
                )
            plt.legend()
            st.pyplot()

            summary = pm.summary(trace, hdi_prob=0.95)
            st.table(summary)

            st.subheader("Difference")
            diff_df = pd.DataFrame()
            for subset in combinations(list(range(len(N))), 2):
                diff = (
                    trace["theta"][:, subset[0]] - trace["theta"][:, subset[1]] > 0
                ).mean() * 100
                diff = diff if diff > 50 else 100 - diff
                diff_df = pd.concat(
                    [
                        diff_df,
                        pd.DataFrame(
                            [
                                {
                                    "trials": "{} - {}".format(
                                        chr(65 + subset[0]), chr(65 + subset[1])
                                    ),
                                    "diff": "{}%".format(diff),
                                }
                            ]
                        ),
                    ]
                )
            st.table(diff_df.sort_values("diff", ascending=False).set_index("trials"))

            st.subheader("Forest Plot")
            pm.forestplot(trace, combined=True, hdi_prob=0.95)
            st.pyplot()
