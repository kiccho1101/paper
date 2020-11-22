import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from math import factorial


def click_rate_bayes():
    st.subheader("Posterior")
    st.markdown(r"$p(\theta|r) = p(r|\theta) * p(\theta)$")

    st.subheader("Prior: Uniform Distribution")
    st.markdown(r"$p(\theta)$ = 1")
    thetas = np.linspace(0, 1, 1001)
    p = np.array([1 / len(thetas) for _ in thetas])
    fig, ax = plt.subplots()
    ax.plot(thetas, p)
    ax.set(xlabel=r"$\theta$", ylabel=r"$p(\theta)$")
    st.pyplot(fig)

    likelihood_fn: str = st.selectbox("Likelihood Function", ["Bernoulli", "Binomial"])
    st.subheader(f"Likelihood: {likelihood_fn} Distribution")
    if likelihood_fn == "Bernoulli":
        st.markdown(r"$p(r|\theta) = \theta^r (1-\theta)^{(1-r)}$")
        theta: float = st.slider("theta", 0.0, 1.0, value=0.5, step=0.01)
        likelihood = lambda r: theta if r == 1 else 1 - theta
        fig, ax = plt.subplots()
        ax.set(xlabel="r", ylabel=r"$p(r|\theta)$")
        ax.set_ylim(0, 1)
        ax.bar([0, 1], [likelihood(0), likelihood(1)])
        st.pyplot(fig)
    else:
        st.markdown(r"$p(a|\theta, N) = {}_N C_a \theta^a (1-\theta)^{(N-a)}$")
        N: int = st.slider("N", 0, 100, value=60, step=1)
        theta: float = st.slider("theta", 0.0, 1.0, value=0.3, step=0.01)
        likelihood = (
            lambda a, N: factorial(N)
            // (factorial(N - a) * factorial(a))
            * theta ** a
            * (1 - theta) ** (N - a)
        )
        a_list = list(range(0, N))
        fig, ax = plt.subplots()
        ax.set(xlabel="a", ylabel=r"$p(r|\theta, N)$")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.bar(a_list, [likelihood(a, N) for a in a_list])
        st.pyplot(fig)

    st.subheader("Posterior")

    if likelihood_fn == "Bernoulli":

        st.markdown(r"$p(\theta|r) = p(r|\theta) * p(\theta)$")

        def posterior(r: int, likelihood: Callable, prior: np.ndarray) -> np.ndarray:
            lp = likelihood(r) * prior
            return lp / lp.sum()

        likelihood = lambda r: thetas if r == 1 else 1 - thetas
        trials_str: str = st.text_input("Trials", "1,1,1,0,0,0")
        trials = [int(trial) for trial in trials_str.split(",")]
        thetas = np.linspace(0, 1, 1001)
        p = np.array([1 / len(thetas) for _ in thetas])
        for i in range(len(trials)):
            st.subheader("{}".format("→".join([str(r) for r in trials[: i + 1]])))
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 3))
            ax1.plot(thetas, p)
            ax1.set(xlabel=r"$\theta$", ylabel=r"$p(\theta)$")
            ax1.set_title("prior")
            ax2.plot(thetas, likelihood(trials[i]))
            ax2.set(xlabel=r"$\theta$", ylabel=r"$p(r|\theta)$")
            ax2.set_title("likelihood")
            p = posterior(trials[i], likelihood, p)
            ax3.plot(thetas, p)
            ax3.set(xlabel=r"$\theta$", ylabel=r"$p(\theta|r)$")
            ax3.set_title("posterior")
            st.pyplot(fig)

    else:

        st.markdown(r"$p(\theta|a, N) = p(a|\theta, N) * p(\theta)$")

        def posterior(
            a: int, N: int, likelihood: Callable, prior: np.ndarray
        ) -> np.ndarray:
            lp = likelihood(a, N) * prior
            return lp / lp.sum()

        thetas = np.linspace(0, 1, 1001)
        likelihood = (
            lambda a, N: factorial(N)
            // (factorial(N - a) * factorial(a))
            * thetas ** a
            * (1 - thetas) ** (N - a)
        )
        n_clicks: int = st.slider("#click", 0, 30, value=3, step=1)
        n_nonclicks: int = st.slider("#nonclick", 0, 30, value=3, step=1)
        p = np.array([1 / len(thetas) for _ in thetas])
        fig, ax = plt.subplots(figsize=(16, 14))
        ax.plot(thetas, posterior(n_clicks, n_clicks + n_nonclicks, likelihood, p))
        ax.set(xlabel=r"$\theta$", ylabel=r"$p(\theta|r)$")
        ax.set_title("posterior")
        st.pyplot(fig)