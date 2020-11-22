import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


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

    st.subheader("Likelihood: Bernoulli Distribution")
    st.markdown(r"$p(r|\theta) = \theta^r (1-\theta)^(1-r)$")
    theta: float = st.slider("theta", 0.0, 1.0, value=0.5, step=0.01)
    likelihood = lambda r: theta if r == 1 else 1 - theta
    fig, ax = plt.subplots()
    ax.set(xlabel="r", ylabel=r"$p(r|\theta)$")
    ax.set_ylim(0, 1)
    ax.bar([0, 1], [likelihood(0), likelihood(1)])
    st.pyplot(fig)

    st.subheader("Posterior")
    st.markdown(r"$p(\theta|r) = p(r|\theta) * p(\theta)$")

    def posterior(
        r: int, thetas: np.ndarray, likelihood: Callable, prior: np.ndarray
    ) -> np.ndarray:
        lp = likelihood(r) * prior
        return lp / lp.sum()

    n_clicks: int = st.slider("#click", 0, 100, value=3, step=1)
    n_nonclicks: int = st.slider("#nonclick", 0, 100, value=3, step=1)
    likelihood = lambda r: thetas if r == 1 else 1 - thetas
    trials = [1] * n_clicks + [0] * n_nonclicks
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
        p = posterior(trials[i], thetas, likelihood, p)
        ax3.plot(thetas, p)
        ax3.set(xlabel=r"$\theta$", ylabel=r"$p(\theta|r)$")
        ax3.set_title("posterior")
        st.pyplot(fig)
