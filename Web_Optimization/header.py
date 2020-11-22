import streamlit as st


def header():
    st.markdown(
        """
            <center>
                <h1>Web Optimization Demo</h1>
                <a target="_blank" href="https://github.com/kiccho1101/paper/tree/master/Web_Optimization">
                    Source Code
                </a> |
                <a target="_blank" href="https://www.ohmsha.co.jp/book/9784873119168/">
                    Book
                </a>
            </center>
        """,
        unsafe_allow_html=True,
    )