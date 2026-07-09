import streamlit as st

def load_css():
    st.markdown("""
    <style>

    .block-container{
        padding-top:2rem;
        padding-bottom:2rem;
        max-width:1200px;
    }

    .hero-card{
        background:#0d1117;
        padding:25px;
        border-radius:15px;
        border:1px solid #30363d;
        margin-bottom:25px;
    }

    .repo-card{
        background:#161b22;
        padding:20px;
        border-radius:15px;
        border:1px solid #30363d;
        margin-bottom:20px;
    }

    .prediction{
        font-size:52px;
        font-weight:800;
        color:#58a6ff;
    }

    .subtitle{
        font-size:18px;
        color:#8b949e;
    }

    .small-title{
        font-size:14px;
        color:#8b949e;
        text-transform:uppercase;
        letter-spacing:1px;
    }

    .metric{
        font-size:30px;
        font-weight:700;
    }

    .section-title{
        font-size:28px;
        font-weight:700;
        margin-bottom:20px;
    }

    .footer{
        text-align:center;
        color:gray;
        padding-top:30px;
    }

    </style>
    """, unsafe_allow_html=True)