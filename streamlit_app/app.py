import streamlit as st
import psycopg2
import pandas as pd

st.set_page_config(
    page_title="Voyage Analytics",
    layout="wide"
)

st.title("✈️ Voyage Analytics – Streamlit Cloud Deployment")

st.markdown("""
This Streamlit application demonstrates the cloud deployment of the  
**Voyage Analytics Travel Intelligence Platform**.

- Backend: Flask + ML models (Docker & Kubernetes)
- Frontend Deployment: Streamlit Cloud
- Database: Cloud PostgreSQL (Neon)
""")

# ---------------- DATABASE CONNECTION ---------------- #
def get_connection():
    return psycopg2.connect(st.secrets["DATABASE_URL"])

@st.cache_data
def load_users():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT user_code, name, age, gender, company FROM users LIMIT 5;",
        conn
    )
    conn.close()
    return df

@st.cache_data
def load_flights():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM flights LIMIT 5;",
        conn
    )
    conn.close()
    return df

@st.cache_data
def load_hotels():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM hotels LIMIT 5;",
        conn
    )
    conn.close()
    return df

tab1, tab2, tab3 = st.tabs(["👤 Users", "✈️ Flights", "🏨 Hotels"])

with tab1:
    st.subheader("Users Data (PostgreSQL)")
    st.dataframe(load_users())

with tab2:
    st.subheader("Flights Data (PostgreSQL)")
    st.dataframe(load_flights())

with tab3:
    st.subheader("Hotels Data (PostgreSQL)")
    st.dataframe(load_hotels())

st.success("✅ Securely connected to PostgreSQL using Streamlit Secrets")
