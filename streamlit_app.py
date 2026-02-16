# ── pkg_resources compatibility patch ───────────────────────────────────────
import sys, types, importlib.metadata as _im

if 'pkg_resources' not in sys.modules:
    _pr = types.ModuleType('pkg_resources')
    class _Dist:
        def __init__(self, name):
            try:
                self.version = _im.version(name)
            except Exception:
                self.version = '0.0.0'
        def __str__(self):
            return self.version
    _pr.get_distribution = lambda name: _Dist(name)
    _pr.require = lambda *a, **kw: None
    _pr.working_set = []
    _pr.DistributionNotFound = Exception
    _pr.VersionConflict = Exception
    sys.modules['pkg_resources'] = _pr

import streamlit as st
import psycopg2
import pandas as pd
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from inference import predict_price
    from recommendation_engine import HotelRecommendationEngine, load_recommendation_models
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── FIXED CSS with proper navbar ────────────────────────────────────────────
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-blue: #2563eb;
        --primary-dark: #1e40af;
        --accent-teal: #14b8a6;
        --accent-purple: #8b5cf6;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --bg-light: #f8fafc;
        --bg-white: #ffffff;
        --border-color: #e2e8f0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background-color: var(--bg-light);
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* FIXED NAVBAR */
    .top-navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: white;
        border-bottom: 1px solid var(--border-color);
        padding: 1rem 2rem;
        z-index: 1000;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .navbar-container {
        max-width: 1400px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .brand-section {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .brand-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .brand-name {
        font-weight: 700;
        font-size: 1.25rem;
        color: var(--text-primary);
    }
    
    .user-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .user-avatar {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    .user-name {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.9375rem;
    }
    
    /* Content spacing for fixed navbar */
    .main-content {
        margin-top: 80px;
    }
    
    /* Welcome Card */
    .welcome-card {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        border-radius: 16px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .welcome-subtitle {
        font-size: 1.125rem;
        opacity: 0.95;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        transform: scaleX(0);
        transition: transform 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
        border-color: var(--primary-blue);
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    .feature-icon {
        width: 56px;
        height: 56px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        margin-bottom: 1.25rem;
        color: white;
    }
    
    .feature-icon.blue {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    }
    
    .feature-icon.teal {
        background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        color: var(--text-primary);
    }
    
    .feature-description {
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
        font-size: 0.9375rem;
        line-height: 1.6;
    }
    
    /* Section Cards */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .section-subtitle {
        color: var(--text-secondary);
        font-size: 0.9375rem;
        margin-bottom: 1.5rem;
    }
    
    .subsection-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    /* Stat Cards */
    .stat-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s;
    }
    
    .stat-card:hover {
        border-color: var(--primary-blue);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: var(--text-muted);
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.25rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.2s;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .metric-icon {
        width: 48px;
        height: 48px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    
    .metric-icon.blue {
        background: rgba(37, 99, 235, 0.1);
        color: var(--primary-blue);
    }
    
    .metric-icon.teal {
        background: rgba(20, 184, 166, 0.1);
        color: var(--accent-teal);
    }
    
    .metric-icon.purple {
        background: rgba(139, 92, 246, 0.1);
        color: var(--accent-purple);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Highlight Box */
    .highlight-box {
        background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(20, 184, 166, 0.2);
        margin: 1.5rem 0;
    }
    
    .highlight-label {
        font-size: 0.875rem;
        font-weight: 600;
        opacity: 0.95;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .highlight-value {
        font-size: 3.5rem;
        font-weight: 700;
    }
    
    /* Info Box */
    .info-box {
        background: var(--bg-light);
        border-left: 4px solid var(--primary-blue);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    
    .info-box-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .info-box-text {
        color: var(--text-secondary);
        font-size: 0.9375rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Feature Items */
    .feature-item {
        background: var(--bg-light);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 3px solid;
        margin-bottom: 1rem;
    }
    
    .feature-item.blue {
        border-left-color: var(--primary-blue);
    }
    
    .feature-item.teal {
        border-left-color: var(--accent-teal);
    }
    
    .feature-item.purple {
        border-left-color: var(--accent-purple);
    }
    
    .feature-item-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .feature-item-text {
        color: var(--text-secondary);
        font-size: 0.9375rem;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary-blue) !important;
        color: white !important;
        border: none !important;
        padding: 0.625rem 1.5rem !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.9375rem !important;
        transition: all 0.2s !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Login Page */
    .brand-header {
        text-align: center;
        margin: 3rem auto 2rem;
        max-width: 440px;
    }
    
    .login-brand-logo {
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    .brand-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    
    .brand-tagline {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Prediction Result */
    .prediction-result {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
    }
    
    .prediction-label {
        font-size: 0.875rem;
        font-weight: 600;
        opacity: 0.95;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .prediction-value {
        font-size: 4rem;
        font-weight: 700;
    }
    
    /* Hotel Cards */
    .hotel-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    
    .hotel-card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border-color: var(--accent-teal);
        transform: translateY(-2px);
    }
    
    /* Profile Items */
    .profile-item {
        background: var(--bg-light);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .profile-label {
        font-size: 0.8125rem;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 0.375rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .profile-value {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 2px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: var(--text-secondary);
        font-weight: 600;
        padding: 0.875rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--primary-blue);
        border-bottom: 2px solid var(--primary-blue);
    }
    
    /* Badge Styles */
    .badge {
        display: inline-block;
        padding: 0.375rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8125rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-primary {
        background: rgba(37, 99, 235, 0.1);
        color: var(--primary-blue);
    }
    
    .badge-success {
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
    }
    
    .badge-info {
        background: rgba(6, 182, 212, 0.1);
        color: #0891b2;
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.1);
        color: #d97706;
    }
    
    @media (max-width: 768px) {
        .welcome-title {
            font-size: 1.5rem;
        }
        .stat-value {
            font-size: 1.5rem;
        }
        .highlight-value {
            font-size: 2.5rem;
        }
        .prediction-value {
            font-size: 3rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Database connection ──────────────────────────────────────────────────────
@st.cache_resource
def get_db_connection():
    try:
        if "DATABASE_URL" in st.secrets:
            return psycopg2.connect(st.secrets["DATABASE_URL"])
    except Exception:
        pass
    return None

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        rec_models = load_recommendation_models(
            os.path.join(current_dir, 'models', 'recommendation')
        )

        recommendation_engine = HotelRecommendationEngine(
            user_hotel_matrix=rec_models['user_hotel_matrix'],
            user_similarity_df=rec_models['user_similarity'],
            hotel_similarity_df=rec_models['hotel_similarity'],
            hotel_features=rec_models['hotel_features'],
            complete_df=rec_models['complete_data'],
            users_df=rec_models['users_data']
        )

        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT user_code, name, age, gender, company, password
                FROM users
                WHERE user_code ~ '^[0-9]+$'
            """)
            db_rows = cur.fetchall()
            cur.close()

            if db_rows:
                db_df = pd.DataFrame(
                    db_rows, columns=["code", "name", "age", "gender", "company", "password"]
                )
                db_df["code"] = db_df["code"].astype(int)

                dataset_df = rec_models['users_data'].copy()
                dataset_df["code"] = dataset_df["code"].astype(int)

                merged = pd.concat(
                    [dataset_df, db_df.drop(columns=["password"])], ignore_index=True
                ).drop_duplicates(subset=["code"], keep="last")

                rec_models['users_data'] = merged
                recommendation_engine.users_df = merged

                # Load ALL users including User 0
                credentials = {}
                for _, r in dataset_df.iterrows():
                    user_code = int(r["code"])
                    user_name = str(r["name"])
                    credentials[user_code] = {"name": user_name, "password": "password123"}
                
                # Override with database passwords
                for code, name, age, gender, company, pwd in db_rows:
                    credentials[int(code)] = {"name": name, "password": pwd}
            else:
                # Load ALL users including User 0  
                credentials = {}
                for _, r in rec_models['users_data'].iterrows():
                    user_code = int(r['code'])
                    user_name = str(r['name'])
                    credentials[user_code] = {"name": user_name, "password": "password123"}
        else:
            # Load ALL users including User 0
            credentials = {}
            for _, r in rec_models['users_data'].iterrows():
                user_code = int(r['code'])
                user_name = str(r['name'])
                credentials[user_code] = {"name": user_name, "password": "password123"}

        # flight_df = pd.read_csv(os.path.join(current_dir, "data", "flights.csv"))
        #from database------------------------------------
        conn = get_db_connection()
        if conn:
           flight_df = pd.read_sql("SELECT * FROM flights", conn)
        else:
           st.error("Database not connected")
           st.stop()
           #-----------------------------------------
        available_users = sorted(rec_models['users_data']['code'].astype(int).unique().tolist())
        available_locations = sorted(rec_models['hotel_features']['location'].unique().tolist())

        return {
            'rec_models': rec_models,
            'recommendation_engine': recommendation_engine,
            'flight_df': flight_df,
            'available_users': available_users,
            'credentials': credentials,
            'available_locations': available_locations
        }

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

models_data = load_models()

# ── Authentication ────────────────────────────────────────────────────────────
def check_login(user_code, password):
    creds = models_data['credentials']
    if user_code in creds and creds[user_code]['password'] == password:
        return True, creds[user_code]['name']
    return False, None

def register_user(name, age, gender, company, password):
    try:
        conn = get_db_connection()
        if not conn:
            return False, None, "Database connection not available"

        cur = conn.cursor()
        cur.execute("""
            SELECT MAX(CAST(user_code AS INTEGER)) FROM users
            WHERE user_code ~ '^[0-9]+$'
        """)
        result = cur.fetchone()[0]
        new_code = (result + 1) if result else 1000

        cur.execute("""
            INSERT INTO users (user_code, name, age, gender, company, password)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (str(new_code), name, age, gender, company, password))
        conn.commit()
        cur.close()

        models_data['credentials'][new_code] = {'name': name, 'password': password}
        models_data['available_users'].append(new_code)
        models_data['available_users'].sort()

        new_row = pd.DataFrame({
            'code': [new_code], 'name': [name],
            'age': [age], 'gender': [gender], 'company': [company]
        })
        models_data['rec_models']['users_data'] = pd.concat(
            [models_data['rec_models']['users_data'], new_row], ignore_index=True
        )
        models_data['recommendation_engine'].users_df = models_data['rec_models']['users_data']

        return True, new_code, None
    except Exception as e:
        return False, None, str(e)

# ── Session state ─────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_code = None
    st.session_state.user_name = None
    st.session_state.page = "dashboard"

# ── FIXED Navbar ──────────────────────────────────────────────────────────────
def show_navbar(active_page="Dashboard"):
    if st.session_state.logged_in:
        user_initial = st.session_state.user_name[0].upper() if st.session_state.user_name else "U"
        
        st.markdown(f"""
        <div class="top-navbar">
            <div class="navbar-container">
                <div class="brand-section">
                    <div class="brand-icon">VA</div>
                    <span class="brand-name">Voyage Analytics</span>
                </div>
                <div class="user-section">
                    <div class="user-avatar">{user_initial}</div>
                    <span class="user-name">{st.session_state.user_name}</span>
                </div>
            </div>
        </div>
        <div class="main-content">
        """, unsafe_allow_html=True)
        
        # Navigation buttons with proper spacing
        st.markdown("<div style='margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            if st.button("Dashboard", use_container_width=True, type="primary" if active_page == "Dashboard" else "secondary"):
                st.session_state.page = "dashboard"
                st.rerun()
        with col2:
            if st.button("Flight Predictor", use_container_width=True, type="primary" if active_page == "Flight" else "secondary"):
                st.session_state.page = "flight"
                st.rerun()
        with col3:
            if st.button("Hotels", use_container_width=True, type="primary" if active_page == "Hotels" else "secondary"):
                st.session_state.page = "hotels"
                st.rerun()
        with col4:
            if st.button("Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user_code = None
                st.session_state.user_name = None
                st.session_state.page = "dashboard"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ── Login page ────────────────────────────────────────────────────────────────
def show_login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="brand-header">
            <div class="login-brand-logo">VA</div>
            <h1 class="brand-title">Voyage Analytics</h1>
            <p class="brand-tagline">Travel Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Sign In", "Register"])
    
    with tab1:
        st.markdown("### Existing User Login")
        
        # Get all available user codes and ensure User 0 is included
        available_codes = models_data['available_users']
        
        # Make sure User 0 is in the list if it's in credentials
        if 0 in models_data['credentials'] and 0 not in available_codes:
            available_codes = [0] + available_codes
        
        # Sort the list
        available_codes = sorted(available_codes)
        
        if available_codes:
            # Add info about number of users
            
            # Add custom CSS for scrollable selectbox
            st.markdown("""
            <style>
            /* Make selectbox dropdown scrollable with better styling */
            div[data-baseweb="select"] > div {
                max-height: 300px;
            }
            [role="listbox"] {
                max-height: 300px !important;
                overflow-y: auto !important;
                scrollbar-width: thin;
                scrollbar-color: #2563eb #e2e8f0;
            }
            [role="listbox"]::-webkit-scrollbar {
                width: 8px;
            }
            [role="listbox"]::-webkit-scrollbar-track {
                background: #e2e8f0;
                border-radius: 4px;
            }
            [role="listbox"]::-webkit-scrollbar-thumb {
                background: #2563eb;
                border-radius: 4px;
            }
            [role="listbox"]::-webkit-scrollbar-thumb:hover {
                background: #1e40af;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # User selection with scroll
            user_code = st.selectbox(
                "Select User Code", 
                options=available_codes,
                format_func=lambda x: f"User {x}" + (" ⭐" if x == 0 else ""),
                key="login_user_code",
                help="Type to search or scroll to find your user code"
            )
            
            # Show selected user info
            if user_code is not None and user_code in models_data['credentials']:
                st.caption(f"Selected: **{models_data['credentials'][user_code]['name']}**")
        else:
            st.error("No users available")
            user_code = None
        
        password = st.text_input("Password", type="password", key="login_pwd")
        st.caption("Default password for demo users: **password123**")
        
        if st.button("Sign In", type="primary", use_container_width=True):
            if user_code is not None and password:
                success, name = check_login(user_code, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_code = user_code
                    st.session_state.user_name = name
                    st.session_state.page = "dashboard"
                    st.success(f"Welcome, {name}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try password: password123")
            else:
                st.warning("Please fill in all fields")
    
    with tab2:
        st.markdown("### New User Registration")
        
        st.markdown("""
        <div class="info-box">
            <div class="info-box-title">Create Your Account</div>
            <div class="info-box-text">
                Get personalized hotel recommendations based on your preferences and travel profile.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        name = st.text_input("Full Name", key="reg_name")
        age = st.number_input("Age", min_value=18, max_value=100, value=25, key="reg_age")
        gender = st.selectbox("Gender", ["male", "female", "other"], key="reg_gender")
        company = st.text_input("Company (Optional)", value="Unknown", key="reg_company")
        password = st.text_input("Password (min 6 characters)", type="password", key="reg_pwd")
        password_confirm = st.text_input("Confirm Password", type="password", key="reg_pwd_confirm")
        
        if st.button("Create Account", type="primary", use_container_width=True):
            if not name or len(password) < 6:
                st.error("Please provide valid name and password (min 6 characters)")
            elif password != password_confirm:
                st.error("Passwords do not match")
            else:
                success, new_code, error = register_user(name, age, gender, company, password)
                if success:
                    st.success(f"Registration successful! Your user code is: **{new_code}**")
                    st.info("Please login with your new credentials")
                    st.balloons()
                else:
                    st.error(f"Registration failed: {error}")

# ── Dashboard home ────────────────────────────────────────────────────────────
def show_dashboard_home():
    show_navbar("Dashboard")
    
    st.markdown(f"""
    <div class="welcome-card">
        <h1 class="welcome-title">Welcome back, {st.session_state.user_name}</h1>
        <p class="welcome-subtitle">Your personalized travel analytics dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon blue"><i class="bi bi-airplane-engines"></i></div>
            <h3 class="feature-title">Flight Price Predictor</h3>
            <p class="feature-description">Get AI-powered price estimates for your next flight based on advanced machine learning algorithms trained on thousands of flight records.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Get Started", key="flight_btn", use_container_width=True):
            st.session_state.page = "flight"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon teal"><i class="bi bi-building-check"></i></div>
            <h3 class="feature-title">Hotel Recommendations</h3>
            <p class="feature-description">Discover personalized hotel recommendations tailored to your preferences, demographics, and travel patterns using collaborative filtering.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Hotels", key="hotel_btn", use_container_width=True):
            st.session_state.page = "hotels"
            st.rerun()
    
    # Platform Overview
    st.markdown("""
    <h2 class="section-title" style="margin-top: 2rem;">Platform Overview</h2>
    <p class="section-subtitle">Key metrics from our travel community</p>
    """, unsafe_allow_html=True)
    
    flight_df = models_data['flight_df']
    hotel_df = models_data['rec_models']['hotel_features']
    users_df = models_data['rec_models']['users_data']
    
    # User Demographics
    st.markdown('<h3 class="subsection-title">User Demographics</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Users</div>
            <div class="stat-value">{len(users_df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_age = users_df['age'].mean()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Average Age</div>
            <div class="stat-value">{avg_age:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        age_range = f"{users_df['age'].min()} - {users_df['age'].max()}"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Age Range</div>
            <div class="stat-value">{age_range}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Gender Distribution
    st.markdown('<div class="info-box"><h4 class="info-box-title">Gender Distribution</h4></div>', unsafe_allow_html=True)
    
    gender_counts = users_df['gender'].value_counts()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        male_count = gender_counts.get('male', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon blue"><i class="bi bi-person"></i></div>
            <div>
                <div class="metric-value">{male_count}</div>
                <div class="metric-label">Male</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        female_count = gender_counts.get('female', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon teal"><i class="bi bi-person"></i></div>
            <div>
                <div class="metric-value">{female_count}</div>
                <div class="metric-label">Female</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        other_count = len(users_df) - male_count - female_count
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon purple"><i class="bi bi-person"></i></div>
            <div>
                <div class="metric-value">{other_count}</div>
                <div class="metric-label">Other</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Hotel Bookings
    st.markdown('<h3 class="subsection-title">Hotel Bookings</h3>', unsafe_allow_html=True)
    
    complete_df = models_data['rec_models']['complete_data']
    total_bookings = len(complete_df)
    unique_hotels = hotel_df['hotel_name'].nunique()
    unique_locations = hotel_df['location'].nunique()
    avg_price = complete_df['price'].mean()
    
    # Check for days column
    if 'days' in complete_df.columns:
        avg_stay = complete_df['days'].mean()
    elif 'n' in complete_df.columns:
        avg_stay = complete_df['n'].mean()
    else:
        avg_stay = 2.5
    
    price_range = f"${complete_df['price'].min():.0f} - ${complete_df['price'].max():.0f}"
    
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    with col1:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Total Bookings</div><div class="stat-value">{total_bookings:,}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Unique Hotels</div><div class="stat-value">{unique_hotels}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Destinations</div><div class="stat-value">{unique_locations}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Price Range</div><div class="stat-value">{price_range}</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Average Price</div><div class="stat-value">${avg_price:.2f}</div></div>""", unsafe_allow_html=True)
    with col6:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Average Stay</div><div class="stat-value">{avg_stay:.1f} days</div></div>""", unsafe_allow_html=True)
    
    # Flight Analytics
    st.markdown('<h3 class="subsection-title">Flight Analytics</h3>', unsafe_allow_html=True)
    
    total_flights = len(flight_df)
    unique_routes = len(flight_df.groupby(['from', 'to']))
    avg_distance = flight_df['distance'].mean()
    avg_flight_price = flight_df['price'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Total Flights</div><div class="stat-value">{total_flights:,}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Unique Routes</div><div class="stat-value">{unique_routes}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="stat-card"><div class="stat-label">Avg Distance</div><div class="stat-value">{avg_distance:.0f} km</div></div>""", unsafe_allow_html=True)
    
    # Average Price Highlight
    st.markdown(f"""
    <div class="highlight-box">
        <div class="highlight-label"><i class="bi bi-currency-dollar"></i> Average Flight Price</div>
        <div class="highlight-value">${avg_flight_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # About Section
    st.markdown('<h2 class="section-title" style="margin-top: 2rem;">About Voyage Analytics</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <p class="info-box-text" style="margin-bottom: 2rem;">
        Voyage Analytics delivers intelligent travel insights powered by advanced machine learning. 
        Our platform analyzes data from <strong>{total_flights:,}+ flights</strong> and 
        <strong>{total_bookings:,}+ hotel bookings</strong> to provide accurate predictions 
        and personalized recommendations for modern travelers.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-item blue">
            <h4 class="feature-item-title"><i class="bi bi-cpu me-2"></i>AI-Powered Predictions</h4>
            <p class="feature-item-text">Advanced algorithms trained on millions of data points</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-item teal">
            <h4 class="feature-item-title"><i class="bi bi-person-hearts me-2"></i>Personalized Results</h4>
            <p class="feature-item-text">Recommendations tailored to your unique preferences</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-item purple">
            <h4 class="feature-item-title"><i class="bi bi-graph-up-arrow me-2"></i>Data-Driven Insights</h4>
            <p class="feature-item-text">Real-time analytics from 1,340+ active travelers</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

# ── Flight prediction ─────────────────────────────────────────────────────────
def show_flight_prediction():
    show_navbar("Flight")
    
    st.markdown("""
    <h1 class="section-title"><i class="bi bi-airplane-engines me-2"></i>Flight Price Predictor</h1>
    <p class="section-subtitle" style="margin-bottom: 2rem;">Get AI-powered price estimates for your next flight</p>
    """, unsafe_allow_html=True)
    
    flight_df = models_data['flight_df']
    
    col1, col2 = st.columns(2)
    
    with col1:
        from_city = st.selectbox("Origin", sorted(flight_df['from'].unique()), key="from_city")
        flight_type = st.selectbox("Flight Type", sorted(flight_df['flightType'].unique()), key="flight_type")
        travel_date = st.date_input("Travel Date", datetime.now(), key="travel_date")
    
    with col2:
        to_city = st.selectbox("Destination", sorted(flight_df['to'].unique()), key="to_city")
        agency = st.selectbox("Agency", sorted(flight_df['agency'].unique()), key="agency")
    
    time, distance = 0, 0
    if from_city != to_city:
        route = flight_df[(flight_df['from'] == from_city) & (flight_df['to'] == to_city)]
        if not route.empty:
            time = float(route.iloc[0]['time'])
            distance = float(route.iloc[0]['distance'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Flight Duration:** {time} hours")
            with col2:
                st.info(f"**Distance:** {distance} km")
        else:
            st.warning("Route not available in our database")
    else:
        st.error("Origin and destination cannot be the same")
    
    st.markdown("---")
    
    if st.button("Predict Price", type="primary", use_container_width=True):
        if from_city == to_city:
            st.error("Origin and destination cannot be the same")
        elif time == 0:
            st.error("Route not available. Please select different cities.")
        else:
            try:
                input_data = {
                    "from": from_city, "to": to_city,
                    "flightType": flight_type, "agency": agency,
                    "time": time, "distance": distance,
                    "date": travel_date.strftime("%Y-%m-%d")
                }
                price = predict_price(input_data)
                
                st.markdown(f"""
                <div class="prediction-result">
                    <div class="prediction-label"><i class="bi bi-currency-dollar"></i> Estimated Price</div>
                    <div class="prediction-value">${round(float(price), 2)}</div>
                    <div style="margin-top: 1rem; background: rgba(255,255,255,0.2); padding: 0.75rem 1.5rem; border-radius: 24px; display: inline-block;">
                        <i class="bi bi-cpu me-1"></i> Powered by Machine Learning
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
                
                with st.expander("View Input Details"):
                    st.json(input_data)
                
                st.markdown("""
                <div class="info-box">
                    <h4 class="info-box-title"><i class="bi bi-info-circle me-1"></i> About this prediction</h4>
                    <p class="info-box-text">
                        This price estimate is generated using advanced machine learning algorithms trained on historical flight data. 
                        Actual prices may vary based on availability, booking time, and market conditions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Route not available. Please try different options.")

# ── Hotel recommendations ─────────────────────────────────────────────────────
def show_hotel_recommendations():
    show_navbar("Hotels")
    
    st.markdown("""
    <h1 class="section-title"><i class="bi bi-building-check me-2"></i>Hotel Recommendations</h1>
    <p class="section-subtitle">Personalized hotel suggestions based on your preferences and booking history</p>
    <div style="margin-top: 1rem; margin-bottom: 2rem;">
        <span class="badge badge-primary"><i class="bi bi-building me-1"></i> 9 Unique Hotels</span>
        <span class="badge badge-success"><i class="bi bi-geo-alt me-1"></i> 9 Destinations</span>
        <span class="badge badge-info"><i class="bi bi-currency-dollar me-1"></i> $60 - $313</span>
        <span class="badge badge-warning"><i class="bi bi-graph-up me-1"></i> 40,552 Bookings</span>
    </div>
    """, unsafe_allow_html=True)
    
    engine = models_data['recommendation_engine']
    available_locations = models_data['available_locations']
    user_code = st.session_state.user_code
    
    # User Profile - IMPROVED VERSION WITH GRADIENT ICONS
    try:
        user_info = engine.get_user_info(user_code)
        user_stats = engine.get_user_stats(user_code)
        
        # Profile Card Container
        st.markdown("""
        <div class="section-card" style="background: linear-gradient(135deg, rgba(37, 99, 235, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%); border: 2px solid var(--primary-blue);">
            <h2 class="section-title"><i class="bi bi-person-circle me-2"></i>Your Profile</h2>
            <p class="section-subtitle" style="margin-bottom: 2rem;">Your personalized travel profile and preferences</p>
        """, unsafe_allow_html=True)
        
        # Profile Grid
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
        """, unsafe_allow_html=True)
        
        # Name Card
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); transition: all 0.3s; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                    <i class="bi bi-person-fill"></i>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Full Name</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: var(--text-primary);">{user_info["name"]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # User Code Card
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); transition: all 0.3s; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                    <i class="bi bi-hash"></i>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">User Code</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: var(--text-primary);">{user_info["code"]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gender Card
        gender_icon = "gender-male" if user_info["gender"].lower() == "male" else "gender-female" if user_info["gender"].lower() == "female" else "gender-ambiguous"
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); transition: all 0.3s; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                    <i class="bi bi-{gender_icon}"></i>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Gender</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: var(--text-primary);">{user_info["gender"].title()}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Age Card
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); transition: all 0.3s; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                    <i class="bi bi-cake2"></i>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Age</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: var(--text-primary);">{user_info["age"]} years</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Company Card
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid var(--border-color); transition: all 0.3s; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                    <i class="bi bi-briefcase-fill"></i>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Company</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: var(--text-primary);">{user_info["company"]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)  # Close profile grid
        
        # Booking Statistics
        st.markdown("""
        <div style="margin-top: 2.5rem;">
            <h3 style="font-size: 1.25rem; font-weight: 700; color: var(--text-primary); margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <i class="bi bi-bar-chart-fill"></i> Booking Statistics
            </h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
        """, unsafe_allow_html=True)
        
        # Total Bookings Card
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 16px; border: 1px solid var(--border-color); text-align: center; transition: all 0.3s; box-shadow: 0 4px 12px rgba(0,0,0,0.08); position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);"></div>
            <div style="width: 64px; height: 64px; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.25rem; color: white; font-size: 2rem; box-shadow: 0 8px 16px rgba(37, 99, 235, 0.3);">
                <i class="bi bi-calendar-check-fill"></i>
            </div>
            <div style="font-size: 0.875rem; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.08em;">Total Bookings</div>
            <div style="font-size: 3rem; font-weight: 700; color: var(--text-primary); line-height: 1; margin-bottom: 0.5rem;">{user_stats["total_bookings"]}</div>
            <div style="font-size: 0.875rem; color: var(--text-secondary); font-weight: 500;">Lifetime reservations</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Average Price Card
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 16px; border: 1px solid var(--border-color); text-align: center; transition: all 0.3s; box-shadow: 0 4px 12px rgba(0,0,0,0.08); position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);"></div>
            <div style="width: 64px; height: 64px; background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%); border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.25rem; color: white; font-size: 2rem; box-shadow: 0 8px 16px rgba(20, 184, 166, 0.3);">
                <i class="bi bi-currency-dollar"></i>
            </div>
            <div style="font-size: 0.875rem; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.08em;">Average Price</div>
            <div style="font-size: 3rem; font-weight: 700; color: var(--text-primary); line-height: 1; margin-bottom: 0.5rem;">${user_stats["avg_price"]:.2f}</div>
            <div style="font-size: 0.875rem; color: var(--text-secondary); font-weight: 500;">Per night</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Average Stay Card
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 16px; border: 1px solid var(--border-color); text-align: center; transition: all 0.3s; box-shadow: 0 4px 12px rgba(0,0,0,0.08); position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);"></div>
            <div style="width: 64px; height: 64px; background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%); border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 1.25rem; color: white; font-size: 2rem; box-shadow: 0 8px 16px rgba(139, 92, 246, 0.3);">
                <i class="bi bi-clock-history"></i>
            </div>
            <div style="font-size: 0.875rem; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.08em;">Average Stay</div>
            <div style="font-size: 3rem; font-weight: 700; color: var(--text-primary); line-height: 1; margin-bottom: 0.5rem;">{user_stats["avg_stay"]:.1f}</div>
            <div style="font-size: 0.875rem; color: var(--text-secondary); font-weight: 500;">Days per trip</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div></div></div>", unsafe_allow_html=True)  # Close stats grid, stats section, and profile card
        
    except Exception:
        pass
    
    # Search Form
    st.markdown('<h3 class="section-title" style="margin-top: 2rem;"><i class="bi bi-funnel me-2"></i>Search Filters</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title"><i class="bi bi-info-circle me-1"></i> Smart Recommendation System</div>
        <div class="info-box-text">
            Our AI analyzes your booking history, demographics, and similar users to personalize recommendations. 
            Each destination has 1 hotel, so use "All locations" with budget filters for best variety.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        destination = st.selectbox("Destination", ["All locations"] + available_locations, key="dest")
    with col2:
        budget_min = st.number_input("Min Budget ($)", min_value=60, max_value=313, value=60, key="budget_min")
    with col3:
        budget_max = st.number_input("Max Budget ($)", min_value=60, max_value=313, value=313, key="budget_max")
    with col4:
        n_recommendations = st.selectbox("Results", [3, 5, 9], index=1, key="n_rec")
    
    if st.button("Get AI Recommendations", type="primary", use_container_width=True):
        try:
            dest_filter = None if destination == "All locations" else destination
            
            engine.validate_inputs(
                user_code=user_code,
                destination=dest_filter,
                budget_min=budget_min,
                budget_max=budget_max
            )
            
            with st.spinner("Generating personalized recommendations..."):
                recommendations = engine.hybrid_recommendations(
                    user_code=user_code,
                    destination=dest_filter,
                    budget_min=budget_min,
                    budget_max=budget_max,
                    n_recommendations=n_recommendations,
                    apply_diversity=True,
                    use_adaptive_weights=True,
                    debug=False
                )
            
            if recommendations:
                
                st.markdown(f"""
                <h2 class="section-title" style="margin-top: 2rem;"><i class="bi bi-stars me-2"></i>AI-Powered Recommendations</h2>
                <p class="section-subtitle">Showing {len(recommendations)} personalized recommendations</p>
                """, unsafe_allow_html=True)
                
                for idx, hotel in enumerate(recommendations, 1):
                    st.markdown(f"""
                    <div class="hotel-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem; flex-wrap: wrap; gap: 1rem;">
                            <div>
                                <h3 style="font-size: 1.25rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem;">
                                    {idx}. <i class="bi bi-building me-2"></i>{hotel['hotel_name']}
                                </h3>
                                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                                    <span class="badge badge-primary"><i class="bi bi-geo-alt-fill me-1"></i> {hotel['location']}</span>
                                    <span class="badge badge-warning"><i class="bi bi-star-fill me-1"></i> {hotel['methods_used']}</span>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Match Score</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: #14b8a6;">{hotel['recommendation_score'] * 100:.1f}%</div>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem;">
                            <div style="background: #f8fafc; padding: 0.875rem; border-radius: 8px; text-align: center;">
                                <span style="display: block; font-size: 0.8125rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.375rem;">
                                    <i class="bi bi-currency-dollar"></i> AVG PRICE
                                </span>
                                <span style="font-size: 1.125rem; font-weight: 700; color: #1e293b;">${hotel['avg_price']:.2f}</span>
                                <small style="display: block; color: #64748b;">per night</small>
                            </div>
                            <div style="background: #f8fafc; padding: 0.875rem; border-radius: 8px; text-align: center;">
                                <span style="display: block; font-size: 0.8125rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.375rem;">
                                    <i class="bi bi-calendar-check"></i> AVG STAY
                                </span>
                                <span style="font-size: 1.125rem; font-weight: 700; color: #1e293b;">{hotel['avg_stay']:.1f}</span>
                                <small style="display: block; color: #64748b;">days</small>
                            </div>
                            <div style="background: #f8fafc; padding: 0.875rem; border-radius: 8px; text-align: center;">
                                <span style="display: block; font-size: 0.8125rem; color: #94a3b8; font-weight: 600; margin-bottom: 0.375rem;">
                                    <i class="bi bi-people"></i> POPULARITY
                                </span>
                                <span style="font-size: 1.125rem; font-weight: 700; color: #1e293b;">{hotel['popularity']}</span>
                                <small style="display: block; color: #64748b;">bookings</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.warning("No hotels found matching your criteria. Try adjusting your filters.")
        
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        show_login_page()
    else:
        if st.session_state.page == "dashboard":
            show_dashboard_home()
        elif st.session_state.page == "flight":
            show_flight_prediction()
        elif st.session_state.page == "hotels":
            show_hotel_recommendations()
        else:
            show_dashboard_home()

if __name__ == "__main__":
    main()
