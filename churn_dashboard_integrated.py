import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deriv | Partner Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BRANDING & CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    
    * { font-family: 'IBM Plex Sans', sans-serif; }
    :root {
        --deriv-red: #FF444F;
        --deriv-black: #0E0E0E;
        --deriv-dark: #151717;
    }
    .main { background-color: #0E0E0E; color: white; }
    .main h1, .main h2, .main h3 { color: white !important; border-bottom: 2px solid var(--deriv-red); padding-bottom: 10px; }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] { color: white !important; }
    [data-testid="stMetricLabel"] { color: #999999 !important; }

    /* Brand Header */
    .brand-header {
        background: linear-gradient(135deg, var(--deriv-dark) 0%, var(--deriv-black) 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #333;
    }

    /* Risk Card */
    .risk-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- FILE PATH SETUP ---
# This ensures it works on GitHub/Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        churn_model = joblib.load(get_path('churn_risk_model.pkl'))
        encoders = joblib.load(get_path('feature_encoders.pkl'))
        lookup_dict = joblib.load(get_path('partner_lookup_model.pkl'))
        return churn_model, encoders, lookup_dict
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}. Ensure all .pkl files are in the GitHub root.")
        return None, None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(get_path('large_partner_churn_dataset_24000.csv'))
        df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True)
        df['join_date'] = pd.to_datetime(df['join_date'], dayfirst=True)
        df['tenure_weeks'] = ((df['week_start_date'] - df['join_date']).dt.days / 7).fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"❌ Data Load Error: {e}. Ensure the .csv file is in the GitHub root.")
        return None

# Initialize App
churn_model, encoders, lookup_dict = load_models()
df_raw = load_data()

if churn_model is None or df_raw is None:
    st.warning("Waiting for data and models to be uploaded to GitHub...")
    st.stop()

# --- PREDICTION LOGIC ---
def get_predictions(df, model, encoders):
    # Get latest snapshot per partner
    latest = df.sort_values('week_number').groupby('partner_id').tail(1).copy()
    
    # Store clean names for display
    latest['Region_Display'] = latest['region']
    latest['Type_Display'] = latest['partner_type']
    latest['Priority_Display'] = latest['priority']
    
    # Encode for model
    for col in ['region', 'partner_type', 'priority']:
        if col in encoders:
            latest[col] = encoders[col].transform(latest[col])
            
    features = ['logins', 'referrals', 'earnings', 'unresolved_tickets', 'region', 
                'partner_type', 'priority', 'days_since_last_outreach', 
                'payout_delay_days', 'commission_dispute_count', 'competitor_mention_flag', 'tenure_weeks']
    
    probs = model.predict_proba(latest[features])[:, 1]
    latest['Risk Score'] = probs
    latest['Status'] = latest['Risk Score'].apply(lambda x: '⚠️ At Risk' if x >= 0.5 else '✅ Healthy')
    return latest

df_all = get_predictions(df_raw, churn_model, encoders)

# --- SIDEBAR ---
st.sidebar.image("https://deriv.com/static/deriv-logo-9759da0.svg", width=150) # Generic Deriv Logo URL
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dashboard", "🔍 Partner Lookup"])

# Stats
st.sidebar.markdown("---")
at_risk_count = len(df_all[df_all['Risk Score'] >= 0.5])
st.sidebar.metric("Total Partners", len(df_all))
st.sidebar.metric("At Risk", at_risk_count, delta=f"{(at_risk_count/len(df_all)):.1%}", delta_color="inverse")

# --- PAGE 1: DASHBOARD ---
if page == "📊 Dashboard":
    st.markdown('<div class="brand-header"><h1>Partner Churn Analytics</h1><p>Real-time risk monitoring & retention strategies</p></div>', unsafe_allow_html=True)
    
    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        reg_filter = st.multiselect("Region", options=df_all['Region_Display'].unique(), default=df_all['Region_Display'].unique())
    with c2:
        pri_filter = st.multiselect("Priority", options=df_all['Priority_Display'].unique(), default=df_all['Priority_Display'].unique())
    with c3:
        status_filter = st.multiselect("Status", options=df_all['Status'].unique(), default=df_all['Status'].unique())

    filtered_df = df_all[
        (df_all['Region_Display'].isin(reg_filter)) & 
        (df_all['Priority_Display'].isin(pri_filter)) &
        (df_all['Status'].isin(status_filter))
    ]

    # CLEAN TABLE DISPLAY
    st.subheader("Partner Risk Registry")
    
    # Create a display-specific dataframe to avoid "disrupting" the UI
    display_df = filtered_df[[
        'partner_id', 'Status', 'Risk Score', 'Region_Display', 'Priority_Display', 
        'earnings', 'churn_driver', 'recommended_action'
    ]].copy()

    # Fill NaNs to prevent table breaks
    display_df = display_df.fillna("N/A")

    st.dataframe(
        display_df,
        column_config={
            "partner_id": "Partner ID",
            "Status": "Health Status",
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Probability",
                help="Prediction from Random Forest Model",
                min_value=0, max_value=1, format="%.2f"
            ),
            "earnings": st.column_config.NumberColumn("Earnings", format="$%d"),
            "Region_Display": "Region",
            "Priority_Display": "Tier",
            "churn_driver": "Primary Churn Driver",
            "recommended_action": "Retention Action"
        },
        hide_index=True,
        use_container_width=True,
        height=500
    )

# --- PAGE 2: LOOKUP ---
elif page == "🔍 Partner Lookup":
    st.markdown('<div class="brand-header"><h1>Individual Partner Audit</h1></div>', unsafe_allow_html=True)
    
    search_id = st.text_input("Enter Partner ID (e.g., P0001):").upper()
    
    if search_id:
        if search_id in lookup_dict:
            p_data = df_all[df_all['partner_id'] == search_id].iloc[0]
            
            # Risk Hero Section
            col1, col2, col3 = st.columns(3)
            with col1:
                color = "#FF444F" if p_data['Risk Score'] >= 0.5 else "#4BB4B3"
                st.markdown(f'<div class="risk-card" style="background:{color}"><h3>Risk Score</h3><h1>{p_data["Risk Score"]:.1%}</h1><p>{p_data["Status"]}</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="risk-card" style="background:#333"><h3>Main Driver</h3><h2 style="font-size:1.2rem">{p_data["churn_driver"]}</h2></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="risk-card" style="background:#333"><h3>Recommended Action</h3><h2 style="font-size:1rem">{p_data["recommended_action"]}</h2></div>', unsafe_allow_html=True)
            
            # Details Grid
            st.markdown("### Behavioral Profile")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Logins (Weekly)", p_data['logins'])
            m2.metric("Referrals", p_data['referrals'])
            m3.metric("Unresolved Tickets", p_data['unresolved_tickets'])
            m4.metric("Payout Delay", f"{p_data['payout_delay_days']} Days")
            
            st.markdown("---")
            st.info(f"**Tenure:** {p_data['tenure_weeks']} weeks | **Region:** {p_data['Region_Display']} | **Tier:** {p_data['Priority_Display']}")
            
        else:
            st.error("Partner ID not found in database.")
