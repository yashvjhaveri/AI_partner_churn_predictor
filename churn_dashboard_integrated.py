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
    
    [data-testid="stMetricValue"] { color: white !important; }
    [data-testid="stMetricLabel"] { color: #999999 !important; }

    .brand-header {
        background: linear-gradient(135deg, var(--deriv-dark) 0%, var(--deriv-black) 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #333;
    }

    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- FILE PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- DATA & MODEL LOADING ---
@st.cache_resource
def load_models():
    try:
        churn_model = joblib.load(get_path('churn_risk_model.pkl'))
        encoders = joblib.load(get_path('feature_encoders.pkl'))
        # We'll use the dataframe for lookup to stay consistent
        return churn_model, encoders
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(get_path('large_partner_churn_dataset_24000.csv'))
        df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True)
        df['join_date'] = pd.to_datetime(df['join_date'], dayfirst=True)
        df['tenure_weeks'] = ((df['week_start_date'] - df['join_date']).dt.days / 7).fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"❌ Data Load Error: {e}")
        return None

churn_model, encoders = load_models()
df_raw = load_data()

if churn_model is None or df_raw is None:
    st.stop()

# --- PREDICTION LOGIC ---
@st.cache_data
def get_predictions(_df, _model, _encoders):
    latest = _df.sort_values('week_number').groupby('partner_id').tail(1).copy()
    
    # Store clean names for display
    latest['Region_Display'] = latest['region'].fillna('Unknown')
    latest['Type_Display'] = latest['partner_type'].fillna('Unknown')
    latest['Priority_Display'] = latest['priority'].fillna('Unknown')
    
    # Create copy for encoding
    encoded_df = latest.copy()
    for col in ['region', 'partner_type', 'priority']:
        if col in _encoders:
            encoded_df[col] = _encoders[col].transform(latest[col].astype(str))
            
    features = ['logins', 'referrals', 'earnings', 'unresolved_tickets', 'region', 
                'partner_type', 'priority', 'days_since_last_outreach', 
                'payout_delay_days', 'commission_dispute_count', 'competitor_mention_flag', 'tenure_weeks']
    
    probs = _model.predict_proba(encoded_df[features])[:, 1]
    latest['Risk Score'] = probs
    latest['Status'] = latest['Risk Score'].apply(lambda x: '⚠️ At Risk' if x >= 0.5 else '✅ Healthy')
    return latest

df_all = get_predictions(df_raw, churn_model, encoders)

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dashboard", "🔍 Partner Lookup"])

# --- PAGE 1: DASHBOARD ---
if page == "📊 Dashboard":
    st.markdown('<div class="brand-header"><h1>Partner Churn Analytics</h1></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        regions = sorted(df_all['Region_Display'].unique().tolist())
        reg_filter = st.multiselect("Region", options=regions, default=regions)
    with c2:
        tiers = sorted(df_all['Priority_Display'].unique().tolist())
        pri_filter = st.multiselect("Priority", options=tiers, default=tiers)
    with c3:
        statuses = sorted(df_all['Status'].unique().tolist())
        status_filter = st.multiselect("Status", options=statuses, default=statuses)

    filtered_df = df_all[
        (df_all['Region_Display'].isin(reg_filter)) & 
        (df_all['Priority_Display'].isin(pri_filter)) &
        (df_all['Status'].isin(status_filter))
    ]

    st.subheader("Partner Risk Registry")
    
    # Use st.column_config without the error
    st.dataframe(
        filtered_df[[
            'partner_id', 'Status', 'Risk Score', 'Region_Display', 
            'earnings', 'churn_driver', 'recommended_action'
        ]].fillna("N/A"),
        column_config={
            "partner_id": "ID",
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Probability", 
                min_value=0, 
                max_value=1, 
                format="%.2f"
            ),
            "earnings": st.column_config.NumberColumn("Earnings", format="$%d"),
            "Region_Display": "Region"
        },
        hide_index=True,
        use_container_width=True
    )

# --- PAGE 2: LOOKUP ---
elif page == "🔍 Partner Lookup":
    st.markdown('<div class="brand-header"><h1>Individual Partner Audit</h1></div>', unsafe_allow_html=True)
    search_id = st.text_input("Enter Partner ID (e.g., P0001):").strip().upper()
    
    if search_id:
        # Better lookup logic
        match = df_all[df_all['partner_id'] == search_id]
        
        if not match.empty:
            p_data = match.iloc[0]
            
            # Risk Hero Section
            col1, col2, col3 = st.columns(3)
            with col1:
                bg_color = "#FF444F" if p_data['Risk Score'] >= 0.5 else "#4BB4B3"
                st.markdown(f"""
                    <div class="risk-card" style="background-color: {bg_color};">
                        <small>RISK SCORE</small>
                        <h1 style="border:none; margin:0; padding:0;">{p_data['Risk Score']:.1%}</h1>
                        <strong>{p_data['Status']}</strong>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="risk-card" style="background-color: #222;">
                        <small>PRIMARY DRIVER</small>
                        <h3 style="border:none; margin:10px 0;">{p_data['churn_driver']}</h3>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="risk-card" style="background-color: #222;">
                        <small>RECOMMENDED ACTION</small>
                        <h3 style="border:none; margin:10px 0;">{p_data['recommended_action']}</h3>
                    </div>
                """, unsafe_allow_html=True)
            
            # Detailed Metrics
            st.markdown("### Behavioral Stats")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Logins", int(p_data['logins']))
            m2.metric("Referrals", int(p_data['referrals']))
            m3.metric("Earnings", f"${p_data['earnings']:,.2f}")
            m4.metric("Unresolved Tickets", int(p_data['unresolved_tickets']))
            
            st.markdown("---")
            st.write(f"**Tenure:** {p_data['tenure_weeks']} weeks | **Region:** {p_data['Region_Display']} | **Tier:** {p_data['Priority_Display']}")
        else:
            st.error(f"Partner ID '{search_id}' not found.")
