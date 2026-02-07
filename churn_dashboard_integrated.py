import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deriv | Partner Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- CUSTOM BRANDING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    * { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0E0E0E; color: white; }
    .stMetric { background-color: #151717; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .brand-header {
        background: linear-gradient(135deg, #151717 0%, #0E0E0E 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid #333;
    }
    .risk-card {
        padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- FILE PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'large_partner_churn_dataset_24000.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'churn_risk_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'feature_encoders.pkl')

# --- AUTOMATIC MODEL GENERATOR (Fixes Model Load Error) ---
def train_backup_model(df):
    """Trains a model on the fly if .pkl files are missing"""
    st.info("🔄 Initializing predictive engine for the first time...")
    
    # Preprocessing
    df = df.copy()
    df['join_date'] = pd.to_datetime(df['join_date'], dayfirst=True)
    df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True)
    df['tenure_weeks'] = ((df['week_start_date'] - df['join_date']).dt.days / 7).fillna(0).astype(int)
    
    # Create target (1 for At Risk/Churning, 0 for Healthy)
    df['target'] = df['behavior'].apply(lambda x: 1 if x in ['at_risk', 'churning'] else 0)
    
    encoders = {}
    for col in ['region', 'partner_type', 'priority']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    features = ['logins', 'referrals', 'earnings', 'unresolved_tickets', 'region', 
                'partner_type', 'priority', 'days_since_last_outreach', 
                'payout_delay_days', 'commission_dispute_count', 'competitor_mention_flag', 'tenure_weeks']
    
    X = df[features].fillna(0)
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Save files so this doesn't run again
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    return model, encoders

# --- DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Dataset not found at {DATA_PATH}. Please upload the CSV to GitHub.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True)
    df['join_date'] = pd.to_datetime(df['join_date'], dayfirst=True)
    df['tenure_weeks'] = ((df['week_start_date'] - df['join_date']).dt.days / 7).fillna(0).astype(int)
    return df

@st.cache_resource
def load_models(df):
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        try:
            return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)
        except:
            return train_backup_model(df)
    else:
        return train_backup_model(df)

# Initialize
df_raw = load_data()
churn_model, encoders = load_models(df_raw)

# --- PREDICTION LOGIC ---
@st.cache_data
def get_predictions(_df, _model, _encoders):
    # Get latest snapshot per partner
    latest = _df.sort_values('week_number').groupby('partner_id').tail(1).copy()
    
    # Encode
    encoded_df = latest.copy()
    for col, le in _encoders.items():
        # Handle unseen categories gracefully
        encoded_df[col] = encoded_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            
    features = ['logins', 'referrals', 'earnings', 'unresolved_tickets', 'region', 
                'partner_type', 'priority', 'days_since_last_outreach', 
                'payout_delay_days', 'commission_dispute_count', 'competitor_mention_flag', 'tenure_weeks']
    
    probs = _model.predict_proba(encoded_df[features].fillna(0))[:, 1]
    latest['Risk Score'] = probs
    latest['Status'] = latest['Risk Score'].apply(lambda x: '⚠️ At Risk' if x >= 0.5 else '✅ Healthy')
    return latest

df_all = get_predictions(df_raw, churn_model, encoders)

# --- NAVIGATION ---
st.sidebar.title("Deriv Partner Risk")
page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🔍 Partner Lookup"])

# --- DASHBOARD PAGE ---
if page == "📊 Dashboard":
    st.markdown('<div class="brand-header"><h1>Partner Churn Analytics</h1></div>', unsafe_allow_html=True)
    
    # Filter Row
    c1, c2, c3 = st.columns(3)
    with c1:
        reg_list = sorted(df_all['region'].unique().tolist())
        reg_filter = st.multiselect("Region", options=reg_list, default=reg_list)
    with c2:
        pri_list = sorted(df_all['priority'].unique().tolist())
        pri_filter = st.multiselect("Tier", options=pri_list, default=pri_list)
    with c3:
        stat_list = sorted(df_all['Status'].unique().tolist())
        stat_filter = st.multiselect("Status", options=stat_list, default=stat_list)

    filtered_df = df_all[
        (df_all['region'].isin(reg_filter)) & 
        (df_all['priority'].isin(pri_filter)) &
        (df_all['Status'].isin(status_filter))
    ]

    # KPIs
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Partners", len(filtered_df))
    m2.metric("At Risk", len(filtered_df[filtered_df['Risk Score'] >= 0.5]))
    m3.metric("Avg Risk Score", f"{filtered_df['Risk Score'].mean():.1%}")

    # Data Table
    st.subheader("Partner Health Registry")
    
    # Check for streamlit version support for progress bars
    if hasattr(st, "column_config"):
        st.dataframe(
            filtered_df[['partner_id', 'Status', 'Risk Score', 'region', 'earnings', 'churn_driver', 'recommended_action']].fillna("N/A"),
            column_config={
                "Risk Score": st.column_config.ProgressColumn("Churn Risk", min_value=0, max_value=1, format="%.2f"),
                "earnings": st.column_config.NumberColumn("Earnings (Wk)", format="$%d"),
            },
            hide_index=True, use_container_width=True
        )
    else:
        st.dataframe(filtered_df[['partner_id', 'Status', 'Risk Score', 'region', 'earnings']].fillna("N/A"))

# --- LOOKUP PAGE ---
elif page == "🔍 Partner Lookup":
    st.markdown('<div class="brand-header"><h1>Individual Partner Audit</h1></div>', unsafe_allow_html=True)
    search_id = st.text_input("Enter Partner ID (e.g., P0001):").strip().upper()
    
    if search_id:
        match = df_all[df_all['partner_id'] == search_id]
        if not match.empty:
            p = match.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                color = "#FF444F" if p['Risk Score'] >= 0.5 else "#4BB4B3"
                st.markdown(f'<div class="risk-card" style="background:{color}"><h2>{p["Risk Score"]:.1%}</h2><p>Risk Score</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="risk-card" style="background:#222"><h3>{p["churn_driver"]}</h3><p>Main Driver</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="risk-card" style="background:#222"><h3>{p["recommended_action"]}</h3><p>Next Steps</p></div>', unsafe_allow_html=True)
            
            st.markdown("### Weekly Performance")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Logins", int(p['logins']))
            d2.metric("Referrals", int(p['referrals']))
            d3.metric("Payout Delay", f"{int(p['payout_delay_days'])} days")
            d4.metric("Disputes", int(p['commission_dispute_count']))
        else:
            st.error(f"Partner ID '{search_id}' not found.")
