import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Partner Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching Deriv's theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    /* Deriv Brand Colors */
    :root {
        --deriv-red: #FF444F;
        --deriv-red-dark: #D43939;
        --deriv-black: #0E0E0E;
        --deriv-dark: #151717;
        --deriv-blue: #377CFC;
        --deriv-green: #4BB4B3;
        --deriv-success: #4BB4B3;
        --deriv-warning: #FFB800;
        --deriv-danger: #FF444F;
        --deriv-gray-light: #F2F3F4;
        --deriv-gray: #999999;
        --deriv-gray-dark: #333333;
    }
    
    /* Main Container Background */
    .main {
        background-color: #0E0E0E;
    }
    
    /* Make all main content text white by default */
    .main * {
        color: #FFFFFF;
    }
    
    /* Ensure section headers are white */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #FFFFFF !important;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: var(--deriv-black);
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 1rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: var(--deriv-gray);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--deriv-dark);
    }
    
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, var(--deriv-gray-light) 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid var(--deriv-red);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .healthy {
        border-left-color: var(--deriv-success);
    }
    
    .at-risk {
        border-left-color: var(--deriv-danger);
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: var(--deriv-red);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: var(--deriv-red-dark);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(255, 68, 79, 0.3);
    }
    
    /* DataFrame Styling */
    .stDataFrame {
        border: 1px solid var(--deriv-gray-light);
        border-radius: 8px;
    }
    
    /* Section Headers */
    h3 {
        color: #FFFFFF !important;
        font-weight: 600;
        border-bottom: 2px solid var(--deriv-red);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Filters Section */
    .stMultiSelect > div > div {
        background-color: #FFFFFF;
        border: 1px solid var(--deriv-gray-light);
        border-radius: 4px;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--deriv-gray-light);
        margin: 2rem 0;
    }
    
    /* Risk Score Card */
    .risk-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    
    .risk-card:hover {
        transform: scale(1.02);
    }
    
    /* Metric Values */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700;
    }
    
    /* Hide metric delta (arrows and change values) */
    [data-testid="stMetricDelta"] {
        display: none !important;
    }
    
    /* Metric Label */
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* Partner Details - ensure all metric content is white */
    .element-container [data-testid="stMetric"] {
        background-color: transparent;
    }
    
    .element-container [data-testid="stMetric"] * {
        color: #FFFFFF !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        border: 2px solid var(--deriv-gray-light);
        border-radius: 4px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--deriv-red);
        box-shadow: 0 0 0 2px rgba(255, 68, 79, 0.1);
    }
    
    /* Info/Warning/Error Boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Make alert text white */
    .stAlert * {
        color: #FFFFFF !important;
    }
    
    /* Info boxes background */
    [data-baseweb="notification"] {
        background-color: rgba(55, 124, 252, 0.2) !important;
    }
    
    /* Warning boxes background */
    .stWarning {
        background-color: rgba(255, 184, 0, 0.2) !important;
    }
    
    /* Error boxes background */
    .stError {
        background-color: rgba(255, 68, 79, 0.2) !important;
    }
    
    /* Success boxes background */
    .stSuccess {
        background-color: rgba(75, 180, 179, 0.2) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: var(--deriv-gray);
        padding: 2rem;
        border-top: 1px solid var(--deriv-gray-light);
        margin-top: 3rem;
    }
    
    /* Logo/Brand Header */
    .brand-header {
        background: linear-gradient(135deg, var(--deriv-dark) 0%, var(--deriv-black) 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .brand-header h1 {
        margin: 0;
        color: white;
        border: none;
    }
    
    .brand-header p {
        margin: 0.5rem 0 0 0;
        color: var(--deriv-gray);
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper to get the absolute path for files in the repository
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models and data
@st.cache_resource
def load_models():
    """Load the trained models and encoders using relative paths"""
    try:
        # Load the churn risk model files from the same folder as this script
        churn_model = joblib.load(os.path.join(BASE_DIR, 'churn_risk_model.pkl'))
        encoders = joblib.load(os.path.join(BASE_DIR, 'feature_encoders.pkl'))
        lookup_dict = joblib.load(os.path.join(BASE_DIR, 'partner_lookup_model.pkl'))
        return churn_model, encoders, lookup_dict
    except Exception as e:
        st.error(f"Error loading model files: {e}. Ensure .pkl files are uploaded to your GitHub repository.")
        return None, None, None

@st.cache_data
def load_data():
    """Load and process the partner data using relative paths"""
    try:
        # Load the raw dataset from the same folder as this script
        data_file = os.path.join(BASE_DIR, 'large_partner_churn_dataset_24000.csv')
        df = pd.read_csv(data_file)
        
        # Convert dates
        df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True)
        df['join_date'] = pd.to_datetime(df['join_date'], dayfirst=True)
        
        # Calculate tenure
        df['tenure_weeks'] = ((df['week_start_date'] - df['join_date']).dt.days / 7).astype(int)
        
        # Get latest record for each partner
        latest_data = df.sort_values('week_number').groupby('partner_id').tail(1).copy()
        
        return df, latest_data
    except Exception as e:
        st.error(f"Error loading data: {e}. Ensure the .csv file is uploaded to your GitHub repository.")
        return None, None

def predict_churn_risk(partner_data, model, encoders):
    """Predict churn risk for a partner"""
    features = ['logins', 'referrals', 'earnings', 'unresolved_tickets', 'region', 
                'partner_type', 'priority', 'days_since_last_outreach', 
                'payout_delay_days', 'commission_dispute_count', 'competitor_mention_flag', 'tenure_weeks']
    
    # Prepare feature vector
    X = partner_data[features].values.reshape(1, -1)
    
    # Get probability
    risk_prob = model.predict_proba(X)[0][1]
    
    return risk_prob

def decode_categorical(value, encoder):
    """Decode encoded categorical value back to original"""
    try:
        return encoder.inverse_transform([int(value)])[0]
    except:
        return value

def get_all_partners_with_predictions(df, model, encoders):
    """Get all partners with their churn predictions"""
    # Get latest record for each partner
    latest_data = df.sort_values('week_number').groupby('partner_id').tail(1).copy()
    
    # Store original categorical values for display BEFORE encoding
    latest_data['region_name'] = latest_data['region'].copy()
    latest_data['partner_type_name'] = latest_data['partner_type'].copy()
    latest_data['priority_name'] = latest_data['priority'].copy()
    
    # Encode categorical columns for prediction
    for col in ['region', 'partner_type', 'priority']:
        if col in encoders:
            latest_data[col] = encoders[col].transform(latest_data[col])
    
    # Feature columns for prediction
    features = ['logins', 'referrals', 'earnings', 'unresolved_tickets', 'region', 
                'partner_type', 'priority', 'days_since_last_outreach', 
                'payout_delay_days', 'commission_dispute_count', 'competitor_mention_flag', 'tenure_weeks']
    
    # Get predictions
    X = latest_data[features].values
    risk_probs = model.predict_proba(X)[:, 1]
    latest_data['churn_risk_score'] = risk_probs
    
    # Classify behavior based on risk score
    latest_data['predicted_behavior'] = latest_data['churn_risk_score'].apply(
        lambda x: 'At Risk' if x >= 0.5 else 'Healthy'
    )
    
    return latest_data

# Load models and data
churn_model, encoders, lookup_dict = load_models()
df_raw, df_latest = load_data()

# Check if models and data loaded successfully
if churn_model is None or df_raw is None:
    st.stop()

# Get predictions for all partners
df_all_partners = get_all_partners_with_predictions(df_raw, churn_model, encoders)

# Sidebar for navigation
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h2 style="color: #FF444F; margin: 0; font-weight: 700;">Deriv</h2>
        <p style="color: #999999; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Churn Analytics</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Select Feature", ["📊 Dashboard", "🔍 Partner Lookup"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.metric("Total Partners", len(df_all_partners))
st.sidebar.metric("At Risk Partners", len(df_all_partners[df_all_partners['predicted_behavior'] == 'At Risk']))
st.sidebar.metric("Healthy Partners", len(df_all_partners[df_all_partners['predicted_behavior'] == 'Healthy']))
st.sidebar.metric("Churn Risk Rate", f"{len(df_all_partners[df_all_partners['predicted_behavior'] == 'At Risk']) / len(df_all_partners) * 100:.1f}%")

# Main content
if page == "📊 Dashboard":
    # Deriv-style Brand Header
    st.markdown("""
        <div class="brand-header">
            <h1>Partner Churn Prediction</h1>
            <p>Advanced analytics and insights for partner retention</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Filters section
    st.markdown("### 🔎 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority_filter = st.multiselect(
            "Priority",
            options=sorted(df_all_partners['priority_name'].unique()),
            default=sorted(df_all_partners['priority_name'].unique())
        )
    
    with col2:
        churn_driver_filter = st.multiselect(
            "Churn Driver",
            options=sorted(df_all_partners['churn_driver'].unique()),
            default=sorted(df_all_partners['churn_driver'].unique())
        )
    
    with col3:
        action_filter = st.multiselect(
            "Recommended Action",
            options=sorted(df_all_partners['recommended_action'].unique()),
            default=sorted(df_all_partners['recommended_action'].unique())
        )
    
    # Apply filters
    filtered_df = df_all_partners[
        (df_all_partners['priority_name'].isin(priority_filter)) &
        (df_all_partners['churn_driver'].isin(churn_driver_filter)) &
        (df_all_partners['recommended_action'].isin(action_filter))
    ]
    
    st.markdown("---")
    
    # Layout: Main table on left (65%), At-Risk table on right (35%)
    col_main, col_side = st.columns([65, 35])
    
    # Define consistent table height
    TABLE_HEIGHT = 600
    
    with col_main:
        st.markdown("### 📋 All Partners Overview")
        st.markdown(f"**Total Records:** {len(filtered_df)}")
        
        # Prepare display dataframe
        display_df = filtered_df[[
            'partner_id', 'week_number', 'week_start_date', 'logins', 'referrals', 
            'earnings', 'unresolved_tickets', 'join_date',
            'region_name', 'partner_type_name', 'priority_name',
            'days_since_last_outreach', 'payout_delay_days', 
            'commission_dispute_count', 'competitor_mention_flag',
            'predicted_behavior', 'churn_risk_score', 'churn_driver', 'recommended_action'
        ]].copy()
        
        # Format the display
        display_df['week_start_date'] = display_df['week_start_date'].dt.strftime('%Y-%m-%d')
        display_df['join_date'] = display_df['join_date'].dt.strftime('%Y-%m-%d')
        display_df['churn_risk_score'] = display_df['churn_risk_score'].apply(lambda x: f"{x:.1%}")
        
        # Rename columns for better display
        display_df.columns = [
            'Partner ID', 'Week', 'Week Date', 'Logins', 'Referrals', 
            'Earnings', 'Support Tickets', 'Join Date',
            'Region', 'Partner Type', 'Priority',
            'Days Since Outreach', 'Payout Delay (days)', 
            'Disputes', 'Competitor Flag',
            'Behavior', 'Churn Risk', 'Churn Driver', 'Recommended Action'
        ]
        
        # Display main table
        st.dataframe(
            display_df,
            use_container_width=True,
            height=TABLE_HEIGHT
        )
    
    with col_side:
        # At Risk partners table
        st.markdown("### ⚠️ At Risk Partners")
        at_risk_df = filtered_df[filtered_df['predicted_behavior'] == 'At Risk'].copy()
        st.markdown(f"**Count:** {len(at_risk_df)}")
        
        # Sort by risk score descending
        at_risk_df = at_risk_df.sort_values('churn_risk_score', ascending=False)
        
        # Prepare display
        at_risk_display = at_risk_df[['partner_id', 'churn_risk_score', 'churn_driver', 'recommended_action']].copy()
        at_risk_display['churn_risk_score'] = at_risk_display['churn_risk_score'].apply(lambda x: f"{x:.1%}")
        at_risk_display.columns = ['Partner ID', 'Risk Score', 'Driver', 'Action']
        
        st.dataframe(
            at_risk_display,
            use_container_width=True,
            height=TABLE_HEIGHT
        )
    
    # Summary metrics
    st.markdown("---")
    st.markdown("### 📊 Summary Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        avg_earnings = filtered_df['earnings'].mean()
        st.metric(
            "Average Earnings",
            f"${avg_earnings:,.2f}"
        )
    
    with metric_col2:
        avg_logins = filtered_df['logins'].mean()
        st.metric(
            "Average Logins",
            f"{avg_logins:.1f}"
        )
    
    with metric_col3:
        avg_tickets = filtered_df['unresolved_tickets'].mean()
        st.metric(
            "Avg Support Tickets",
            f"{avg_tickets:.1f}"
        )
    
    with metric_col4:
        avg_churn_risk = filtered_df['churn_risk_score'].mean()
        st.metric(
            "Avg Churn Risk",
            f"{avg_churn_risk:.1%}"
        )

elif page == "🔍 Partner Lookup":
    # Deriv-style Brand Header
    st.markdown("""
        <div class="brand-header">
            <h1>Partner Churn Prediction</h1>
            <p>Individual partner risk assessment</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Enter Partner ID")
    
    # Partner ID input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        partner_id_input = st.text_input(
            "Partner ID",
            placeholder="e.g., P0001",
            help="Enter the partner ID to view their churn risk details"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button or partner_id_input:
        if partner_id_input:
            # Search for partner in lookup dict
            partner_id_upper = partner_id_input.upper()
            
            if partner_id_upper in lookup_dict:
                partner_data = lookup_dict[partner_id_upper]
                
                # Calculate risk score
                partner_df = pd.DataFrame([partner_data])
                risk_score = predict_churn_risk(partner_df, churn_model, encoders)
                behavior = 'At Risk' if risk_score >= 0.5 else 'Healthy'
                
                st.markdown("---")
                st.markdown(f"## Partner: {partner_id_upper}")
                
                # Risk Score Display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_color = "#FF444F" if risk_score > 0.5 else "#4BB4B3"
                    
                    st.markdown(f"""
                    <div class="risk-card" style="background-color: {risk_color};">
                        <h3 style="color: white; margin: 0; border: none;">Churn Risk Score</h3>
                        <h1 style="color: white; margin: 1rem 0; font-size: 4rem; font-weight: 700;">{risk_score:.0%}</h1>
                        <p style="color: white; margin: 0; font-size: 1.2rem; font-weight: 600;">{behavior}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="risk-card" style="background-color: #FF444F;">
                        <h3 style="color: white; margin: 0; border: none;">Churn Driver</h3>
                        <h2 style="color: white; margin: 1rem 0; font-size: 1.8rem; font-weight: 600;">{partner_data['churn_driver']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="risk-card" style="background-color: #FF444F;">
                        <h4 style="color: white; margin: 0; border: none;">Recommended Action</h4>
                        <h2 style="color: white; margin: 1rem 0; font-size: 1.2rem; font-weight: 600;">{partner_data['recommended_action']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Partner Details
                st.markdown("### 📊 Partner Details")
                
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                # Decode categorical values
                region_name = decode_categorical(partner_data['region'], encoders['region'])
                partner_type_name = decode_categorical(partner_data['partner_type'], encoders['partner_type'])
                priority_name = decode_categorical(partner_data['priority'], encoders['priority'])
                
                with detail_col1:
                    st.metric("Region", region_name)
                    st.metric("Partner Type", partner_type_name)
                    st.metric("Priority", priority_name)
                    st.metric("Week", partner_data['week_number'])
                
                with detail_col2:
                    st.metric("Earnings", f"${partner_data['earnings']:,.2f}")
                    st.metric("Logins", partner_data['logins'])
                    st.metric("Referrals", partner_data['referrals'])
                    st.metric("Tenure (weeks)", partner_data['tenure_weeks'])
                
                with detail_col3:
                    st.metric("Support Tickets", partner_data['unresolved_tickets'])
                    ticket_sent = partner_data.get('ticket_sentiment', 'N/A')
                    if pd.isna(ticket_sent):
                        ticket_sent = 'N/A'
                    else:
                        ticket_sent = f"{ticket_sent:.2f}"
                    st.metric("Ticket Sentiment", ticket_sent)
                    st.metric("Payout Delay (days)", partner_data['payout_delay_days'])
                    st.metric("Commission Disputes", partner_data['commission_dispute_count'])
                
                st.markdown("---")
                
                # Additional Information
                st.markdown("### 📅 Timeline Information")
                
                timeline_col1, timeline_col2, timeline_col3 = st.columns(3)
                
                with timeline_col1:
                    join_date_str = partner_data['join_date'].strftime('%Y-%m-%d') if hasattr(partner_data['join_date'], 'strftime') else str(partner_data['join_date'])
                    st.info(f"**Join Date:** {join_date_str}")
                
                with timeline_col2:
                    week_date_str = partner_data['week_start_date'].strftime('%Y-%m-%d') if hasattr(partner_data['week_start_date'], 'strftime') else str(partner_data['week_start_date'])
                    st.info(f"**Latest Week Date:** {week_date_str}")
                
                with timeline_col3:
                    st.info(f"**Days Since Last Outreach:** {partner_data['days_since_last_outreach']}")
                
                # Risk Indicators
                st.markdown("---")
                st.markdown("### ⚠️ Risk Indicators")
                
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    competitor_flag = "Yes" if partner_data['competitor_mention_flag'] == 1 else "No"
                    st.warning(f"**Competitor Mention:** {competitor_flag}")
                
                with risk_col2:
                    if partner_data['payout_delay_days'] > 5:
                        st.error(f"**Payout Delay:** {partner_data['payout_delay_days']} days (High Risk)")
                    elif partner_data['payout_delay_days'] > 0:
                        st.warning(f"**Payout Delay:** {partner_data['payout_delay_days']} days")
                    else:
                        st.success(f"**Payout Delay:** No delays")
                
            else:
                st.error(f"❌ Partner ID '{partner_id_input}' not found. Please check and try again.")
                st.info(f"💡 Tip: Partner IDs are in format like 'P0001', 'P0002', etc. Total partners in database: {len(lookup_dict)}")
        else:
            st.warning("⚠️ Please enter a Partner ID to search.")
