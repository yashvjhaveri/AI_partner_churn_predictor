import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION & PATHS ---
# This ensures files are saved in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

print("🚀 Starting model training for GitHub deployment...")

# 1. Load and Preprocess Data
# We look for the file in the current folder
DATA_FILE = get_path('large_partner_churn_dataset_24000.csv')

if not os.path.exists(DATA_FILE):
    print(f"❌ Error: {DATA_FILE} not found. Ensure the CSV is in the same folder.")
    exit()

df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(df)} records")

# Feature Engineering
df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True)
df['join_date'] = pd.to_datetime(df['join_date'], dayfirst=True)
df['tenure_weeks'] = ((df['week_start_date'] - df['join_date']).dt.days / 7).fillna(0).astype(int)
df['is_risk'] = df['behavior'].apply(lambda x: 1 if x in ['at_risk', 'churning'] else 0)

print(f"Risk distribution: {df['is_risk'].value_counts().to_dict()}")

# 2. Encode Categorical Features
features = ['logins', 'referrals', 'earnings', 'unresolved_tickets', 'region', 
            'partner_type', 'priority', 'days_since_last_outreach', 
            'payout_delay_days', 'commission_dispute_count', 'competitor_mention_flag', 'tenure_weeks']

le_dict = {}
cat_cols = ['region', 'partner_type', 'priority']

for col in cat_cols:
    le = LabelEncoder()
    # Convert to string to handle any NaN values during training
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"Encoded {col}: {list(le.classes_)}")

# 3. Train Model 1 (Churn Prediction)
X = df[features].fillna(0) # Safety fill for any missing numeric values
y = df['is_risk']



print("\nTraining Random Forest model...")
clf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    max_depth=10, 
    min_samples_split=5
)
clf.fit(X, y)

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# 4. Prepare Model 2 (Data Lookup)
print("\nCreating lookup dictionary...")
latest_data = df.sort_values('week_number').groupby('partner_id').tail(1)
lookup_dict = latest_data.set_index('partner_id').to_dict('index')
print(f"Lookup dictionary created with {len(lookup_dict)} partners")

# 5. Save all components to .pkl files using relative paths
print("\nSaving models to repository folder...")
joblib.dump(clf, get_path('churn_risk_model.pkl'))
joblib.dump(le_dict, get_path('feature_encoders.pkl'))
joblib.dump(lookup_dict, get_path('partner_lookup_model.pkl'))

print("\n✅ All models saved successfully!")
print(f"Files created in {BASE_DIR}:")
print("  - churn_risk_model.pkl")
print("  - feature_encoders.pkl")
print("  - partner_lookup_model.pkl")

# --- QUICK TEST ---
print("\n" + "="*50)
print("Testing prediction on sample data...")
test_sample = latest_data.head(3)
test_X = test_sample[features]
test_probs = clf.predict_proba(test_X)[:, 1]

for i, p_id in enumerate(test_sample.index):
    print(f"Partner {p_id} -> Risk Probability: {test_probs[i]:.2%}")
