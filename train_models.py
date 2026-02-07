import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("🚀 Starting model training...")

# ===============================
# 1. Load Dataset
# ===============================
DATA_PATH = "large_partner_churn_dataset_24000.csv"

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} records")

# ===============================
# 2. Feature Engineering
# ===============================
df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True)
df['join_date'] = pd.to_datetime(df['join_date'], dayfirst=True)

df['tenure_weeks'] = (
    (df['week_start_date'] - df['join_date']).dt.days / 7
).astype(int)

# Target: early churn risk
df['is_risk'] = df['behavior'].apply(
    lambda x: 1 if x.lower() in ['at_risk', 'churning'] else 0
)

print("📊 Risk distribution:")
print(df['is_risk'].value_counts())

# ===============================
# 3. Encode Categorical Features
# ===============================
features = [
    'logins',
    'referrals',
    'earnings',
    'unresolved_tickets',
    'region',
    'partner_type',
    'priority',
    'days_since_last_outreach',
    'payout_delay_days',
    'commission_dispute_count',
    'competitor_mention_flag',
    'tenure_weeks'
]

cat_cols = ['region', 'partner_type', 'priority']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"🔹 Encoded {col}: {list(le.classes_)}")

# ===============================
# 4. Train Churn Risk Model
# ===============================
X = df[features]
y = df['is_risk']

print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42
)

model.fit(X, y)

# ===============================
# 5. Feature Importance (Debug Info)
# ===============================
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n🔍 Top Feature Importances:")
print(importance.head(10))

# ===============================
# 6. Partner Lookup Model
# ===============================
print("\n📦 Creating partner lookup dictionary...")
latest_data = df.sort_values('week_number').groupby('partner_id').tail(1)
lookup_dict = latest_data.set_index('partner_id').to_dict('index')
print(f"✅ Lookup created for {len(lookup_dict)} partners")

# ===============================
# 7. Save Artifacts (ROOT)
# ===============================
joblib.dump(model, "churn_risk_model.pkl")
joblib.dump(encoders, "feature_encoders.pkl")
joblib.dump(lookup_dict, "partner_lookup_model.pkl")

print("\n✅ Training complete!")
print("Saved files:")
print(" - churn_risk_model.pkl")
print(" - feature_encoders.pkl")
print(" - partner_lookup_model.pkl")
