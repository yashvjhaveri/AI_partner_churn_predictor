import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print("Starting model training...")

# 1. Load and Preprocess Data
df = pd.read_csv(r'A:\Deriv2.0\large_partner_churn_dataset_24000.csv')
print(f"Loaded {len(df)} records")

# Feature Engineering
df['week_start_date'] = pd.to_datetime(df['week_start_date'], dayfirst=True)
df['join_date'] = pd.to_datetime(df['join_date'], dayfirst=True)
df['tenure_weeks'] = ((df['week_start_date'] - df['join_date']).dt.days / 7).astype(int)
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
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"Encoded {col}: {list(le.classes_)}")

# 3. Train Model 1 (Churn Prediction)
X = df[features]
y = df['is_risk']

print("\nTraining Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
clf.fit(X, y)

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# 4. Prepare Model 2 (Data Lookup)
# Using latest record for each partner for the lookup model
print("\nCreating lookup dictionary...")
latest_data = df.sort_values('week_number').groupby('partner_id').tail(1)
lookup_dict = latest_data.set_index('partner_id').to_dict('index')
print(f"Lookup dictionary created with {len(lookup_dict)} partners")

# 5. Save all components to .pkl files
print("\nSaving models...")
joblib.dump(clf, r'A:\Deriv2.0\churn_risk_model.pkl')
joblib.dump(le_dict, r'A:\Deriv2.0\feature_encoders.pkl')
joblib.dump(lookup_dict, r'A:\Deriv2.0\partner_lookup_model.pkl')

print("\n✅ All models saved successfully!")
print("Files created in A:\\Deriv2.0:")
print("  - churn_risk_model.pkl")
print("  - feature_encoders.pkl")
print("  - partner_lookup_model.pkl")

# Test the model
print("\n" + "="*50)
print("Testing the model with sample predictions...")
print("="*50)

# Test prediction on first 5 partners
test_sample = latest_data.head(5)
test_X = test_sample[features]
test_predictions = clf.predict_proba(test_X)[:, 1]

for idx, (partner_id, row) in enumerate(test_sample.iterrows()):
    risk_prob = test_predictions[idx]
    actual_behavior = row['behavior']
    print(f"\nPartner {partner_id}:")
    print(f"  Predicted Risk: {risk_prob:.1%}")
    print(f"  Actual Behavior: {actual_behavior}")
    print(f"  Status: {'✓ Match' if (risk_prob >= 0.5 and actual_behavior in ['at_risk', 'churning']) or (risk_prob < 0.5 and actual_behavior == 'healthy') else '✗ Mismatch'}")
