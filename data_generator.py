import os
import random
import uuid
import json
import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from faker import Faker
from supabase import create_client, Client
import xgboost as xgb
import shap

fake = Faker()

# Configuration
NUM_VENDORS = 100
NUM_INVOICES = 2000
FRAUD_DIST = {
    'clean': 0.80,
    'duplicate': 0.06,
    'inflated': 0.06,
    'ghost': 0.05,
    'cycle_anomaly': 0.03
}

CATEGORIES = ['Software', 'Hardware', 'Consulting', 'Office Supplies', 'Marketing', 'Legal']

def generate_data():
    print("Generating vendors...")
    vendors = []
    for _ in range(NUM_VENDORS):
        vendors.append({
            'id': str(uuid.uuid4()),
            'name': fake.company(),
            'age_days': random.randint(1, 3650),
            'category': random.choice(CATEGORIES),
            'created_at': datetime.now().isoformat()
        })
    
    print("Generating invoices...")
    invoices = []
    
    # Pre-calculate counts based on distribution
    counts = {k: int(NUM_INVOICES * v) for k, v in FRAUD_DIST.items()}
    # Adjust for rounding errors
    counts['clean'] += NUM_INVOICES - sum(counts.values())
    
    fraud_labels = []
    for k, v in counts.items():
        fraud_labels.extend([k] * v)
    random.shuffle(fraud_labels)
    
    vendor_history = {v['id']: [] for v in vendors}
    
    predictions = []
    
    for i in range(NUM_INVOICES):
        v = random.choice(vendors)
        f_type = fraud_labels[i]
        
        # Base invoice
        inv_date = fake.date_between(start_date='-1y', end_date='today')
        amount = round(random.uniform(100, 5000), 2)
        po_match = random.random() > 0.2
        line_count = random.randint(1, 20)
        
        # Apply anomalies based on f_type
        if f_type == 'ghost':
            v['age_days'] = random.randint(1, 14) # Ghost vendors are usually new
            po_match = False
        elif f_type == 'inflated':
            amount = round(amount * random.uniform(3.0, 10.0), 2)
        elif f_type == 'duplicate':
            if vendor_history[v['id']]:
                # Copy last invoice
                last_inv = vendor_history[v['id']][-1]
                amount = last_inv['amount']
                inv_date = last_inv['date'] + timedelta(days=random.randint(0, 2))
                line_count = last_inv['line_count']
        elif f_type == 'cycle_anomaly':
            # E.g. billed on a weekend when usually not
            while inv_date.weekday() < 5:
                inv_date = fake.date_between(start_date='-1y', end_date='today')
                
        inv = {
            'id': str(uuid.uuid4()),
            'vendor_id': v['id'],
            'amount': amount,
            'date': inv_date,
            'status': 'pending' if random.random() > 0.5 else ('approved' if random.random() > 0.2 else 'escalated'),
            'line_count': line_count,
            'po_match': po_match,
            'created_at': datetime.now().isoformat()
        }
        invoices.append(inv)
        vendor_history[v['id']].append(inv)
        predictions.append({'invoice_id': inv['id'], 'fraud_type_label': f_type})

    return pd.DataFrame(vendors), pd.DataFrame(invoices), pd.DataFrame(predictions)

def extract_features(df_inv, df_ven):
    print("Extracting features...")
    df = df_inv.merge(df_ven, left_on='vendor_id', right_on='id', suffixes=('', '_vendor'))
    
    # 1. amount_z_score
    df['amount_z_score'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    
    # 2. days_since_last
    df = df.sort_values(by=['vendor_id', 'date'])
    df['days_since_last'] = df.groupby('vendor_id')['date'].diff().dt.days.fillna(30)
    
    # 3. vendor_age
    df['vendor_age'] = df['age_days']
    
    # 4. frequency (invoices per vendor)
    vendor_freq = df.groupby('vendor_id').size().reset_index(name='frequency')
    df = df.merge(vendor_freq, on='vendor_id')
    
    # 5. round_number_flag
    df['round_number_flag'] = (df['amount'] % 100 == 0).astype(int)
    
    # 6. weekend_flag
    df['weekend_flag'] = df['date'].dt.weekday.isin([5, 6]).astype(int)
    
    # 7. line_count
    # already in df
    
    # 8. ghost_flag
    df['ghost_flag'] = ((df['vendor_age'] < 30) & (~df['po_match'])).astype(int)
    
    # 9. category_z_score
    df['category_z_score'] = df.groupby('category')['amount'].transform(lambda x: (x - x.mean()) / x.std()).fillna(0)
    
    # 10. po_match (bool to int)
    df['po_match_int'] = df['po_match'].astype(int)
    
    features = ['amount_z_score', 'days_since_last', 'vendor_age', 'frequency', 
                'round_number_flag', 'weekend_flag', 'line_count', 'ghost_flag', 
                'category_z_score', 'po_match_int']
                
    return df[features], df

def train_model(df_features, df_labels):
    print("Training XGBoost...")
    X = df_features
    # Binary target: 1 if not clean
    y = (df_labels['fraud_type_label'] != 'clean').astype(int)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    
    # Get probabilities
    probs = model.predict_proba(X)[:, 1]
    
    # Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Save artifacts
    print("Saving model and explainer...")
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('shap_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)
        
    return probs, shap_values, X.columns

def push_to_supabase(df_ven, df_inv, df_pred):
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("Skipping Supabase push: SUPABASE_URL or SUPABASE_KEY not found.")
        return
        
    print(f"Connecting to Supabase at {url}")
    supabase: Client = create_client(url, key)
    
    print("Pushing vendors...")
    # Convert date cols if necessary, fillna
    v_records = df_ven[['id', 'name', 'age_days', 'category', 'created_at']].to_dict(orient='records')
    # Batch insert in chunks of 50
    for i in range(0, len(v_records), 50):
        supabase.table('vendors').insert(v_records[i:i+50]).execute()
        
    print("Pushing invoices...")
    df_inv['date'] = df_inv['date'].astype(str)
    i_records = df_inv[['id', 'vendor_id', 'amount', 'date', 'status', 'line_count', 'po_match', 'created_at']].to_dict(orient='records')
    for i in range(0, len(i_records), 500):
        supabase.table('invoices').insert(i_records[i:i+500]).execute()
        
    print("Pushing predictions...")
    p_records = df_pred.to_dict(orient='records')
    for i in range(0, len(p_records), 500):
        supabase.table('predictions').insert(p_records[i:i+500]).execute()
        
    print("Data seeded successfully!")

def main():
    df_ven, df_inv, df_pred_labels = generate_data()
    # Ensure datetimes
    df_inv['date'] = pd.to_datetime(df_inv['date'])
    
    df_features, df_merged = extract_features(df_inv, df_ven)
    
    # Ensure correct order for labels
    df_pred_labels = pd.merge(df_merged[['id']], df_pred_labels, left_on='id', right_on='invoice_id', how='left')
    
    probs, shap_values, feature_names = train_model(df_features, df_pred_labels)
    
    # Prepare predictions table
    pred_records = []
    for i, row in df_merged.iterrows():
        # Get top 3 drivers
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[-3:][::-1]
        drivers = {feature_names[j]: float(sv[j]) for j in top_idx}
        
        # Risk score from model
        score = float(probs[i])
        
        # Apply rule engine overrides
        final_fraud_type = df_pred_labels.iloc[i]['fraud_type_label']
        if final_fraud_type == 'clean' and score > 0.8:
            final_fraud_type = 'extreme_outlier'
            
        pred_records.append({
            'id': str(uuid.uuid4()),
            'invoice_id': row['id'],
            'risk_score': round(score * 100, 2),
            'fraud_type': final_fraud_type,
            'drivers': drivers,
            'explanation': None,
            'created_at': datetime.now().isoformat()
        })
        
    df_predictions = pd.DataFrame(pred_records)
    
    push_to_supabase(df_ven, df_inv, df_predictions)

if __name__ == "__main__":
    main()
