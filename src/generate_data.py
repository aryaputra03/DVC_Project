"""
Generate Synthetic Dataset untuk DVC Classification Project
- 10 fitur (4 kategorik, 6 numerik)
- Missing values di semua fitur (1 fitur dengan missing parah ~30%)
- 2 fitur numerik dengan distribusi sangat skewed
- Total 10.000 baris
- Target: Binary classification (0, 1)
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import argparse

def generate_dataset(n_rows: int = 10000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    print(f"🔄 Generating dataset with {n_rows} rows...")

    age = np.random.normal(loc=40, scale=15, size=n_rows)
    age = np.clip(age, 18, 80).astype(int)

    income = np.random.lognormal(mean=10.5, sigma=0.8, size=n_rows)
    income = np.clip(income, 10000, 500000).astype(int)
    
    transaction_amount = np.random.exponential(scale=500, size=n_rows)
    outlier_indices = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    transaction_amount[outlier_indices] = np.random.uniform(5000, 20000, size=len(outlier_indices))
    transaction_amount = np.round(transaction_amount, 2)

    credit_score = np.random.normal(loc=650, scale=80, size=n_rows)
    credit_score = np.clip(credit_score, 300, 850).astype(int)

    account_balance = np.random.uniform(0, 100000, size=n_rows)
    account_balance = np.round(account_balance, 2)

    years_customer = np.random.poisson(lam=5, size=n_rows)
    years_customer = np.clip(years_customer, 0, 30).astype(int)
    
    # ========================================
    # FITUR KATEGORIK (4 fitur)
    # ========================================

    education_levels = ['High School', 'Bachelor', 'Master', 'PhD', 'Other']
    education_probs = [0.30, 0.40, 0.20, 0.05, 0.05]
    education = np.random.choice(education_levels, size=n_rows, p=education_probs)

    employment_status = ['Employed', 'Self-Employed', 'Unemployed', 'Retired', 'Student']
    employment_probs = [0.55, 0.15, 0.10, 0.12, 0.08]
    employment = np.random.choice(employment_status, size=n_rows, p=employment_probs)

    regions = ['North', 'South', 'East', 'West', 'Central']
    region = np.random.choice(regions, size=n_rows)

    account_types = ['Basic', 'Standard', 'Premium', 'VIP']
    account_probs = [0.35, 0.35, 0.20, 0.10]
    account_type = np.random.choice(account_types, size=n_rows, p=account_probs)
    
    # ========================================
    # TARGET VARIABLE (berkorelasi dengan fitur)
    # ========================================
    
    # Buat target dengan beberapa logika
    target_prob = (
        0.1 +
        0.2 * (income > 50000) +
        0.15 * (credit_score > 700) +
        0.1 * (years_customer > 5) +
        0.1 * (np.isin(account_type, ['Premium', 'VIP'])) +
        0.1 * (np.isin(education, ['Master', 'PhD']))
    )
    target_prob = np.clip(target_prob + np.random.uniform(-0.1, 0.1, n_rows), 0.05, 0.95)
    target = (np.random.random(n_rows) < target_prob).astype(int)
    
    # ========================================
    # CREATE DATAFRAME
    # ========================================
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'transaction_amount': transaction_amount,
        'credit_score': credit_score,
        'account_balance': account_balance,
        'years_customer': years_customer,
        'education': education,
        'employment_status': employment,
        'region': region,
        'account_type': account_type,
        'target': target
    })
    
    # ========================================
    # INJECT MISSING VALUES
    # ========================================
    
    print("🔄 Injecting missing values...")
    
    missing_severe = np.random.choice(n_rows, size=int(n_rows * 0.30), replace=False)
    df.loc[missing_severe, 'employment_status'] = np.nan

    missing_moderate1 = np.random.choice(n_rows, size=int(n_rows * 0.15), replace=False)
    df.loc[missing_moderate1, 'income'] = np.nan
  
    missing_moderate2 = np.random.choice(n_rows, size=int(n_rows * 0.12), replace=False)
    df.loc[missing_moderate2, 'credit_score'] = np.nan

    missing_light1 = np.random.choice(n_rows, size=int(n_rows * 0.08), replace=False)
    df.loc[missing_light1, 'education'] = np.nan

    missing_light2 = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    df.loc[missing_light2, 'age'] = np.nan

    missing_light3 = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
    df.loc[missing_light3, 'transaction_amount'] = np.nan

    missing_light4 = np.random.choice(n_rows, size=int(n_rows * 0.03), replace=False)
    df.loc[missing_light4, 'account_balance'] = np.nan

    missing_vlight1 = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    df.loc[missing_vlight1, 'years_customer'] = np.nan

    missing_vlight2 = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    df.loc[missing_vlight2, 'region'] = np.nan

    missing_vlight3 = np.random.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    df.loc[missing_vlight3, 'account_type'] = np.nan
    
    return df


def print_dataset_info(df: pd.DataFrame):
    """Print dataset information"""
    
    print("\n" + "="*60)
    print("📊 DATASET INFORMATION")
    print("="*60)
    
    print(f"\n📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Columns & Types:")
    for col in df.columns:
        dtype = df[col].dtype
        print(f"   - {col}: {dtype}")
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    for col in df.columns:
        if missing[col] > 0:
            pct = (missing[col] / len(df)) * 100
            severity = "🔴 PARAH" if pct > 25 else "🟡 SEDANG" if pct > 10 else "🟢 RINGAN"
            print(f"   - {col}: {missing[col]} ({pct:.1f}%) {severity}")
    
    print("\n📈 Skewness (Numerical Features):")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col != 'target':
            skew = df[col].skew()
            severity = "⚠️ HIGHLY SKEWED" if abs(skew) > 2 else ""
            print(f"   - {col}: {skew:.3f} {severity}")
    
    print("\n🎯 Target Distribution:")
    target_counts = df['target'].value_counts()
    for val, count in target_counts.items():
        pct = (count / len(df)) * 100
        print(f"   - Class {val}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--rows', type=int, default=10000, help='Number of rows')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='data/raw/dataset.csv', help='Output path')
    args = parser.parse_args()
    
    # Generate dataset
    df = generate_dataset(n_rows=args.rows, random_state=args.seed)
    
    # Print info
    print_dataset_info(df)
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save dataset
    df.to_csv(args.output, index=False)
    print(f"\n✅ Dataset saved to: {args.output}")
    print(f"📦 File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == "__main__":
    main()