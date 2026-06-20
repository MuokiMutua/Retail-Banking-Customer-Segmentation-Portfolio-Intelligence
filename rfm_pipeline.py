import pandas as pd
import numpy as np
import random
import os

def generate_banking_data(num_customers=5000):
    """
    Generates a synthetic but realistic retail banking dataset based on 
    mobile money and traditional branch banking behaviors.
    """
    print(f"Generating synthetic behavioral data for {num_customers} customers...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate Customer IDs
    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, num_customers + 1)]
    
    # Generate Demographics
    ages = np.random.normal(loc=35, scale=12, size=num_customers).astype(int)
    ages = np.clip(ages, 18, 80) # Ensure ages are between 18 and 80
    
    # Generate RFM (Recency, Frequency, Monetary) Metrics
    # Recency: Days since last transaction (0 to 90 days)
    # Most people transact often, so we use an exponential distribution to skew towards recent days
    recency = np.random.exponential(scale=15, size=num_customers).astype(int)
    recency = np.clip(recency, 0, 90)
    
    # Frequency: Number of transactions per month
    frequency = np.random.lognormal(mean=2.5, sigma=1.0, size=num_customers).astype(int)
    frequency = np.clip(frequency, 1, 150)
    
    # Monetary: Total transaction value per month (in KES)
    # We correlate monetary value slightly with frequency
    base_monetary = np.random.lognormal(mean=9.0, sigma=1.5, size=num_customers)
    monetary = (base_monetary + (frequency * 500)).round(2)
    
    # Generate Behavioral Features
    channels = ["Mobile App", "USSD / Mobile Money", "Branch", "ATM"]
    # Probabilities leaning heavily towards digital, as is typical in modern African banking
    preferred_channel = np.random.choice(channels, size=num_customers, p=[0.45, 0.40, 0.10, 0.05])
    
    # Loan dependency: How many active micro-loans do they take per year?
    active_loans = np.random.poisson(lam=1.5, size=num_customers)
    active_loans = np.clip(active_loans, 0, 10)
    
    # Compile into a DataFrame
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "recency_days": recency,
        "transaction_frequency": frequency,
        "monetary_value_kes": monetary,
        "preferred_channel": preferred_channel,
        "active_microloans": active_loans
    })
    
    return df

def run_pipeline():
    # 1. Generate Data
    df = generate_banking_data(num_customers=8000)
    
    # 2. Save to CSV for the Machine Learning phase
    output_file = "customer_rfm_data.csv"
    df.to_csv(output_file, index=False)
    
    # 3. Print a quick exploratory summary
    print(f"\n[✓] Successfully saved data to {output_file}")
    print("-" * 50)
    print("DATASET PREVIEW (First 5 rows):")
    print(df.head())
    print("-" * 50)
    print("\nCHANNEL DISTRIBUTION:")
    print(df['preferred_channel'].value_counts(normalize=True).map('{:.1%}'.format))
    print("-" * 50)
    print("RFM STATISTICS:")
    print(df[['recency_days', 'transaction_frequency', 'monetary_value_kes']].describe().round(2).loc[['mean', 'min', 'max']])

if __name__ == "__main__":
    run_pipeline()