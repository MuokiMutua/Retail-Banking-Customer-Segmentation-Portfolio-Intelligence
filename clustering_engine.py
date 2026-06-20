import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# Suppress sklearn warnings for a clean terminal output
warnings.filterwarnings('ignore')

def run_clustering():
    print("Loading customer RFM data...")
    try:
        df = pd.read_csv("customer_rfm_data.csv")
    except FileNotFoundError:
        print("Error: Could not find customer_rfm_data.csv. Run the rfm_pipeline.py script first.")
        return
    
    # 1. Feature Engineering
    # Machine learning algorithms require numbers, not text. 
    # We use 'get_dummies' to convert the 'preferred_channel' text into binary columns (0 or 1).
    print("Preprocessing and encoding features...")
    df_encoded = pd.get_dummies(df, columns=['preferred_channel'])
    
    # Select the features we want the AI to base its decisions on
    features = [
        'age', 'recency_days', 'transaction_frequency', 
        'monetary_value_kes', 'active_microloans'
    ]
    
    # Dynamically add the newly created channel columns
    channel_cols = [col for col in df_encoded.columns if col.startswith('preferred_channel_')]
    features.extend(channel_cols)
    
    X = df_encoded[features]
    
    # 2. Feature Scaling (Crucial for K-Means)
    # Normalizes data so KES values don't overpower frequency or age
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. K-Means Clustering
    # We instruct the model to find exactly 5 distinct customer profiles
    num_clusters = 5
    print(f"Running K-Means algorithm to find {num_clusters} distinct segments...")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)
    
    # 4. Profiling the Clusters
    # We group by the new cluster ID and calculate the average stats for each group
    # to understand the 'persona' the AI has discovered.
    cluster_summary = df.groupby('cluster_id').agg({
        'age': 'mean',
        'recency_days': 'mean',
        'transaction_frequency': 'mean',
        'monetary_value_kes': 'mean',
        'active_microloans': 'mean',
        'preferred_channel': lambda x: x.mode()[0] # Get the most frequent channel
    }).round(1)
    
    print("\n--- ML CLUSTER PROFILES DISCOVERED ---")
    print(cluster_summary.to_string())
    
    # 5. Save the results
    output_file = "clustered_customers.csv"
    df.to_csv(output_file, index=False)
    print(f"\n[✓] Segmented dataset saved to {output_file}")

if __name__ == "__main__":
    run_clustering()