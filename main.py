from modules.data_preprocessing import load_data
from modules.clustering import HybridLinkBasedClustering

# Load Dataset
dataset_path = 'dataset/sample_dataset.csv'
df = load_data(dataset_path)

# Initialize HLBCA Algorithm
hlbca = HybridLinkBasedClustering(threshold=0.75)
clusters = hlbca.fit(df)

print("Clustering completed successfully.")
