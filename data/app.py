import pandas as pd

# Load the dataset
df = pd.read_csv('optimized_dataset.csv')

# Display the first 5 rows
print("\n", df.head(), "\n", df.count(), "\n")
