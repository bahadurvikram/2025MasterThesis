import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import pdist, squareform

# Download historical data for a list of stocks
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
data = yf.download(stocks, start='2020-01-01', end='2023-01-01')['Close']

# Normalize price series
normalized_data = data.apply(lambda x: (x - x.mean()) / x.std())

# Compute distance matrix
distance_matrix = pd.DataFrame(squareform(pdist(normalized_data.T)), columns=stocks, index=stocks)

# Find pairs with the smallest distance
min_distance_pairs = []
for i in range(len(distance_matrix.columns)):
    for j in range(i + 1, len(distance_matrix.columns)):
        min_distance_pairs.append((distance_matrix.columns[i], distance_matrix.columns[j], distance_matrix.iloc[i, j]))

# Sort pairs by distance
min_distance_pairs.sort(key=lambda x: x[2])

print("Pairs with Smallest Distance:")
for pair in min_distance_pairs[:5]:  # Top 5 pairs
    print(pair)