# %% [markdown]
## Temporal Fusion Transformer for Product Demand Forecasting
### Feature Extraction - Category
# Setup: import libraries

# %%
# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from sklearn.decomposition import PCA
from functions import plot_kmeans_tsne_3d

# %% [markdown]
#### Category Labels

# %%
"""
Dataset: Online Retail II
Modifications: Extracted and cleaned descriptions with stock codes
Author: Daqing Chen
Source: UCI Machine Learning Repository
URL: https://doi.org/10.24432/C5CG6D
"""

# Dataset with keywords removed from descriptions
df = pd.read_csv('data/stockcode_descriptions_with_category_embeddings.csv')
embeddings = df.iloc[:, 2:].values

# Apply PCA with the number of components equal to the original embedding size
pca = PCA(n_components=len(embeddings[0]))
principal_components = pca.fit_transform(embeddings)

# dataframe of explained variance per component
explained_variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Explained Variance Ratio': (pca.explained_variance_ratio_).round(4)})

# cumulative explained variance
explained_variance_df['Cumulative Explained Variance'] = (
    (explained_variance_df['Explained Variance Ratio'].cumsum() * 100).round(2))

# Scree Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, len(embeddings[0])+1),
             y=pca.explained_variance_ratio_,
             marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# 25% explained variance
plot_kmeans_tsne_3d(df, 'Clean_Category', principal_components, 10, 10)

# Create dataframe with stockcode and columns that end in 'Clust'
cluster_cols = [col for col in df.columns if col.endswith('Clust')]
df_labels = df[['StockCode'] + cluster_cols]

# %% [markdown]
#### Subcategory Labels

# %%
# Dataset with unaltered descriptions
df = pd.read_csv('data/stockcode_descriptions_with_embeddings.csv')
embeddings = df.iloc[:, 2:].values

# Apply PCA with the number of components equal to the original embedding size
pca = PCA(n_components=len(embeddings[0]))
principal_components = pca.fit_transform(embeddings)

# dataframe of explained variance per component
explained_variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Explained Variance Ratio': (pca.explained_variance_ratio_).round(4)})

# cumulative explained variance
explained_variance_df['Cumulative Explained Variance'] = (
    (explained_variance_df['Explained Variance Ratio'].cumsum() * 100).round(2))

# Scree Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, len(embeddings[0])+1),
             y=pca.explained_variance_ratio_,
             marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# 50%, 75%, and 85% explained variance
n_comp = [29, 71, 107]
n_clust = [20, 40, 60]

for n_c, n_cl in zip(n_comp, n_clust):
    plot_kmeans_tsne_3d(df, 'Clean', principal_components, n_c, n_cl)

# Add 'Clean' and columns that end in 'Clust' to existing df_labels
cluster_cols = [col for col in df.columns if col.endswith('Clust')]
df_labels = df_labels.merge(
    df[['StockCode', 'Clean'] + cluster_cols],
    on='StockCode',
    how='left')

df_labels = df_labels.iloc[:, [0, 2, 1, 3, 4, 5]]

# Summary statistics of cluster labels
for col in df_labels.columns[2:]:
    print(f"Summary statistics for {col}:")
    print(f"Minimum count: {df_labels[col].value_counts().min()}")
    print(f"Maximum count: {df_labels[col].value_counts().max()}")
    print(f"Average count: {df_labels[col].value_counts().mean()}")
    print("\n")

# Unique combinations of cluster labels
print(f"Number of unique combinations: {df_labels.iloc[:, 2:].drop_duplicates().shape[0]}")

# Save df_labels to csv
df_labels.to_csv('data/stockcode_category_labels.csv', index=False)

# %% [markdown]
#### Sources
"""
Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5CG6D.
"""
# %%
