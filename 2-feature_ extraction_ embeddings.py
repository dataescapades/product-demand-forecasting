# %% [markdown]
## Temporal Fusion Transformer for Product Demand Forecasting
### Feature Extraction - Color and SBERT Embeddings
# Setup: import libraries

# %%
# import libraries
import pandas as pd
from sentence_transformers import SentenceTransformer

# %% [markdown]
#### Dataset

# %%
"""
Dataset: Online Retail II
Modifications: Extracted and cleaned descriptions with stock codes
Author: Daqing Chen
Source: UCI Machine Learning Repository
URL: https://doi.org/10.24432/C5CG6D
"""

# Load the dataset
df = pd.read_csv('data/stockcode_descriptions.csv')


# %% [markdown]
#### Feature Extraction: SBERT Embeddings

# %%
# Instantiate model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(df['Clean'].tolist())

# Map embeddings to stock codes
embedding_df = pd.DataFrame(embeddings, index=df.index)
df = df.join(embedding_df)

# Save to CSV
df.to_csv('data/stockcode_descriptions_with_embeddings.csv', index=False)

# %% [markdown]
#### Feature Extraction: Category SBERT Embeddings

# %%
keywords = ['set of', 'polka dot', 'heart', 'christmas']
df['Clean_Category'] = df['Clean'].copy()

# remove keywords and numbers from descriptions
for keyword in keywords:
    df['Clean_Category'] = df['Clean_Category'].str.replace(keyword, '', regex=False)
df['Clean_Category'] = df['Clean_Category'].str.replace(r'\d+', '', regex=True)

# Create embeddings for category descriptions
embeddings = model.encode(df['Clean_Category'].tolist())

# Map embeddings to stock codes
embedding_df = pd.DataFrame(embeddings, index=df.index)
df = df[['StockCode', 'Clean_Category']].join(embedding_df)

# Save to CSV
df.to_csv('data/stockcode_descriptions_with_category_embeddings.csv', index=False)

# %% [markdown]
#### Sources
"""
Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5CG6D.
"""
# %%
