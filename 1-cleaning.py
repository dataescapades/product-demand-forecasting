# %% [markdown]
## Temporal Fusion Transformer for Product Demand Forecasting
### Data Cleaning and Preprocessing
# Setup: import libraries

# %%
# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('src')
from functions import basic_text_clean
import json

# %% [markdown]
#### Dataset

# %%
"""
Dataset: Online Retail II
Author: Daqing Chen
Source: UCI Machine Learning Repository
URL: https://doi.org/10.24432/C5CG6D
"""

# Load the dataset and set InvoiceDate as a datetime
df = pd.read_csv('data/online_retail_II - edited.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Sort by InvoiceDate
df = df.sort_values('InvoiceDate').reset_index(drop=True)

# %% [markdown]
#### Duplicates

# %%
# Create returns and duplicates dataframes
df['Invoice'] = df['Invoice'].astype(str)
returns = df[df['Invoice'].str.contains('C')]
duplicates = df[df.duplicated()]

# plot distribution of Overall sales over time
fig, ax = plt.subplots(figsize=(24, 6))
sns.histplot(df['InvoiceDate'], bins=105, kde=True)
plt.title('Distribution of Overall Sales Over Time')
ax.xaxis.set_major_locator(plt.MaxNLocator(25))
plt.show()

# plot distributions of returns and duplicates over time on the same graph
fig, ax = plt.subplots(figsize=(24, 6))
sns.histplot(returns['InvoiceDate'], bins=105, kde=True, label='Returns',
             color='red', alpha=0.3)
sns.histplot(duplicates['InvoiceDate'], bins=105, kde=True, label='Duplicates',
             color='blue', alpha=0.3)
plt.title('Distribution of Returns and Duplicates Over Time')
ax.xaxis.set_major_locator(plt.MaxNLocator(25))
plt.legend()
plt.show()

#%% [markdown]
#### Missing Values

# %%
# Identify records with missing values
nulls = df[df.isna().any(axis=1)]
print(nulls.info())

# plot distribution of missing values over time
fig, ax = plt.subplots(figsize=(24, 6))
sns.histplot(nulls['InvoiceDate'], bins=105, kde=True)
plt.title('Distribution of Missing Values Over Time')
ax.xaxis.set_major_locator(plt.MaxNLocator(25))
plt.show()

# Remove records with missing descriptions
df = df.dropna(subset=['Description'])

# %% [markdown]
#### Non-Product Entries

# %%
# Remove records with lowercase descriptions
lowercase = df[df['Description'].astype(str).str.islower()]
df = df[~df['Description'].astype(str).str.islower()]

# Remove records with keywords in StockCode or Description
nonproduct_stockcodes = (
    'm', 'gift_0001_80', 'gift_0001_70', 'gift_0001_50', 'gift_0001_40', 
    'gift_0001_30', 'gift_0001_20', 'gift_0001_10', 'TEST002', 'TEST001', 
    'S', 'POST', 'M', 'DOT', 'D', 'CRUK', 'C2', 'BANK CHARGES', 'B',
    'AMAZONFEE', 'ADJUST2', 'ADJUST')
df = df[~df['StockCode'].isin(nonproduct_stockcodes)]
keywords = [
    'Amazon', 'adjust', 'found', 'mailout', 'next day carriage', 'sold ', 
    'fba', ' aside', 'error', 'incorr', '22719', 'john', 'marked']
for keyword in keywords:
    df = df[~df['Description'].str.contains(keyword, case=False, na=False)]

# Remove records with other indicators of non-product entries
df = df[~((df['Price'] == 0) & (df['Quantity'] < 0))]
df = df[df['Description'] != '?']
df = df[df['Quantity'].abs() <= 20000]
df = df[df['Quantity'] > 0]

# %% [markdown]
#### Anomaly: StockCode 84016

# %%
# Distribution of StockCode 84016 over time
stockcode_84016 = df[df['StockCode'] == '84016']

fig, ax = plt.subplots(figsize=(24, 6))
sns.histplot(stockcode_84016['InvoiceDate'], bins=105, kde=True)
plt.title('Distribution of StockCode 84016 Over Time')
ax.xaxis.set_major_locator(plt.MaxNLocator(25))
plt.xticks(rotation=45)
plt.show()

# Remove StockCode 84016
df = df[df['StockCode'] != '84016']

# %% [markdown]
#### Quantity Caps

# %%
# Distribution of Quantity - boxplot
plt.figure(figsize=(12, 3))
sns.boxplot(x=df['Quantity'])
plt.title('Quantity')
plt.show()

# Evaluate possible quantity caps
pct_over_100_qty = (df[df['Quantity'] > 100].shape[0] / df.shape[0]) * 100
pct_over_500_qty = (df[df['Quantity'] > 500].shape[0] / df.shape[0]) * 100
pct_over_1000_qty = (df[df['Quantity'] > 1000].shape[0] / df.shape[0]) * 100
pct_over_2000_qty = (df[df['Quantity'] > 2000].shape[0] / df.shape[0]) * 100
pct_over_2500_qty = (df[df['Quantity'] > 2500].shape[0] / df.shape[0]) * 100
pct_over_5000_qty = (df[df['Quantity'] > 5000].shape[0] / df.shape[0]) * 100

qty_df = pd.DataFrame({
    'Quantity Threshold': ['>100', '>500', '>1000', '>2000', '>2500', '>5000'],
    'Percentage of Records': [pct_over_100_qty, pct_over_500_qty, pct_over_1000_qty,
                              pct_over_2000_qty, pct_over_2500_qty, pct_over_5000_qty]
})
qty_df['Percentage of Records'] = qty_df['Percentage of Records'].round(4)
print(qty_df)

# Remove records with quantity over 500
df = df[df['Quantity'] <= 500]

# %% [markdown]
#### Description: Cleaning and Standardization

# %%
# Create 'Clean' column and apply basic text cleaning
df['Clean'] = df['Description'].copy()
df['Clean'] = basic_text_clean(df['Clean'])

# Convert StockCode to uppercase string
df['StockCode'] = df['StockCode'].astype(str).str.upper()

# Import dictionaries and apply them to the 'Clean' column
with open('data/variations_dict.json', 'r') as f:
    variations_dict = json.load(f)
with open('data/variations_dict_regex.json', 'r') as f:
    variations_dict_regex = json.load(f)
with open('data/misspelled_words.json', 'r') as f:
    misspelled_words = json.load(f)
for standard, variations in variations_dict.items():
    for variation in variations:
        df['Clean'] = df['Clean'].str.replace(variation, standard, regex=False)
for standard, patterns in variations_dict_regex.items():
    for pattern in patterns:
        df['Clean'] = df['Clean'].str.replace(pattern, standard, regex=True)
for misspelled, correct in misspelled_words.items():
    df['Clean'] = df['Clean'].str.replace(misspelled, correct, regex=False)

# Fix other errors
df['Clean'] = df['Clean'].str.replace('set of set of ', 'set of ', regex=False)
df['Clean'] = df['Clean'].str.replace('box/', '', regex=False)
df['Clean'] = df['Clean'].str.replace('spots', 'polka dot', regex=False)
df['Clean'] = df['Clean'].str.replace('m.o.p', 'mother-of-pearl', regex=False)
df['Clean'] = df['Clean'].str.replace('spot', 'polka dot', regex=False)

# Final basic cleaning to address any new issues created in previous steps
df['Clean'] = basic_text_clean(df['Clean'])

# Identify StockCodes with multiple descriptions after cleaning
description_counts = df.groupby('StockCode')['Clean'].nunique()
multiple_descriptions = (
    description_counts[description_counts > 1])

# Filter dataframe for stock codes with multiple descriptions
df_multiple_descriptions = df[df['StockCode']
                              .isin(multiple_descriptions.index)]

# Add column for quantity of each description for each stock code
df_multiple_descriptions = (
    df_multiple_descriptions
    .groupby(['StockCode', 'Clean'], as_index=False)
    .agg({'Quantity': 'sum'}))

# Add column for total quantity for each description within each stock code
df_multiple_descriptions['Total Quantity'] = (
    df_multiple_descriptions
    .groupby('StockCode')['Quantity']
    .transform('sum'))

# Add column for percentage of quantity for each description within each stock code
df_multiple_descriptions['Percentage'] = (
    df_multiple_descriptions['Quantity'] 
    / df_multiple_descriptions['Total Quantity']).round(4)

# Standardize descriptions in df based on the most common description for each stock code
most_common_descriptions = (
    df_multiple_descriptions
    .groupby('StockCode')['Percentage']
    .idxmax())

most_common_pct = df_multiple_descriptions.loc[most_common_descriptions,
                                               'Percentage']
print(most_common_pct.describe())

description_map = (
    df_multiple_descriptions.loc[most_common_descriptions]
    .set_index('StockCode')['Clean']
    .to_dict())
df['Clean'] = df.apply(
    lambda row: description_map[row['StockCode']] 
    if row['StockCode'] in description_map
    else row['Clean'], axis=1)

# %% [markdown]
#### Aggregate by Day and StockCode

# %%
# Convert InvoiceDate to date format (remove time component)
df['InvoiceDate'] = df['InvoiceDate'].dt.date

# Remove unnecessary columns for modeling
df = df.drop(columns=['Customer ID', 'Country', 'Description'])

# Aggregate by Day and StockCode (include "Clean")
df = (
    df
    .groupby(['InvoiceDate', 'StockCode', 'Clean'], as_index=False)
    .agg({'Quantity': 'sum',
          'Price': 'mean'}).round(4))

# %% [markdown]
#### Uncommon Products

# %%
# Identify uncommon StockCodes (those that appear less than 20 times)
stockcode_counts = df['StockCode'].value_counts()
uncommon_stockcodes = stockcode_counts[stockcode_counts < 20].index

# Remove records with uncommon StockCodes
df = df[~df['StockCode'].isin(uncommon_stockcodes)]

# %% [markdown]
#### New Products

# %%
# Remove products that first appear within test and encoder windows (final 150 days)
first_appearance = df.groupby('StockCode')['InvoiceDate'].min()
cutoff_date = df['InvoiceDate'].max() - pd.Timedelta(days=150)
new_products = first_appearance[first_appearance > cutoff_date].index
df = df[~df['StockCode'].isin(new_products)]

# %% [markdown]
#### Final Check

# %%
fig, ax = plt.subplots(figsize=(24, 6))
sns.histplot(pd.to_datetime(df['InvoiceDate']), bins=105, kde=True)
plt.title('Distribution of Sales Over Time After Cleaning')
ax.xaxis.set_major_locator(plt.MaxNLocator(25))
plt.show()

# Export cleaned dataset
df.to_csv('data/online_retail_II_clean.csv', index=False)

# Export unique StockCode and Clean descriptions for feature engineering
stockcode_descriptions = (df[['StockCode', 'Clean']].drop_duplicates().reset_index(drop=True))
stockcode_descriptions.to_csv('data/stockcode_descriptions.csv', index=False)

# %% [markdown]
#### Sources
"""
Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5CG6D.
"""
# %%
