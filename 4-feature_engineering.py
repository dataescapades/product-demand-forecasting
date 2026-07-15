# %% [markdown]
## Temporal Fusion Transformer for Product Demand Forecasting
### Feature Engineering
# Setup: import libraries

# %%
# import libraries
import pandas as pd
import numpy as np
import sys
sys.path.append('src')
from functions import days_to_xmas_break
import random

# %% [markdown]
#### Dataset

# %%
"""
Dataset: Online Retail II
Modifications: Cleaned and aggregated in previous notebooks
Author: Daqing Chen
Source: UCI Machine Learning Repository
URL: https://doi.org/10.24432/C5CG6D
"""

# Load the dataset and set InvoiceDate as a date
df = pd.read_csv('data/online_retail_II_clean.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.normalize()

# %% [markdown]
#### Time Series: Variable Start and time_idx

# %%
# Define date thresholds and first date of sale for each product
dataset_start = df['InvoiceDate'].min()
dataset_end = df['InvoiceDate'].max()
all_dates = pd.date_range(start=dataset_start, end=dataset_end)
time_idx = pd.DataFrame({'InvoiceDate': all_dates, 'time_idx': range(len(all_dates))})
threshold_date = dataset_start + pd.Timedelta(days=14)

# Calculate start date for each product's timeseries
first_sales = df.groupby('StockCode')['InvoiceDate'].min()
product_starts = {}
for stockcode, first_sale in first_sales.items():
    if first_sale <= threshold_date:
        product_starts[stockcode] = dataset_start
    else:
        product_starts[stockcode] = first_sale

# Create variable start index
variable_index = []
for stockcode, start_date in product_starts.items():
    product_dates = pd.date_range(start=start_date, end=dataset_end.date())
    variable_index.append(pd.DataFrame({'InvoiceDate': product_dates, 'StockCode': stockcode}))
variable_index = pd.concat(variable_index, ignore_index=True)

variable_index = pd.merge(variable_index, time_idx, on='InvoiceDate', how='left')

# Create timeseries dataframe with full date range for each product
timeseries_df = pd.merge(variable_index, df, on=['InvoiceDate', 'StockCode'], how='left')

# Fill missing values for Quantity and Price
timeseries_df['Quantity'] = timeseries_df['Quantity'].fillna(0)
timeseries_df = timeseries_df.sort_values(['InvoiceDate', 'StockCode']).reset_index(drop=True)
timeseries_df['Price'] = timeseries_df.groupby('StockCode')['Price'].ffill().bfill()

# Drop 'Clean' as related engineered features will be mapped to StockCode
timeseries_df = timeseries_df.drop(columns='Clean')

# %% [markdown]
#### Calendar Features

# %%
# Create a new feature for the day of the week
timeseries_df['Day of Week'] = timeseries_df['InvoiceDate'].dt.day_name()
timeseries_df['Month'] = timeseries_df['InvoiceDate'].dt.month_name()

# Identify missing dates (Closed)
all_dates = (all_dates.to_series()).reset_index(drop=True)
missing_dates = (
    pd.DataFrame(
        all_dates[~all_dates.isin(df['InvoiceDate'].unique())]
        .reset_index(drop=True)))
missing_dates.columns = ['Date']
missing_dates['Date'] = pd.to_datetime(missing_dates['Date'])
missing_dates['Day of Week'] = missing_dates['Date'].dt.day_name()

# Create 'Operating Status' feature
timeseries_df['Is Open'] = np.where(
    timeseries_df['InvoiceDate'].isin(missing_dates['Date']), 0, 1)

# Identify holidays
christmas_2009 = pd.Series(pd.date_range(start='2009-12-24', end='2010-01-03'))
easter_2010 = pd.Series(pd.date_range(start='2010-04-02', end='2010-04-05'))
may_day_2010 = pd.Series(pd.date_range(start='2010-05-03', end='2010-05-03'))
spring_bank_2010 = pd.Series(pd.date_range(start='2010-05-31', end='2010-05-31'))
summer_bank_2010 = pd.Series(pd.date_range(start='2010-08-30', end='2010-08-30'))
christmas_2010 = pd.Series(pd.date_range(start='2010-12-24', end='2011-01-03'))
easter_2011 = pd.Series(pd.date_range(start='2011-04-22', end='2011-04-25'))
royal_wedding = pd.Series(pd.date_range(start='2011-04-29', end='2011-04-29'))
may_day_2011 = pd.Series(pd.date_range(start='2011-05-02', end='2011-05-02'))
spring_bank_2011 = pd.Series(pd.date_range(start='2011-05-30', end='2011-05-30'))
summer_bank_2011 = pd.Series(pd.date_range(start='2011-08-29', end='2011-08-29'))

holidays = pd.concat(
    [christmas_2009, christmas_2010, easter_2010, easter_2011, royal_wedding,
     may_day_2010, may_day_2011, spring_bank_2010, spring_bank_2011,
     summer_bank_2010, summer_bank_2011])

# Create 'Holiday' feature
timeseries_df['Is Holiday'] = np.where(
    timeseries_df['InvoiceDate'].isin(holidays), 1, 0)

# Create feature for countdown to Christmas break
timeseries_df = days_to_xmas_break(timeseries_df, 'InvoiceDate', 'Days to Christmas Break')

timeseries_df['Christmas 90'] = (
    timeseries_df['Days to Christmas Break'].clip(lower=0, upper=90)
    / 90)

# %% [markdown]
#### Map Product Features to StockCode

# %%
# Import product features dataset
features = pd.read_csv('data/stockcode_category_labels.csv')

# Rename columns
features = features.rename(columns={'10 Comp, 10 Clust': 'Cat1', 
                                    '29 Comp, 20 Clust': 'Cat2',
                                    '71 Comp, 40 Clust': 'Cat3',
                                    '107 Comp, 60 Clust': 'Cat4'})

# Add column name prefix to groups for model encoding accuracy
for cat in ['Cat1', 'Cat2', 'Cat3', 'Cat4']:
    features[cat] = cat + '_' + features[cat].astype(str)

# Merge features with timeseries dataframe
timeseries_df = timeseries_df.merge(features, on='StockCode', how='left')

# %% [markdown]
#### Discontinued Products

# %%
# Identify stockcodes purchased within past 6 months
six_months_ago = df['InvoiceDate'].max() - pd.Timedelta(days=180)
recent_stockcodes = df[df['InvoiceDate'] >= six_months_ago]['StockCode'].unique()

# Identify Christmas seasonal products
df['Month'] = df['InvoiceDate'].dt.month
all_stockcodes = df['StockCode'].unique()
previous_christmas_start = pd.to_datetime('2010-10-01')
previous_christmas_end = pd.to_datetime('2010-12-31')

# Calculate off-season mean and std and Chrstmas mean for each StockCode
off_season_mean = df[df['Month'] < 10].groupby('StockCode')['Quantity'].mean()
off_season_mean = off_season_mean.reindex(all_stockcodes, fill_value=0)
off_season_std = df[df['Month'] < 10].groupby('StockCode')['Quantity'].std()
off_season_std = off_season_std.reindex(all_stockcodes, fill_value=0)
christmas_mean = df[df['Month'] >= 10].groupby('StockCode')['Quantity'].mean()
christmas_mean = christmas_mean.reindex(all_stockcodes, fill_value=0)

# Identify StockCodes that had sales in the previous Christmas season
sold_previous_christmas = (
    df[(df['InvoiceDate'] >= previous_christmas_start) 
       & (df['InvoiceDate'] <= previous_christmas_end)]
    .groupby('StockCode')['Quantity'].sum() > 0
).reindex(all_stockcodes, fill_value=False)

# Identify Christmas StockCodes
is_seasonal = (
    (christmas_mean > off_season_mean + 2 * (off_season_std + 1e-5))
    & sold_previous_christmas)
christmas_stockcodes = pd.DataFrame({'StockCode': is_seasonal[is_seasonal].index})

# Map descriptions to Christmas StockCodes for review
christmas_stockcodes = christmas_stockcodes.merge(
    features[['StockCode', 'Clean']].drop_duplicates(),
    on='StockCode',
    how='left')[['StockCode', 'Clean']]

# Identify discontinued StockCodes (those that are not recent or christmas stockcodes)
discontinued_stockcodes = (
    set(df['StockCode'].unique())
    - set(recent_stockcodes)
    - set(christmas_stockcodes['StockCode'].unique()))

# Create 'Is Discontinued' feature, changing to 1 after final date of sales for each discontinued product
discontinued_dates = (
    df[df['StockCode'].isin(discontinued_stockcodes)]
    .groupby('StockCode')['InvoiceDate']
    .max()
    .reset_index()
)
discontinued_dates.columns = ['StockCode', 'Last Sale Date']
timeseries_df = timeseries_df.merge(discontinued_dates, on='StockCode', how='left')
timeseries_df['Is Discontinued'] = np.where(
    (timeseries_df['StockCode'].isin(discontinued_stockcodes)) &
    (timeseries_df['InvoiceDate'] > timeseries_df['Last Sale Date']), 1, 0)
timeseries_df = timeseries_df.drop(columns='Last Sale Date')

# %% [markdown]
##### Validation of Product Category Features on Discontinued Products

# %%
# Map 'Is Discontinued' feature to features dataframe
features['Is Discontinued'] = np.where(
    features['StockCode'].isin(discontinued_stockcodes), 1, 0)

# Select a sample of discontinued products with their descriptions and category labels
random.seed(42)
discontinued_sample = random.sample(sorted(list(discontinued_stockcodes)), 10)
disc_sample = features[
    features['StockCode'].isin(discontinued_sample)
    ][['StockCode', 'Clean', 'Cat1', 'Cat2', 'Cat3', 'Cat4']]

# Create dataframe to store potential replacements for discontinued products
replacements_df = pd.DataFrame(columns=[
    'Discontinued StockCode',
    'Discontinued Description',
    'Replacement StockCode',
    'Replacement Description'])

# For each sample, identify non-discontinued products with the same category labels
for sample in disc_sample['StockCode']:
    cat1 = disc_sample[disc_sample['StockCode'] == sample]['Cat1'].iloc[0]
    cat2 = disc_sample[disc_sample['StockCode'] == sample]['Cat2'].iloc[0]
    cat3 = disc_sample[disc_sample['StockCode'] == sample]['Cat3'].iloc[0]
    cat4 = disc_sample[disc_sample['StockCode'] == sample]['Cat4'].iloc[0]

    replacements = features[
        (features['Cat1'] == cat1) &
        (features['Cat2'] == cat2) &
        (features['Cat3'] == cat3) &
        (features['Cat4'] == cat4) &
        (features['Is Discontinued'] == 0)][['StockCode', 'Clean']]
    
    if len(replacements) > 0:
        for index, row in replacements.iterrows():
            row = {
                'Discontinued StockCode': sample,
                'Discontinued Description': disc_sample[disc_sample['StockCode'] == sample]['Clean'].iloc[0],
                'Replacement StockCode': row['StockCode'],
                'Replacement Description': row['Clean']}
            replacements_df = pd.concat([replacements_df, pd.DataFrame([row])], ignore_index=True)
    else:
        row = {
            'Discontinued StockCode': sample,
            'Discontinued Description': disc_sample[disc_sample['StockCode'] == sample]['Clean'].iloc[0],
            'Replacement StockCode': None,
            'Replacement Description': None}
        replacements_df = pd.concat([replacements_df, pd.DataFrame([row])], ignore_index=True)

# %% [markdown]
#### Price Scaled

# %%
# Scale price by StockCode Z-score
timeseries_df['Price Scaled'] = timeseries_df.groupby('StockCode')['Price'].transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 1e-5 else 0)

# %% [markdown]
#### Rename Columns and Export Final DataFrame

# %%
columns = {
    'InvoiceDate': 'invoice_date',
    'StockCode': 'stock_code',
    'Quantity': 'quantity',
    'Price': 'price',
    'Day of Week': 'day_of_week',
    'Month': 'month',
    'time_idx': 'time_idx',
    'Is Open': 'is_open',
    'Is Holiday': 'is_holiday',
    'Days to Christmas Break': 'days_to_christmas_break',
    'Christmas 90': 'christmas_break_countdown_90',
    'Clean': 'clean',
    'Cat1': 'category_level_1',
    'Cat2': 'category_level_2',
    'Cat3': 'category_level_3',
    'Cat4': 'category_level_4',
    'Is Discontinued': 'is_discontinued',
    'Price Scaled': 'price_scaled'}

for old_name, new_name in columns.items():
    timeseries_df = timeseries_df.rename(columns={old_name: new_name})

# Export final dataset for modeling
timeseries_df.to_csv('data/online_retail_II_engineered.csv', index=False)

# %% [markdown]
#### Sources
"""
Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5CG6D.
"""
# %%
