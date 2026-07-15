import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px

def count_word_frequencies(text):
    """Function to count the frequency of each word in a given text data.
    The input can be a Series or a list of strings."""

    vocab_freq = {}
    for t in text:
        t = str(t)
        for word in t.split():
            if word in vocab_freq:
                vocab_freq[word] += 1
            else:
                vocab_freq[word] = 1
    return vocab_freq


def basic_text_clean(text):
    """Function to perform basic text cleaning by converting to lowercase,
    removing commas, 'and', and extra whitespace."""

    text = text.astype(str)
    output = pd.Series(text)
    output = output.str.lower()
    output = output.str.replace(',', ' ', regex=False)
    output = output.str.replace(' and ', ' ', regex=False)
    output = output.str.strip().str.replace(r'\s+', ' ', regex=True)
    return output



def assign_category(description, keywords):
    """Function to assign category based on keywords in the description.
    If multiple categories are found, they will be concatenated together.
    If no categories are found, 'unknown' will be returned."""

    categories = []
    for category, keywords in keywords.items():
        if any(keyword in description for keyword in keywords):
            categories.append(category)
    
    if len(categories) == 1:
        return categories[0]
    elif len(categories) > 1:
        return ' '.join(categories)
    else:
        return 'unknown'


def override(df, category, keywords):
    """Function to override category based on specific keywords if
    multiple categories are present. Requires columns 'Product Category Count',
    'Clean', and 'Product Category' in the DataFrame."""

    if df['Product Category Count'] > 1:
        description = df['Clean']
        if any(word in description for word in keywords):
            return category  
    return df['Product Category']


def days_to_xmas_break(df, date_column, column_name):
    """Function to calculate the number of days until the next Christmas break
    (starting on December 24th) for each date in the specified column of a DataFrame."""

    def get_xmas_break(date):
        next_xmas = pd.Timestamp(year=date.year, month=12, day=24)
        if date > next_xmas:
            next_xmas = pd.Timestamp(year=date.year + 1, month=12, day=24)
        return next_xmas    

    next_xmas_break = df[date_column].apply(get_xmas_break)
    df[column_name] = (next_xmas_break - df[date_column])
    df[column_name] = df[column_name].dt.days
    return df

def plot_kmeans_tsne_3d(df, clean_col, components_df, n_comp, n_clust):
    components = components_df[:, :n_comp]
    kmeans = KMeans(n_clusters=n_clust, random_state=42)
    clusters = kmeans.fit_predict(components)

    tsne = TSNE(n_components=3, random_state=42)
    tsne_projections = tsne.fit_transform(components)

    cluster_col_name = f'{n_comp} Comp, {n_clust} Clust'
    df[cluster_col_name] = clusters
    df[f'{cluster_col_name}_str'] = clusters.astype(str)

    df['TSNE-1'] = tsne_projections[:, 0]
    df['TSNE-2'] = tsne_projections[:, 1]
    df['TSNE-3'] = tsne_projections[:, 2]

    palette = sns.color_palette('husl', n_colors=n_clust).as_hex()

    fig = px.scatter_3d(
        df,
        x='TSNE-1',
        y='TSNE-2',
        z='TSNE-3',
        color=f'{cluster_col_name}_str',
        color_discrete_sequence=palette,
        labels={clean_col: 'Description', cluster_col_name: 'Cluster'},
        hover_data={
            cluster_col_name: True,
            clean_col: True,
            'TSNE-1':False,
            'TSNE-2':False,
            'TSNE-3':False,
            f'{cluster_col_name}_str':False},
        title=("3D t-SNE Projection, Components: {}, Clusters: {}".format(n_comp, n_clust)))

    fig.update_layout(
        # height=600,
        # width=900,
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False)

    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.show()