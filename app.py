import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(layout="wide")
st.title('FIFA 20 Player Clustering Analysis')

# --- 1. Data Loading and Preprocessing ---

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)

    # Standardize text columns (strip whitespace)
    text_cols = df.select_dtypes(include='object').columns.tolist()
    df[text_cols] = df[text_cols].apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

    # Money conversion: create numeric columns for value & wage
    def convert_money(series):
        s = series.astype(str).str.replace('€','').str.replace(',','').str.strip()
        mask_M = s.str.endswith('M', na=False)
        mask_K = s.str.endswith('K', na=False)
        base = s.str.rstrip('MK').replace('', np.nan)
        num = pd.to_numeric(base, errors='coerce')
        num = num * np.where(mask_M, 1e6, np.where(mask_K, 1e3, 1.0))
        return num

    if 'value_eur' in df.columns:
        df['value_num'] = convert_money(df['value_eur'])
    elif 'value' in df.columns:
        df['value_num'] = convert_money(df['value'])

    if 'wage_eur' in df.columns:
        df['wage_num'] = convert_money(df['wage_eur'])
    elif 'wage' in df.columns:
        df['wage_num'] = convert_money(df['wage'])

    # Ensure numeric basic columns exist and convert
    for c in ['age','overall','potential']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Impute simple missing values:
    # numeric columns -> fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # categorical columns -> fill with mode for small set
    for c in ['nationality','player_positions','preferred_foot']:
        if c in df.columns and df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    return df

# For deployment, assume the data file is in the same directory as app.py
dataset_path = 'players_20 (1).csv'

if not os.path.exists(dataset_path):
    st.error(f"Dataset not found at {dataset_path}. Please make sure 'players_20 (1).csv' is in the same directory as app.py.")
    st.stop()

df = load_data(dataset_path)

# --- 2. Clustering Preparation ---

skill_candidates = [
    'attacking_crossing','attacking_finishing','attacking_heading_accuracy','attacking_short_passing','attacking_volleys',
    'skill_dribbling','skill_curve','skill_fk_accuracy','skill_long_passing','skill_ball_control',
    'movement_acceleration','movement_sprint_speed','movement_agility','movement_reactions','movement_balance',
    'power_shot_power','power_jumping','power_stamina','power_strength','power_long_shots',
    'mentality_aggression','mentality_interceptions','mentality_positioning','mentality_vision','mentality_penalties','mentality_composure',
    'defending_marking','defending_standing_tackle','defending_sliding_tackle'
]

# Select skills that exist in the dataset
skills = [c for c in skill_candidates if c in df.columns]

# Fallback to aggregated skill columns if detailed skills are not enough
if len(skills) < 4:
    fallback = ['pace','shooting','passing','dribbling','defending']
    skills = [c for c in fallback if c in df.columns]

X = df[skills].apply(pd.to_numeric, errors='coerce').fillna(df[skills].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. KMeans Clustering ---

# Use the 'k' determined in the notebook or a default
k = 2 # From the notebook, best_k was 2

@st.cache_data
def perform_kmeans(X_scaled, k_clusters):
    km = KMeans(n_clusters=k_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    return labels, km

labels_km, kmeans_model = perform_kmeans(X_scaled, k)
df['cluster_kmeans'] = labels_km

# --- Streamlit App Layout ---

st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', ['Cluster Overview', 'Player Search', 'Cluster Visualizations'])

if page == 'Cluster Overview':
    st.header('KMeans Cluster Characteristics')
    cluster_profile = df.groupby('cluster_kmeans')[skills].mean().round(2)
    st.write("**Mean skill attributes for each KMeans cluster:**")
    st.dataframe(cluster_profile.T)

    st.subheader('Interpretation of Clusters')
    st.write(f"With k={k} clusters, we typically see distinct player types:")
    for i in range(k):
        st.markdown(f"**Cluster {i}:**")
        top_skills = cluster_profile.loc[i].nlargest(3)
        bottom_skills = cluster_profile.loc[i].nsmallest(3)
        st.write(f"*   **Top Skills:** {', '.join([f'{s} ({v})' for s, v in top_skills.items()])}")
        st.write(f"*   **Weaknesses:** {', '.join([f'{s} ({v})' for s, v in bottom_skills.items()])}")

    st.subheader('Top Players per Cluster (by Overall Rating)')
    display_cols = ['short_name','overall','potential','nationality','club','value_num','wage_num']
    for cl in sorted(df['cluster_kmeans'].unique()):
        st.write(f"**Cluster {cl} top 5 players:**")
        top_players = df[df['cluster_kmeans'] == cl].sort_values('overall', ascending=False)[display_cols].head(5)
        st.dataframe(top_players)

elif page == 'Player Search':
    st.header('Search for Players by Cluster or Name')

    cluster_filter = st.sidebar.selectbox('Filter by Cluster', ['All'] + sorted(df['cluster_kmeans'].unique().tolist()))
    player_name_search = st.sidebar.text_input('Search by Player Name (e.g., Messi, Ronaldo)')

    filtered_df = df.copy()
    if cluster_filter != 'All':
        filtered_df = filtered_df[filtered_df['cluster_kmeans'] == cluster_filter]
    if player_name_search:
        filtered_df = filtered_df[filtered_df['short_name'].str.contains(player_name_search, case=False, na=False)]

    st.write(f"Showing {len(filtered_df)} players:")
    st.dataframe(filtered_df[['short_name', 'overall', 'potential', 'club', 'nationality', 'cluster_kmeans'] + skills])

elif page == 'Cluster Visualizations':
    st.header('Cluster Visualizations')

    st.subheader('Skill Attributes by Cluster (Heatmap)')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_profile, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Mean Skill Attributes by KMeans Cluster')
    ax.set_xlabel('Skill Attribute')
    ax.set_ylabel('KMeans Cluster')
    st.pyplot(fig)

    st.subheader('Player Value Distribution by Cluster')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='cluster_kmeans', y='value_num', data=df, palette='viridis', ax=ax)
    ax.set_title('Distribution of Player Value by KMeans Cluster')
    ax.set_xlabel('KMeans Cluster')
    ax.set_ylabel('Player Value (EUR)')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.subheader('Overall Rating Distribution by Cluster')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='overall', hue='cluster_kmeans', kde=True, palette='tab10', ax=ax)
    ax.set_title('Overall Rating Distribution by Cluster')
    ax.set_xlabel('Overall Rating')
    ax.set_ylabel('Number of Players')
    st.pyplot(fig)
