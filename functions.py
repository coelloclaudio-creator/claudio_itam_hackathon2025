import pandas as pd
from dotenv import load_dotenv
import os
from statsbombpy import sb
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
load_dotenv()

def load_player_season_stats():

    creds = {"user": "itam_hackathon@hudl.com", "passwd":  "pGwIprel"}
    
    player_season_stats = []
    for i, row in sb.competitions(creds=creds).iterrows():
        comp_id = row['competition_id']
        s_id = row['season_id']
        aux = sb.player_season_stats(competition_id=comp_id, season_id=s_id, creds=creds)
        player_season_stats.append(aux)

    player_season_stats = pd.concat(player_season_stats)
    df = player_season_stats
    return df



def add_player_age(df):
    df['season_start_year'] = df['season_name'].str[:4].astype(int)
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors = 'coerce')

    df['age'] = df['season_start_year'] - df['birth_date'].dt.year
    return df

def filter_players(df, minutes_column='player_season_minutes',
                   position_column='primary_position'):
    q1_threshold = df[minutes_column].quantile(0.25)

    df = df[(df[minutes_column] > q1_threshold) &
    (df[position_column] != 'Goalkeeper')]


    return df, q1_threshold

def find_age_group (df, age_column= 'age'):
    bins = [16, 20, 24, 27, 29, 32, float('inf')]
    labels = ['17-20', '21-24', '25-27', '28-29', '30-32', '33+']
    df['age_group'] = pd.cut(df['age'], bins= bins, labels=labels, right= True)
    return df


def prepare_final_dataset(df):
    df_clean = df[[
        'player_name','age_group', 'competition_name','season_name','season_start_year','team_name', 'primary_position',
        'player_season_obv_90', 'player_season_obv_pass_90', 'player_season_obv_shot_90',
        'player_season_obv_defensive_action_90', 'player_season_obv_dribble_carry_90', 
        'player_season_np_xg_90', 'player_season_xa_90'
    ]]

    return df_clean

def find_prev_season (row, df_clean):
    prev_season_year = row['season_start_year'] - 1
    player_name = row['player_name']

    prev_season = df_clean[(df_clean['player_name'] == player_name) & (df_clean['season_start_year'] == prev_season_year)]

    if not prev_season.empty:
        return prev_season.iloc[0] 
    else:
        return None
def find_prev_stat (row, df, stat_columns):
    prev_row = find_prev_season (row, df, id_col =  'player_name', season_col = 'season_start_year')
    if prev_row is not None:
        return prev_row [stat_columns]
    return None 

def create_pivot_df(df, stat_columns):
    player_season_dict = {}  # create dictionary
    for _, current_row in df.iterrows():  # tuple. name, season stats
        player_name = current_row['player_name']
        season = current_row['season_start_year']

        if player_name not in player_season_dict:
            player_season_dict[player_name] = {}
        player_season_dict[player_name][season] = {
            stat: current_row[stat] for stat in stat_columns
        }

    records = []
    for _, current_row in df.iterrows():
        player_name = current_row['player_name']
        current_season = current_row['season_start_year']
        prev_season = current_season - 1

        if player_name in player_season_dict and prev_season in player_season_dict[player_name]:
            prev_stats = player_season_dict[player_name][prev_season]
            row_data = {
                'player_name': player_name,
                'season_start_year': current_season,
                'prev_season_year': prev_season
            }

            # If original df contains team info, carry it over
            if 'team_name' in current_row.index:
                row_data['team_name'] = current_row['team_name']
            if 'team_name_prev' in current_row.index:
                row_data['team_name_prev'] = current_row['team_name_prev']

            # Current stats
            for stat in stat_columns:
                row_data[stat] = current_row[stat]

            # Previous stats
            for stat in stat_columns:
                row_data[stat + '_prev'] = prev_stats[stat]

            records.append(row_data)

    pivot_df = pd.DataFrame(records)
    return pivot_df

    # Choose dedupe keys adaptively:
    dedupe_subset = ['player_name', 'season_start_year']
    if 'team_name' in pivot_df.columns:
        dedupe_subset.append('team_name')
    if 'team_name_prev' in pivot_df.columns:
        dedupe_subset.append('team_name_prev')

    # Drop exact duplicate rows per player-season (and team context if present)
    pivot_df = pivot_df.drop_duplicates(subset=dedupe_subset, keep='first')

    return pivot_df

def find_stat_diff (df, stat_columns):
    for stat in stat_columns:
        stat_prev = stat + '_prev'
        stat_diff = stat + '_diff'
        df[stat_diff] = df[stat] - df[stat_prev]

    return df

def load_team_season_stats():
    creds = {"user": "itam_hackathon@hudl.com", "passwd":  "pGwIprel"}
    
    team_season_stats = []
    for i, row in sb.competitions(creds=creds).iterrows():
        comp_id = row['competition_id']
        s_id = row['season_id']
        aux = sb.team_season_stats(competition_id=comp_id, season_id=s_id, creds=creds)
        team_season_stats.append(aux)
    
    team_season_stats = pd.concat(team_season_stats)
    df_team = team_season_stats

    pd.set_option('display.max_columns', None)

    return df_team

def prepare_team_intention(df_team):
    df_intention = df_team[[
        'team_name', 'season_name', 'team_season_possession', 'team_season_pace_towards_goal', 
        'team_season_box_cross_ratio', 'team_season_ppda', 'team_season_counter_attacking_shots_pg', 
        'team_season_aggressive_actions_pg', 'team_season_pressures_pg', 
        'team_season_deep_progressions_pg', 'team_season_np_shots_pg'
    ]]

    return df_intention

def prepare_team_performance(df_team):
        df_performance = df_team[['team_name', 'season_name','team_season_xgd', 'team_season_np_xg_pg', 'team_season_op_xg_pg', 
                          'team_season_obv_pg', 'team_season_obv_conceded_pg']]
        return df_performance

def scale_dataframes(df_intention, df_performance):
    scaler = StandardScaler()

    df_intention_scaled = pd.DataFrame(
        scaler.fit_transform(df_intention.select_dtypes(include=['float64', 'int64'])),
        columns=df_intention.select_dtypes(include=['float64', 'int64']).columns,
        index=df_intention.index
    )

    df_performance_scaled = pd.DataFrame(
        scaler.fit_transform(df_performance.select_dtypes(include=['float64', 'int64'])),
        columns=df_performance.select_dtypes(include=['float64', 'int64']).columns,
        index=df_performance.index
    )

    return df_intention_scaled, df_performance_scaled

def apply_pca_intention(df_intention_scaled, df_intention):
    """
    Applies PCA on the scaled intention DataFrame and returns a DataFrame
    with the minimum number of components needed to retain ≥ 80% of variance.
    """
    X = df_intention_scaled

    # Fit PCA to find number of components
    pca = PCA()
    pca.fit(X)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1

    # Refit PCA with optimal number of components
    pca = PCA(n_components=n_components_80)
    X_pca = pca.fit_transform(X)

    # Create PCA DataFrame
    df_intention_pca = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(n_components_80)],
        index=df_intention_scaled.index
    )

    df_intention_pca = pd.concat(
        [df_intention[['team_name', 'season_name']], df_intention_pca],
        axis=1
    )

    return df_intention_pca

def apply_offensive_pca(df_performance_scaled, df_performance):
    """
    Applies PCA on offensive features (xG and OBV) and returns a DataFrame
    with the offensive PCA score and identifiers (team_name, season_name).
    """
    offensive_features = ['team_season_np_xg_pg', 'team_season_obv_pg']
    X_offensive = df_performance_scaled[offensive_features]

    # Step 1: Fit PCA
    pca_off = PCA()
    X_off_pca = pca_off.fit_transform(X_offensive)

    # Step 2: Determine number of components to retain ≥ 80% variance
    cumulative_variance = np.cumsum(pca_off.explained_variance_ratio_)
    n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1

    # Step 3: Refit PCA with 1 component
    pca_off = PCA(n_components=1)
    X_off_pca = pca_off.fit_transform(X_offensive)

    # Step 4: Create PCA DataFrame
    df_offensive_score = pd.DataFrame(
        X_off_pca,
        columns=[f'Off_PC{i+1}' for i in range(n_components_80)],
        index=df_performance_scaled.index
    )

    # Step 5: Add identifiers back (keeping your concat logic)
    df_offensive_score = pd.concat(
        [df_performance[['team_name', 'season_name']], df_offensive_score],
        axis=1
    )

    return df_offensive_score

def apply_defensive_pca(df_performance_scaled, df_performance):
    defensive_features = ['team_season_obv_conceded_pg', 'team_season_op_xg_pg']
    X_defensive = df_performance_scaled[defensive_features]

    pca_def = PCA()
    X_def_pca_temp = pca_def.fit_transform(X_defensive)
    cumulative_variance = np.cumsum(pca_def.explained_variance_ratio_)
    n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
    

    pca_def = PCA(n_components= 1)
    X_def_pca = pca_def.fit_transform(X_defensive)

    #New data frame
    df_defensive_score = pd.DataFrame(
    X_def_pca,
    columns=['Def_PC1'],
    index=df_performance_scaled.index
)

    df_defensive_score = pd.concat([df_performance[['team_name', 'season_name']], df_defensive_score], axis = 1) 
    return df_defensive_score

def cluster_intention(df_intention_pca):
    """
    Applies KMeans clustering on the first four PCA intention components
    and returns a DataFrame with team_name, season_name, and the cluster label.
    """
    X_intention = df_intention_pca[['PC1', 'PC2', 'PC3', 'PC4']]

    # Fit KMeans with 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_intention_pca['intention_cluster'] = kmeans.fit_predict(X_intention)

    df_cluster = df_intention_pca[['team_name', 'season_name', 'intention_cluster']]

    return df_cluster

def merge_final_scores(df_offensive_score, df_defensive_score, df_cluster):
    """
    Merges offensive, defensive, and intention cluster DataFrames into one final DataFrame.
    """
    df_scores = pd.merge(
        df_offensive_score,
        df_defensive_score,
        on=['team_name', 'season_name'],
        how='inner'
    )

    df_final = pd.merge(
        df_scores,
        df_cluster[['team_name', 'season_name', 'intention_cluster']],
        on=['team_name', 'season_name'],
        how='inner'
    )
    df_final = df_final.drop_duplicates(subset=['team_name', 'season_name'])
    return df_final

def merge_pivot_with_scores(pivot_df, df_final):
    """
    Merges pivot_df with df_final twice:
    1. On current team (team_name, season_name)
    2. On previous team (team_name_prev, season_name_prev)
    """
    # Merge current team data
    pivot_df = pd.merge(
        pivot_df,
        df_final,
        on=['team_name', 'season_name'],
        how='left',
        suffixes=('', '_current_team')
    )

    # Merge previous team data
    pivot_df = pd.merge(
        pivot_df,
        df_final,
        left_on=['team_name_prev', 'season_name_prev'],
        right_on=['team_name', 'season_name'],
        suffixes=('', 'prev_team')
    )
    pivot_df = pivot_df.drop(columns=['team_nameprev_team', 'season_nameprev_team'])
    return pivot_df

def create_template_matrix(df):
    origin_clusters = sorted(df['intention_clusterprev_team'].dropna().unique())
    destination_clusters = sorted(df['intention_cluster'].dropna().unique())

    template_matrix = pd.DataFrame(np.nan, index=origin_clusters, columns=destination_clusters, dtype=float)

    return template_matrix

def stat_columns():
    stat_columns = ['player_season_obv_90', 'player_season_obv_pass_90', 'player_season_obv_shot_90',
                'player_season_obv_defensive_action_90', 'player_season_obv_dribble_carry_90', 
                'player_season_np_xg_90', 'player_season_xa_90']
    return stat_columns

def populate_matrix(df, stat_columns, template_matrix):
    matrices = {}

    for stat in stat_columns:
        stat_prev = stat + '_prev'

        # 1. Group by cluster transitions and aggregate totals
        grouped = df.groupby(['intention_clusterprev_team', 'intention_cluster'])[[stat, stat_prev]].sum()

        # 2. Compute percentage change
        grouped['pct_change'] = ((grouped[stat] - grouped[stat_prev]) / grouped[stat_prev]) * 100

        # 3. Build transition dictionary
        transition_dict = {
            (origin, destination): round(row['pct_change'], 4)
            for (origin, destination), row in grouped.iterrows()
            if not pd.isnull(row['pct_change']) and np.isfinite(row['pct_change'])
        }

        # 4. Fill matrix
        matrix = template_matrix.copy()
        for (origin, destination), value in transition_dict.items():
            if origin in matrix.index and destination in matrix.columns:
                matrix.at[origin, destination] = value

        matrices[stat] = matrix

    return matrices


def calculate_transition_score(pivot_df):
    """
    Calculates transition_score based on the difference between current and previous team
    offensive and defensive PC1 scores.
    """
    conditions = [
        (pivot_df['Off_PC1'] + pivot_df['Def_PC1']) > (pivot_df['Off_PC1prev_team'] + pivot_df['Def_PC1prev_team']),
        (pivot_df['Off_PC1'] + pivot_df['Def_PC1']) < (pivot_df['Off_PC1prev_team'] + pivot_df['Def_PC1prev_team'])
    ]
    choices = [1, -1]

    pivot_df['transition_score'] = np.select(conditions, choices, default=0)

    return pivot_df

def get_rows_by_name(pivot_df, name, column='player_name', ):
    return pivot_df[pivot_df[column].str.contains(name, case=False, na=False)]

def plot_intention_clusters(df_intention_pca):
    """
    Plots the PCA intention space using PC1 and PC2,
    coloring the points by their intention_cluster label.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df_intention_pca['PC1'],
        df_intention_pca['PC2'],
        c=df_intention_pca['intention_cluster'],
        cmap='tab10',
        alpha=0.7
    )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Intention Clusters (PC1 vs PC2)')
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.show()

def style_matrix(df):
    """
    Styles a DataFrame to highlight positive and negative values:
    - Green background for positive
    - Red background for negative
    - No color for NaN or zero
    Formats values to 2 decimal places.
    """
    def color_values(val):
        if pd.isna(val) or val == 0:
            return ''
        elif val > 0:
            return 'background-color: green; color: white;'
        else:  # val < 0
            return 'background-color: red; color: white;'

    return df.style.applymap(color_values).format("{:.2f}")


from PIL import Image
import streamlit as st
st.header("Glossary")
st.subheader('What was done in this project?')

paragraph = """Voy"""

st.subheader('Contact the App Developer')

st.write('Claudio Coello. Data science student at the University of Florida.')
st.markdown('LinkedIn: [Claudio Coello](www.linkedin.com/in/claudiocoello)')
st.write('Email: coelloclaudio@gmail.com')

st.subheader('Data')
st.markdown("""
  
  Thanks Hudl Statsbomb for opening the data to the public via: https://github.com/statsbomb/statsbombpy.""")
image = Image.open("./assets/hudl-statsbomb-logo-default.png")
st.image(image, caption='Hudl Statsbomb')