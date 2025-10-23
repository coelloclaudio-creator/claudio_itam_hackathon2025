from functions import (
    load_player_season_stats, add_player_age, filter_players, find_age_group, prepare_final_dataset , find_prev_season, find_prev_stat, create_pivot_df, find_stat_diff,
    load_team_season_stats, prepare_team_intention, prepare_team_performance, scale_dataframes, apply_pca_intention, apply_offensive_pca,
    apply_defensive_pca, cluster_intention, merge_final_scores, merge_pivot_with_scores, create_template_matrix, stat_columns, populate_matrix,
    calculate_transition_score, get_rows_by_name, plot_intention_clusters, style_matrix)

from dotenv import load_dotenv
load_dotenv()
import os


import pandas as pd
from statsbombpy import sb
import matplotlib.pyplot as plt
import streamlit as st

st.markdown("# Using Kmeans Clustering of Team Playing Styles to Predict Player Performance After Transfers in Liga MX")
st.subheader('2025 ISAC Sports Analytics Hackathon.')
st.subheader('A project by Claudio Coello, University of Florida, Class of 2028')
st.write('This app displays the results of the analysis, for further details, please visit the glossary and the Readme file in the repository.')

df = load_player_season_stats()

df = add_player_age(df)


df = find_age_group(df)

df, q1_threshold = filter_players(df)
  # Should be <class 'pandas.core.frame.DataFrame'>


final_stat_columns = ['player_season_obv_90', 'player_season_obv_pass_90', 'player_season_obv_shot_90',
        'player_season_obv_defensive_action_90', 'player_season_obv_dribble_carry_90', 
        'player_season_np_xg_90', 'player_season_xa_90', 
        'age_group', 'competition_name', 'season_name', 'team_name', 'primary_position']

pivot_df = create_pivot_df(df, final_stat_columns)
st.subheader("This Dataframe shows a player's season stats compared to the previous season's stats")
st.write("We are using statsbomb's player_season_stats")
st.dataframe(pivot_df.head(n=10))


team_df = load_team_season_stats()
st.subheader("We are now using Statsbomb's team season stats dataset to create team intention clusters and offensive/defensive scores.")
st.write("What is an intention cluster? It is a grouping of teams based on their playing style, which we derive from various metrics ")
st.write("These metrics do not measure performance, but how a team attempts to play the game.")
st.dataframe(team_df.head(n=10))
df_intention = prepare_team_intention(team_df)
df_performance = prepare_team_performance(team_df)

df_intention_scaled, df_performance_scaled = scale_dataframes(df_intention, df_performance)
df_intention_pca = apply_pca_intention(df_intention_scaled, df_intention)
df_offensive_score = apply_offensive_pca(df_performance_scaled, df_performance)
df_defensive_score = apply_defensive_pca(df_performance_scaled, df_performance)
df_cluster = cluster_intention(df_intention_pca)



st.subheader("Intention Clusters Visualized in PCA Space")
st.write("We picked 5 clusters based on the amount of teams in Liga MX and the shape of this PCA plot.")
plot_intention_clusters(df_intention_pca)
st.pyplot(plt)

df_final_scores = merge_final_scores(df_offensive_score, df_defensive_score, df_cluster)
st.subheader("This dataframe displays each team's offensive score, defensive score, and intention cluster for each season.")
st.write('The offensive and defensive scores are calculated using PCA on 2 relevant metrics, obv per game, and xG per game (conceded for defense).')
st.write("Anyone who has watched a little bit of Liga MX can see how offensive and defensive scores make sense. Look at Cruz Azul's 24/25 campaign!")

st.dataframe(df_final_scores.head(10))

# Display pivot df again, now w scores merged in
st.subheader('The player level df we previously displayed has now been updated to include team offensive/defensive scores and intention cluster.')
st.write('Notice how each player has these 3 data points for their current and previous seasons. The last column shows if the transition was to a better or worse team. If the player did not change teams, the score signals his team getting better or worse the next season')
pivot_df = merge_pivot_with_scores(pivot_df, df_final_scores)
pivot_df = calculate_transition_score(pivot_df)
st.dataframe(pivot_df.head(10))

#Here i need to use the player selecting widget so any player's transitions can be analyzed.

player_query = st.text_input("Type a player's name to see his apperances in this dataframe (Case sensitive, include accents for spanish names)")

if player_query:
    player_rows = get_rows_by_name(pivot_df, player_query)
    st.subheader(f"Transition data for players matching: {player_query}")
    st.dataframe(player_rows)

template_matrix = create_template_matrix(pivot_df)
matrix_stat_columns = stat_columns = ['player_season_obv_90', 'player_season_obv_pass_90', 'player_season_obv_shot_90',
                'player_season_obv_defensive_action_90', 'player_season_obv_dribble_carry_90', 
                'player_season_np_xg_90', 'player_season_xa_90']
matrices = populate_matrix(pivot_df, matrix_stat_columns, template_matrix)

st.subheader("Now, we move on to crown jewel of this analysis: The Transition Matrix.")
st.write("After looking at an individual's performance, you can use this table to see how the average player that transitioned from his current cluster of teams performed in clusters they moved to.")
st.write("This is the promise of this project. Finding predictive value in team playing styles and how players adapt to them.")
st.write("If you like a player's statistics, but don't know if his playing style will fit América, use this table to find which clusters have historically transitioned well to América's cluster.")
st.write('Note: rows are origin, columns are destination clusters. The values are changes in percentage points for the selected statistic.')
st.write("Select a statistic to display in the transition matrix:")

selected_stat = st.selectbox("Select a stat to view its transition matrix:", matrix_stat_columns)

if selected_stat in matrices:
    st.subheader(f"Transition Matrix for {selected_stat}")
    styled_matrix = style_matrix(matrices[selected_stat])
    st.dataframe(styled_matrix)

st.write("Note: Only 3 transitions worth of data were provided for this competition, so the matrices may be missing some cluster-to-cluster transitions as they did not pass the threshold of at least 5 changes. The idea is for more seasons to be included to improve the analysis.")
st.write("There is also one major hole in the analysis. Our league is divided into Apertura and Clausura seasons, however, Statsbomb does not partition the data.")
st.write("Having a more complete database would drastically improve the execution of this idea.")

st.subheader("Thank you for checking out my project, please direct yourself to the glossary for further explanations.")
st.write("...")
