from PIL import Image
import streamlit as st
st.header("Glossary")
st.subheader('What was done in this project?')

paragraph = """The first step was to use the player_season_stats from Statsbombpy to create a personalized data frame. What we did was take every row in this data frame, which represents one player’s statistical output over a full season, and try to find that same player’s previous season’s row. For example, we need to take Alvaro Fidalgo’s 2023/24 and 2022/23 seasons and add these to our new data frame. 
The columns are organized in this order:
Fidalgo’s 2023/24 stats.
Fidalgo’s 2022/23 stats.
Very importantly, the difference between the first and second columns for each stat. In this project, we call the most recent of the two years “current”, and the former “previous.”

We do this for every player and create a big database with each player's current and previous seasons.

In a world where América signs players from every Liga MX team every year, we would have enough information to see which teams send over high-performing players to América, and which teams we should stop signing from. But this is not the case, in fact América has only signed a couple of players in the last 5 seasons, and some teams have not sent a player to América in over a decade. This is the hole our project is trying to fill. 

In order to do this, we started by using Statsbombpy’s team_season_stats and curated a selection of stats that tell us about the intention and the means a team employs when playing their games. These stats include:
 'Team_season_possession'
 'Team_season_pace_towards_goal'
 'team_season_box_cross_ratio'
 'Team_season_ppda'
 'Team_season_counter_attacking_shots_pg'
 'Team_season_aggressive_actions_pg'
  'Team_season_pressures_pg'
 'Team_season_deep_progressions_pg'
 'Team_season_np_shots_pg'

We also picked 4 key stats that explain teams’ performance during a season. 2 explain their offensive output, and 2 their defensive prowess. These are:
'team_season_np_xg_pg', 'team_season_op_xg_pg', 
                          'team_season_obv_pg', 'team_season_obv_conceded_pg

We then conducted a Principal Component Analysis on these 2 groups of statistics separately. 

For the intention group, we found that 4 PCs explained 80% of the variance in the statistics, and we used these 4 to perform a KMeans clustering operation.
The 4 PCs were plotted, and using the visualizations and knowledge that there are 18 teams in Liga MX, the number of clusters was set at 5 (about 3-4 teams per cluster per year). It is important to note that a team might change intention clusters on a year to year basis, due to a change in strategy or league-wide trends.

For the offensive and defensive statistics, 80% of the variance was explained by 1 PC, which we will call their offensive or defensive score. We manually cross checked these scores with their positions on the Liga MX table, and there was strong correlation between the two (eg. the highest scores for 2024-2025 coincided with the team that earned the most league points). 

After having these two, we merged them to create a small dataframe that displayed each team, the season, their defensive and offensive scores, and the cluster they belonged to. We then added these key details to the original player level dataframe.

This is the key step in this project. To follow the example, we now have Fidalgo, who played for América in the 2 seasons aforementioned. In both seasons, his team belonged to cluster 2. However, multiple other players changed teams, or their teams changed clusters, so we will collect all of the instances of identical transitions (eg. Cluster 0 -> Cluster 2).

Once we have this collection of transitions, we will add up all of the statistical outputs from the previous cluster and the current cluster, and perform this operation to find the percentage change in performance after a transition:

Sum of Total Output Current - Sum of Total Output Previous/ (Sum of Total Output Previous) * 100.

With this, we can say for example:
“The players that moved from Cluster 1 to Cluster 0 had a 20% dip in xG per 90 performance during their first season in their new cluster (often meaning new team).”

The idea is to use this information to sign or scout players who currently play at teams belonging to clusters that transition well to América’s cluster. This goes one step further than a simple “this player has performed well, let’s sign him and see how he does in our scheme”. The clustering is an attempt to adjust to schematic differences and changes in team level.

However, this project has a lot of holes, and it could be improved with some changes and more time to work on it. There are 3 main ways this project could be improved with some adjustments to the database and outside help.

First of all, Statsbomb does not partition Liga MX stats into apertura and clausura, which leaves a lot of holes and unknowns in this analysis. If we had access to an accurate database, there would be a lot more transitions counted and more granular data to work with.

Second, a data base including 10 or 15 years of league play would greatly increase the scope of the analysis, and we could even see how teams have changed their playing styles over the years.

Finally, access to more advanced scouting data, such as physical combine results, projected transfer fees, and even previous data from players that have played in other competitions would help better predict how a player might perform at América. 

This is meant to be a tool to increase knowledge about signings in the transfer market, but a responsible scouting department would not use it as the end-all-be-all. A team needs to cross check the clustering data with their own “eye test” to see if the players our matrix predicts would perform well would actually fit their team needs.
"""
st.markdown(paragraph)

st.subheader('Contact the App Developer')

st.write('Claudio Coello. Data science student at the University of Florida.')
st.markdown('LinkedIn: [Claudio Coello](www.linkedin.com/in/claudiocoello)')
st.write('Email: coelloclaudio@gmail.com')

st.subheader('Data')
st.markdown("""
  
  Thanks Hudl Statsbomb for opening the data to the public via: https://github.com/statsbomb/statsbombpy.""")
image = Image.open("assets/hudl-statsbomb-logo-default.png")
st.image(image, caption='Hudl Statsbomb')

