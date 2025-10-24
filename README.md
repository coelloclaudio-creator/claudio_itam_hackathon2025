# ISAC 2025 Hackathon, Claudio Coello
Project for the player recommendation hackathon in the 2025 ITAM Sports Analytics Conference.

# Order of tasks performed:

## -1. Loading data from Statsbomb and processing dataframes for analysis
## -2. Selecting statistics
## -3. Principal Component Analysis and KMeans Clustering
## -4. Building the cluster transition matrix
## -5. Deploying a streamlit app

1. The data was loaded using the credentials provided by ITAM for this hackathon. All of the data used in this project is not avalaible as open data from statsbombpy. The dataframes were created using pandas, and they use statsbomb's naming conventions.

2. This project could be recreated using different statistics for the intention clustering or for the matrices, however, the names of the statistics are hard coded into lists called stat_columns. If more data is available, appending these lists could increase the scope of the project.

3. The PCA and KMeans clustering were run by using scikit-learn, a python library intended for machine learning. Installing this into your dependency is essential in order to perform this project. The clustering was done in order to assign every team in Liga MX to one cluster that best fit their playing style. We grouped teams with similar playing styles together, and 5 different clusters were created in this process.
Each team was assigned an offensive and defensive score based on a PCA of xG output and OBV per game.

4. A transition matrix (pandas dataframe) was used to observe changes in statistical output in pairs of player seasons. These were strictly back to back seasons (eg Jesus Orozco 2023-2024, Jesus Orozco 2024-2025). These were grouped by their current and previous cluster. To continue our example, Orozco's Chivas were part of intention cluster 3, and then he moved to Cruz Azul, who belonged to Cluster 0 in the 2025 clausura season. All of the identical transitions were grouped together to create a matrix for each stat selected.

5. A
 
