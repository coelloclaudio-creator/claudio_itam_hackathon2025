# Claudio Coello's project for the ISAC 2025 HAackathon.
Project for the player recommendation hackathon in the 2025 ITAM Sports Analytics Conference.

### LinkedIn: www.linkedin.com/in/claudiocoello
### Contact me at coelloclaudio@gmail.com
### Data Science student and the University of Florida.

# Order of tasks performed:

## -1. Loading data from Statsbomb and processing dataframes for analysis
 The data was loaded using the credentials provided by ITAM for this hackathon. All of the data used in this project is not avalaible as open data from statsbombpy. The dataframes were created using pandas, and they use statsbomb's naming conventions.

## -2. Selecting statistics
This project could be recreated using different statistics for the intention clustering or for the matrices, however, the names of the statistics are hard coded into lists called stat_columns. If more data is available, appending these lists could increase the scope of the project.

## -3. Principal Component Analysis and KMeans Clustering
The PCA and KMeans clustering were run by using scikit-learn, a python library intended for machine learning. Installing this into your dependency is essential in order to perform this project. The clustering was done in order to assign every team in Liga MX to one cluster that best fit their playing style. We grouped teams with similar playing styles together, and 5 different clusters were created in this process.
Each team was assigned an offensive and defensive score based on a PCA of xG output and OBV per game.

## -4. Building the cluster transition matrix
 A transition matrix (pandas dataframe) was used to observe changes in statistical output in pairs of player seasons. These were strictly back to back seasons (eg Jesus Orozco 2023-2024, Jesus Orozco 2024-2025). These were grouped by their current and previous cluster. To continue our example, Orozco's Chivas were part of intention cluster 3, and then he moved to Cruz Azul, who belonged to Cluster 0 in the 2025 clausura season. All of the identical transitions were grouped together to create a matrix for each stat selected.


## -5. Deploying a streamlit app

This app is divided into 2 pages. 
Page 1: shows a visualization of the tools used during the project
Page 2: glossary that explains key terms and methodology.

This project requires Python 3.10+. After cloning the repository and moving into the project root:

0.1 Create a virtual environment

`conda create -n football_env python=3.10`

`conda activate football_env`

0.2 Install dependencies

`pip install -r requirements.txt`

# Step 1) Run the App:

`streamlit run App.py`

Wait a few seconds. You’ll see a local URL in your terminal that you can click to open the app in your browser.

# Step 2) Use the app:

Inspect the sample dataframes to see their content, search for players (case and accent senstivie), and change the statistic displayed in the cluster transition matrix (only a couple are included in the app due to efficiency purposes but could be changed if the function code is changed.)

Visit page 2 to read the glossary on key terms, definitions, and formulas.






