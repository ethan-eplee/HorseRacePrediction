import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import pickle

# Load the model
def load_model():
    with open('saved_steps.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()

model = data['model']
le_horse_name = data['le_horse_name']
le_jockey = data['le_jockey']
df_train = data['dataset']

# define function to show data
def show_explore_page():
    st.title("A Primer into Horse Racing")
    st.write("""
        ### This data analysis was done from a dataset of horse racing data from \
        Hong Kong Jockey Club. The dataset contains the race results of 1561 local \
        races from 2014 to 2017. 

        ### On average, betting on all horses equally will incur a loss of 17.5%.\
         Let's see if we can do better using machine learning!
        """)

    #---------------------------------------------------------------------------------- 
    # insert new line
    
    st.write("")
    st.write("""#### Number of Horses per Race""")

    # Plot the distribution of the number of horses
    fig1 = plt.figure(figsize=(10, 5))
    numHorsePerRace = df_train.groupby('race_id')['horse_id'].count().value_counts()
    sns.barplot(x=numHorsePerRace.index, y=numHorsePerRace.values)
    plt.xlabel('Number of Horses', fontsize=12)
    plt.ylabel
    plt.title('Number of Horses per Race', fontsize=15)
    st.pyplot(fig1)

    # Describe the data
    st.write("""The most common number of horses per race is 12. The maximum number of horses \
            per race is 14.""")

    #---------------------------------------------------------------------------------- 
    # insert new line
    st.write("")
    st.write("""#### Distribution of a Horse's Weight""")   

    # Draw the mean of a horse weight on same plot
    fig2 = plt.figure(figsize=(10, 6))
    sns.distplot(df_train['declared_horse_weight'], bins=100, kde=False)
    plt.title("Distribution of a horse weight")
    plt.xlabel("Weight")
    plt.ylabel("Count")
    plt.axvline(df_train['declared_horse_weight'].mean(), color='r', linestyle='dashed', linewidth=2)
    plt.show()
    st.pyplot(fig2)
    
    # Describe the data
    st.write("""The weight of a horse can vary from 900 to 1360kg. The average \
        weight of a horse is 1160kg. The distribution is skewed to the right.""")

    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### Weight Handicap""")

    # Plot the distribution of actual weight
    fig3= plt.figure(figsize=(10, 5))
    sns.distplot(df_train['actual_weight'], kde=False)
    plt.xlabel('Actual Weight', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Weights Carried', fontsize=15)
    st.pyplot(fig3)

    # Find the mean of actual weight for each horse number
    meanWtPerHorse = df_train.groupby('horse_number')['actual_weight'].mean()
    
    # change horse number to int
    meanWtPerHorse.index = meanWtPerHorse.index.astype(int)

    # Plot the distribution of mean actual weight for each horse number
    fig4 = plt.figure(figsize=(10, 5))
    sns.barplot(x=meanWtPerHorse.index, y=meanWtPerHorse.values)
    plt.xlabel('Horse Number', fontsize=12)
    plt.ylabel('Mean Weight Handicap (lbs)', fontsize=12)
    plt.title('Mean Weight Handicap for Each Horse Number', fontsize=15)
    st.pyplot(fig4)

    # Describe the data
    st.write("Hong Kong horse racing uses a weight handicap system. The weight ranges from \
            100lbs to 133lbs. The average weight carried is 95lbs. The distribution is skewed to the right. \
            Horse number indicates a higher rating, and usually Horse Number 1 carries the most weight.")

    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### Draw and Gate Number Effect""")

    # Return only 14 columns. Draw #15 is not included.
    avgPos_vs_Draw = df_train.groupby('draw')['finishing_position'].mean()[:14]

    # Plot the distribution of average finishing position against draw
    fig5 = plt.figure(figsize=(10, 5))
    sns.barplot(x=avgPos_vs_Draw.index, y=avgPos_vs_Draw.values)
    plt.xlabel('Draw', fontsize=12)
    plt.ylabel('Average Finishing Position', fontsize=12)
    plt.title('Distribution of Average Finishing Position against Draw', fontsize=15)
    st.pyplot(fig5)

    # Describe the data
    st.write("""Draw is the position of the horse in the starting gate. The average finishing position \
            is higher for horses with a lower draw number and vice versa. This is because \
            horses with draw 1 have the advantage of being the first to start closest to the inside rail.""")

    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### Race Type/ Distance""")

    # Plot the distribution of race distance
    fig6 = plt.figure(figsize=(10, 5))
    
    # Do a groupby to see distribution
    df_racetype = df_train[['race_id', 'race_distance']].\
        drop_duplicates().groupby('race_distance').count().reset_index()
    
    # Plot the distribution of df_racetype
    sns.barplot(x=df_racetype['race_distance'], y=df_racetype['race_id'])
    plt.xlabel("Race Distance (m)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Race Types", fontsize=15)
    st.pyplot(fig6)

    # Describe the data
    st.write("""The most popular race distance is 1200m. The longer distances races are \
        not as popular as the shorter distances.""")
    
    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### Correlation between Winning Position and Other Variables""")

    # view the correlation matrix
    corr = df_train.corr()

    # do a mask to hide the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # plot the heatmap with the mask and correct aspect ratio
    fig6 = plt.figure(figsize=(15, 10))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, 
                center=0, square=False, linewidths=.5, 
                cbar_kws={"shrink": .8}, annot=True)
    plt.title('Correlation Matrix', fontsize=15)
    st.pyplot(fig6)

    # Describe the data
    st.write("""There is a positive correlation between the winning position and race odds. Recent \
        performances of the jockey and horse also seems to have an effect on the winning position.""")

    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### Jockeys to Watch""")

    # Finding the number of wins per jockey
    jockeyWins = df_train.groupby('jockey')['finishing_position'].apply(lambda x: (x==1).sum())

    # Finding the win rate per jockey
    jockeyWinRate = df_train.groupby('jockey')['HorseWin'].mean()

    # Plot win rate against number of wins
    fig7 = plt.figure(figsize=(10, 5))
    sns.scatterplot(x=jockeyWinRate, y=jockeyWins)
    plt.xlabel('Win Rate', fontsize=12)
    plt.ylabel('Number of Wins', fontsize=12)
    plt.title('Distribution of Win Rate against Number of Wins', fontsize=15)

    # Annotate the top 5 jockeys with the most wins
    for i in jockeyWins.sort_values(ascending=False)[:5].index:
        plt.annotate(i, (jockeyWinRate[i], jockeyWins[i]))

    # Annotation for the top 5 jockeys with the highest win rate
    for i in jockeyWinRate.sort_values(ascending=False)[:5].index:
        plt.annotate(i, (jockeyWinRate[i], jockeyWins[i]))
    
    st.pyplot(fig7)

    # Describe the data
    st.write("""The top jockeys with the most wins as well as high win rates are Zac Purton \
        and Joao Moreira.""")

    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### Horses to Watch""")

    # Find the number of wins per horse
    horseWins = df_train.groupby('horse_name')['finishing_position'].apply(lambda x: (x==1).sum())

    # Find the win rate per horse
    horseWinRate = df_train.groupby('horse_name')['HorseWin'].mean()

    # Plot win rate against number of wins
    fig8 = plt.figure(figsize=(15, 10))
    sns.scatterplot(x=horseWinRate, y=horseWins)
    plt.xlabel('Win Rate', fontsize=12)
    plt.ylabel('Number of Wins', fontsize=12)
    plt.title('Distribution of Win Rate against Number of Wins', fontsize=15)

    # Annotate the top 5 horses with the most wins
    for i in horseWins.sort_values(ascending=False)[:3].index:
        plt.annotate(i, (horseWinRate[i], horseWins[i]))

    # Annotation for the top 5 horses with the highest win rate
    for i in horseWinRate.sort_values(ascending=False)[3:7].index:
        plt.annotate(i, (horseWinRate[i], horseWins[i]))
    
    st.pyplot(fig8)

    # Describe the data
    st.write("""The top horses with the most wins as well as high win rates are CONTENTMENT, BLIZZARD \
        and SUPREME PROFIT.""")

    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### The Important Question. Can you make money betting on horse racing?""")

    st.write("""The answer is yes! We have done backtesting on multiple models, and 75% of the time, \
        the model will make a profit. The models are not perfect, and there are times when they will lose money. \
            However, with more data to train on, the model will be able to make more accurate predictions over time. """)