import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
from PIL import Image

# Load the model
def load_model():
    with open('saved_steps.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()

model = data['model']
df_test = data['test_data']
df_train = data['train_data']

# define function to show data
def show_explore_page():
    st.title("Using statistical models to predict horse racing results")
    st.write("""
        #### Data analysis and backtesting was done on a dataset of horse racing data from \
        Hong Kong Jockey Club. The dataset contains the race results of more than 2000 local \
        races from 2014 to 2017. 

        #### On average, betting equally on all horses will incur a loss of 17.5%. \
        Even if we were to only bet on the horse with the lowest odds at every race, chances are that you will \
            be losing quite abit over the long run.
        """)

    image1 = Image.open('./images/bet_lowest_odds.jpg')
    st.image(image1, caption='Betting at each race on the horse with the lowest odds',\
         use_column_width=True)

    st.write("""
        #### Our backtesting results show that using a statistical model to predict horse racing \
        can yield better performance.
        """)

    image2 = Image.open('./images/deployment.jpg')
    st.image(image2, caption='Betting at each race using our app prediction',\
         use_column_width=True)

    #---------------------------------------------------------------------------------- 
    # insert new line
    
    st.write("")
    st.write("""#### Winning Odds""")

    # Plot mean of win odds against finishings
    fig1 = plt.figure(figsize=(10, 5))
    sns.barplot(x=df_train['finishing_position'].unique(), y=df_train.groupby('finishing_position')['win_odds'].mean(), palette='Greens_d')
    plt.xlabel('Finishing Position', fontsize=12)
    plt.ylabel('Mean Win Odds', fontsize=12)
    plt.title('Mean Win Odds against Finishing Position', fontsize=15)
    st.pyplot(fig1)

    # Describe the data
    st.write("""A horse with lower odds usually finishes higher. However this does not mean\
        that you should only bet on the horse with the lowest odds. Other factors also contribute\
            to the horse's finishing position. For example, the horse's weight, jockey's weight \
                """)
    #---------------------------------------------------------------------------------- 
 
    # insert new line
    st.write("")
    st.write("""#### Weight Handicap""")

    # Find the mean of actual weight for each horse number
    meanWtPerHorse = df_train.groupby('horse_number')['actual_weight'].mean()
    
    # change horse number to int
    meanWtPerHorse.index = meanWtPerHorse.index.astype(int)

    # Plot the distribution of mean actual weight for each horse number
    fig2= plt.figure(figsize=(10, 5))
    sns.barplot(x=meanWtPerHorse.index.astype(int), y=meanWtPerHorse.values, palette='Greens_d')
    plt.xlabel('Horse Number', fontsize=12)
    plt.ylabel('Mean Actual Weight', fontsize=12)
    plt.title('Distribution of Mean Weight Handicap for Each Horse Number', fontsize=15)
    st.pyplot(fig2)

    # Describe the data
    st.write("Hong Kong horse racing uses a weight handicap system to make races more competitive. The weight ranges from \
            100lbs to 133lbs. The average weight carried is around 95lbs. \
            The horse number indicates a higher rating, and usually Horse Number 1 carries the most weight.")

    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### Draw and Gate Number Effect""")

    # Return only 14 columns. Draw #15 is not included.
    avgPos_vs_Draw = df_train.groupby('draw')['finishing_position'].mean()[:14]

    # Plot the distribution of average finishing position against draw
    fig3 = plt.figure(figsize=(10, 5))
    sns.barplot(x=avgPos_vs_Draw.index, y=avgPos_vs_Draw.values, palette='Greens_d')
    plt.xlabel('Draw', fontsize=12)
    plt.ylabel('Average Finishing Position', fontsize=12)
    plt.title('Distribution of Draw against Average Finishing Position ', fontsize=15)
    st.pyplot(fig3)

    # Describe the data
    st.write("""Draw is the position of the horse in the starting gate. The average finishing position \
            is higher for horses with a lower draw number and vice versa. This is because \
            horses with draw 1 have the advantage of being the first to start closest to the inside rail.""")

    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### Race Type/ Distance""")

    # Plot the distribution of race distance
    fig4 = plt.figure(figsize=(10, 5))
    
    # Do a groupby to see distribution
    df_racetype = df_train[['race_id', 'race_distance']].\
        drop_duplicates().groupby('race_distance').count().reset_index()
    
    # Plot the distribution of df_racetype
    sns.barplot(x=df_racetype['race_distance'], y=df_racetype['race_id'])
    plt.xlabel("Race Distance (m)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Race Types", fontsize=15)
    st.pyplot(fig4)

    # Describe the data
    st.write("""There are different race distances. Like humans, horses have different \
            strengths and weaknesses. Some horses are better at longer distances, \
                while others are better at shorter distances. \
                    """)
    
    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### The Form Factor""")

    # keep only features we want
    cols = ['finishing_position', 'actual_weight', 'declared_horse_weight', \
        'draw', 'recent_ave_rank', \
        'jockey_ave_rank','trainer_ave_rank', 'race_distance']

    # view the correlation matrix
    corr = df_train[cols].corr()

    # do a mask to hide the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # plot the heatmap with the mask and correct aspect ratio
    fig5 = plt.figure(figsize=(15, 10))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, 
                center=0, square=False, linewidths=.5, 
                cbar_kws={"shrink": .8}, annot=True)
    plt.title('Correlation Matrix', fontsize=15)
    st.pyplot(fig5)

    # Describe the data
    st.write("""Recent performances of the jockey and horse seems to have a strong effect \
        on the winning position. We call this the 'form factor'. The form factor looks to be \
            the most important factor in predicting the winning position. \
                """)

    #----------------------------------------------------------------------------------
    # insert new line
    st.write("")
    st.write("""#### The Important Question. Can you make money betting on horse racing?""")

    st.write("""While we do not guarantee success with every bet (nobody can!), our prediction\
        model has shown a recall of 40% and precision of 20%. This means that if you bet on every horse\
           as predicted by our model, you will win 20% of the time.""")