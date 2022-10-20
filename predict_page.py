import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the model
def load_model():
    with open('saved_steps.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()

model = data['model']
df_test = data['test_data']
df_train = data['train_data']

# define show prediction function
def show_predict_page():
    st.title("Will your horse win?! 🐎")

    st.write("""### We need some information to predict whether you should place your bets on the horse!""")

    # define tuple of horse names based on the training data
    horse_names = tuple(df_train['horse_name'].unique())

    # define tuple of jockey names based on the training data
    jockey_names = tuple(df_train['jockey'].unique())

    # Define race distance
    race_dist = (1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400)

    # -------------------------------------------------------------------------------
    # create a selectbox for the horse name
    horse_name = st.selectbox('Horse Name', horse_names)
    
    # create input box for the horse number
    horse_number = st.number_input('Horse Number', min_value=1, max_value=14, value=1)

    # create a selectbox for the jockey
    jockey = st.selectbox('Jockey', jockey_names)

    # create an input box for draw
    draw = st.number_input('Draw number', min_value=1, max_value=14, value=1)

    # create an input box for the odds
    odds = st.number_input('Odds', min_value=1.0, max_value=100.0, value=1.0, step=0.1)

    # create selectbox for the race type
    race_type = st.selectbox("Race Distance", race_dist)
    
    # -------------------------------------------------------------------------------
    # Based on above inputs, we have to transform some of it into variables describing inputs

    # Transform horse_name into actual horse weight
    horse_weight = df_test[df_test['horse_name'] == horse_name]['declared_horse_weight'].mean()
    if np.isnan(horse_weight):
            horse_weight = df_train[df_train['horse_name'] == horse_name]['declared_horse_weight'].mean()

    # Transform horse name into recent average rank
    avg_rank = df_test[df_test['horse_name'] == horse_name]['recent_ave_rank'].mean()
    if np.isnan(avg_rank):
            avg_rank = df_train[df_train['horse_name'] == horse_name]['recent_ave_rank'].mean()

    # Transform horse number into handicap weights
    handicap_weight = df_test[df_test['horse_number'] == horse_number]['actual_weight'].mean()
    if np.isnan(handicap_weight):
            handicap_weight = df_train[df_train['horse_name'] == horse_name]['actual_weight'].mean()

    # Transform jockey into jockey recent average rank
    jockey_avg_rank = df_test[df_test['jockey'] == jockey]['jockey_ave_rank'].mean()
    if np.isnan(jockey_avg_rank):
            jockey_avg_rank = df_train[df_train['jockey'] == jockey]['jockey_ave_rank'].mean()
    

    # create a button to predict
    if st.button('Predict'):
        # create a numpy array of the input values
        input_variables = np.array([[handicap_weight, horse_weight, draw, odds, 
                                    jockey_avg_rank, avg_rank, race_type]])

        # change the input variable type to float
        input_variables = input_variables.astype(np.float)

        # get the prediction from the model
        prediction = model.predict(input_variables)[0]
        
        # print the prediction on whether to bet or not
        # Make this bigger and bolder
        st.write(f"""### The model prediction is: {prediction}""")
        if prediction == 1:
            st.success('Bet on this horse!')
        else:
            st.warning('Do not bet on this horse!')

 
