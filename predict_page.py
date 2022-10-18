import streamlit as st
import pickle
import numpy as np

# Load the model
def load_model():
    with open('saved_steps.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()

model = data['model']
le_horse_name = data['le_horse_name']
le_jockey = data['le_jockey']

# define show prediction function
def show_predict_page():
    st.title("Horse Winning Prediction")

    st.write("""### We need some information to predict whether you should place your bets on the horse!""")

    # define tuple of horse names by invoking the inverse_transform method on the label encoder
    horse_names = tuple(le_horse_name.inverse_transform([i for i in range(len(le_horse_name.classes_))]))

    # define tuple of jockey names by invoking the inverse_transform method on the label encoder
    jockeys = tuple(le_jockey.inverse_transform([i for i in range(len(le_jockey.classes_))]))

    # Define race distance
    race_dist = (1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400)

    # -------------------------------------------------------------------------------
    # create a selectbox for the horse name
    
    horse_name = st.selectbox('Horse Name', horse_names)
    
    # create input box for the horse number
    horse_number = st.number_input('Horse Number', min_value=1, max_value=14, value=1)

    # create a selectbox for the jockey
    jockey = st.selectbox('Jockey', jockeys)

    # create a slider for the actual weight
    actual_weight = st.slider('Extra Weights Added', 100, 135, 1)

    # create a slider for the declared horse weight
    declared_horse_weight = st.slider('Declared Horse Weight', 1000, 1400, 1)

    # create an input box for draw
    draw = st.number_input('Draw number', min_value=1, max_value=14, value=1)

    # create an input box for the odds
    odds = st.number_input('Odds', min_value=1.0, max_value=100.0, value=1.0, step=0.1)

    # create a slider for the jockey_ave_rank
    jockey_ave_rank = st.slider('Jockey Recent Average Rank', 1.0, 7.0)

    # create a slider for the trainer_ave_rank
    trainer_ave_rank = st.slider('Trainer Recent Average Rank', 1.0, 7.0)

    # create a slider for the recent_ave_rank
    recent_ave_rank = st.slider('Horse Recent Average Rank', 1.0, 7.0)

    # create selectbox for the race type
    race_type = st.selectbox("Race Distance", race_dist)

    # create a button to predict
    if st.button('Predict'):
        # create a numpy array of the input values
        input_variables = np.array([[horse_name, horse_number, jockey, actual_weight, 
                                    declared_horse_weight, draw, odds, jockey_ave_rank, 
                                    trainer_ave_rank, recent_ave_rank, race_type]])

        # encode the input variables using the label encoders
        input_variables[:, 0] = le_horse_name.transform(input_variables[:, 0])
        input_variables[:, 2] = le_jockey.transform(input_variables[:, 2])

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

 
