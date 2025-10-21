# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

password_guess = st.text_input("What is the password?")
if password_guess != st.secrets["password"]:
    st.stop()
st.title('Graduate Admission Prediction') 

# st.markdown(
#     "<h1 style='color: green;'>Graduate Admission prediction</h1>",
#     unsafe_allow_html=True
# )

# Display the image
st.image('admission.jpg', width = 400)

st.write("This app uses multiple inputs to predict" \
" the probability of admission to grad school") 



# Load the pre-trained model from the pickle file
reg_pickle = open('reg_admission.pickle', 'rb') 
clf = pickle.load(reg_pickle) 
reg_pickle.close()

# Create a sidebar for input collection
st.sidebar.header('Enter Your Profile Details')

# # Option 1: Asking users to input their data as a file
# penguin_file = st.sidebar.file_uploader('Option 1: Upload your own penguin data')

# # Option 2: Asking users to input their data using a form in the sidebar
# st.sidebar.write('Option 2: Use the following form')

#---------------------------------------------------------------------------------------------
# Using Default (Original) Dataset to Automate Few Items
#---------------------------------------------------------------------------------------------

# Load the default dataset
default_df = pd.read_csv('Admission_Predict.csv')
default_df = default_df.dropna().reset_index(drop = True) 
# NOTE: drop = True is used to avoid adding a new column for old index


# Sidebar input fields for numerical variables using sliders
# NOTE: Make sure that variable names are same as that of training dataset
GRE_Score = st.sidebar.number_input('GRE Score', 
                                   min_value = default_df['GRE Score'].min(), 
                                   max_value = default_df['GRE Score'].max(), 
                                   step = 1)

TOEFL_Score = st.sidebar.number_input('TOEFL Score', 
                                  min_value = default_df['TOEFL Score'].min(), 
                                  max_value = default_df['TOEFL Score'].max(), 
                                  step = 1)

CGPA = st.sidebar.number_input('CGPA', 
                                  min_value = default_df['CGPA'].min(), 
                                  max_value = default_df['CGPA'].max(), 
                                  step = .01)

# For categorical variables, using selectbox
research = st.sidebar.selectbox('Research Experience', options = default_df['Research'].unique()) 

University_Rating = st.sidebar.number_input('University Rating', 
                                  min_value = default_df['University Rating'].min(), 
                                  max_value = default_df['University Rating'].max(), 
                                  step = 1)

SOP = st.sidebar.slider('Statement of Purpose (SOP)', 
                                      min_value = default_df['SOP'].min(), 
                                      max_value = default_df['SOP'].max(), 
                                      step = 0.5)

LOR = st.sidebar.slider('Letter of Recommendation (LOR)', 
                                min_value = default_df['LOR'].min(), 
                                max_value = default_df['LOR'].max(), 
                                step = .5)




# If no file is provided, then allow user to provide inputs using the form
#if penguin_file is None:
    # Encode the inputs for model prediction
encode_df = default_df.copy()
encode_df = encode_df.drop(columns = ['Chance of Admit'])
# Combine the list of user data as a row to default_df
encode_df.loc[len(encode_df)] = [GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, research]

# Create dummies for encode_df
encode_dummy_df = pd.get_dummies(encode_df)

# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

# Using predict() with new data provided by the user
#new_prediction = clf.predict(user_encoded_df)

# Show the predicted species on the app
st.subheader("Predicted Admission Probability")

y_pred, y_pis = clf.predict(user_encoded_df, alpha=.1)

# Display results
lower, upper = y_pis[0]
lower_val = float(np.ravel(lower)[0])
upper_val = float(np.ravel(upper)[0])
st.subheader("Prediction Results")
st.write(f"**Predicted Admission Probability:** {y_pred[0]:.2f}")
st.write(f"**90% Prediction Interval:** ({lower_val:.2f}, {upper_val:.2f})")

# else:
#    # Loading data
#    user_df = pd.read_csv(penguin_file) # User provided data
#    original_df = pd.read_csv('penguins.csv') # Original data to create ML model
   
#    # Dropping null values
#    user_df = user_df.dropna().reset_index(drop = True) 
#    original_df = original_df.dropna().reset_index(drop = True)
   
#    # Remove output (species) and year columns from original data
#    original_df = original_df.drop(columns = ['species', 'year'])
#    # Remove year column from user data
#    user_df = user_df.drop(columns = ['year'])
   
#    # Ensure the order of columns in user data is in the same order as that of original data
#    user_df = user_df[original_df.columns]

#    # Concatenate two dataframes together along rows (axis = 0)
#    combined_df = pd.concat([original_df, user_df], axis = 0)

#    # Number of rows in original dataframe
#    original_rows = original_df.shape[0]

#    # Create dummies for the combined dataframe
#    combined_df_encoded = pd.get_dummies(combined_df)

#    # Split data into original and user dataframes using row index
#    original_df_encoded = combined_df_encoded[:original_rows]
#    user_df_encoded = combined_df_encoded[original_rows:]

#    # Predictions for user data
#    user_pred = clf.predict(user_df_encoded)

#    # Predicted species
#    user_pred_species = user_pred

#    # Adding predicted species to user dataframe
#    user_df['Predicted Species'] = user_pred_species
   
#    # Show the predicted species on the app
#    st.subheader("Predicting Your Penguin's Species")
#    st.dataframe(user_df)

# Showing additional items in tabs
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted vs. Actual", "Coverage Plot"])


# Tab 1: Feature Importance Visualization
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Features used in this prediction are ranked by relative importance.")

# Tab 2: Visualizing Histogram of Residuals
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residuals_dist.svg')
    st.caption("Distribution of residuals to evaluate prediction quality")

# Tab 3: Predicted vs. Actual
with tab3:
    st.write("### PLot of Predicted vs. Actual")
    st.image('pred_vs_act.svg')
    st.caption("Visual comparison of predicted actual values.")

# Tab 4: Coverage plot
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage_plot.svg')
    st.caption("Range of predictions with confience intervals.")