import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the saved model and preprocessors
with open('decision_tree_regressor_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('scaler1.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('preprocessor1.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Function to preprocess the input data
def preprocess_data(new_data):
    new_df = pd.DataFrame(new_data, columns=['quantity tons_log', 'application', 
                                             'thickness_log', 'width', 'country', 'customer', 'product_ref',
                                               'item type', 'status'])
    new_data_processed = preprocessor.transform(new_df)
    new_data_scaled = scaler.transform(new_data_processed)
    return new_data_scaled

# Function to predict selling price
def predict_price(new_data_scaled):
    predicted_log_prices = loaded_model.predict(new_data_scaled)
    predicted_prices = np.exp(predicted_log_prices)
    return predicted_prices[0]

# Main function to run the Streamlit web app
def main():
    st.title(":rainbow[Selling Price Prediction]")

    st.write('Fill in the details to predict the selling price.')

    # Define min and max values for each column
    min_max_values = {
        'quantity_tons_log': (0, 100000),
        'thickness_log': (0, 400),
        'width': (0, 3000),
        'customer': (12000,30408000),
        'product_ref': (0, 100000)
    }

    # Display input fields along with min and max values
    for column_name, (min_value, max_value) in min_max_values.items():
        if column_name == 'quantity_tons_log':
            quantity_tons_log = st.number_input('Quantity Tons (log)', min_value=min_value, max_value=max_value)
            st.text(f"Min Value: {min_value}, Max Value: {max_value}")
        elif column_name == 'thickness_log':
            thickness_log = st.number_input('Thickness (log)', min_value=min_value, max_value=max_value)
            st.text(f"Min Value: {min_value}, Max Value: {max_value}")
        elif column_name == 'width':
            width = st.number_input('Width', min_value=min_value, max_value=max_value)
            st.text(f"Min Value: {min_value}, Max Value: {max_value}")
        elif column_name == 'customer':
            customer = st.number_input('Customer', min_value=min_value, max_value=max_value)
            st.text(f"Min Value: {min_value}, Max Value: {max_value}")
        elif column_name == 'product_ref':
            product_ref = st.number_input('Product Reference', min_value=min_value, max_value=max_value)
            st.text(f"Min Value: {min_value}, Max Value: {max_value}")
        else:
            st.text_input(column_name)
    # Other inputs
    country = st.selectbox('Country',[28, 25, 30, 32, 38, 78, 27, 77, 113, 79, 26, 39, 40, 84, 80, 107,89])
    application = st.selectbox('Application',[10, 41, 28, 59, 15, 4, 38, 56, 42, 26, 27, 19, 20, 
                                            66, 29, 22, 40, 25, 67, 79, 3, 99, 2, 5, 39, 69, 70, 
                                            65, 58, 68])
    item_type = st.selectbox('Item Type', ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'])  
    status = st.selectbox('Status', ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful',
                                      'Revised', 'Offered', 'Offerable'])

    new_data = [[quantity_tons_log, application, thickness_log, width, country, customer, product_ref, item_type, 
                    status]]

     # Predict selling price if the user clicks the button
    if st.button('Predict'):
                new_data_scaled = preprocess_data(new_data)
                predicted_price = predict_price(new_data_scaled)
                st.success(f'Predicted selling price: {predicted_price}')

if __name__ == '__main__':
        main()