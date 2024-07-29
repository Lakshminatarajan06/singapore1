import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Flat Resale Price - Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
def background():
    st.markdown("""
        <style>
        .main {
            background-color: #ffcc99;
        }
        .title {
            font-size: 2.5em;
            color: #4a4a4a;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 1.5em;
            color: #4a4a4a;
            text-align: center;
            margin-bottom: 30px;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            padding: 20px;
        }
        .stButton>button {
            color: white;
            background-color: #007BFF;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        </style>
        """, unsafe_allow_html=True)
    
background()

# Title and subtitle
st.markdown('<div class="title">Singapore Resale Flat Price Prediction App</div>', unsafe_allow_html=True)


# Sidebar
st.sidebar.title("Sidebar Menu")
st.sidebar.write("Use this sidebar to navigate through the app.")

option = st.sidebar.radio("Select an option", ['About Project', 'Price Prediction'])

# Mapped Data
town_map={'ANG MO KIO':1, 'BEDOK':2, 'BISHAN':3, 'BUKIT BATOK':4, 'BUKIT MERAH':5, 'BUKIT TIMAH':6, 'CENTRAL AREA':7,
        'CHOA CHU KANG':8, 'CLEMENTI':9, 'GEYLANG':10, 'HOUGANG':11, 'JURONG EAST':12, 'JURONG WEST':13, 'KALLANG/WHAMPOA':14,
        'MARINE PARADE':15, 'QUEENSTOWN':16, 'SENGKANG':17, 'SERANGOON':18, 'TAMPINES':19, 'TOA PAYOH':20,
        'WOODLANDS':21, 'YISHUN':22, 'LIM CHU KANG':23, 'SEMBAWANG':24, 'BUKIT PANJANG':25, 'PASIR RIS':26, 'PUNGGOL':27}

block_map= {'309':1, '216':2, '211':3, '202':4, '235':5,
            '85C': 576, '269B':604, '335A':602, '27A':617, '23A':618}

flat_type_map= {'1 ROOM':1, '3 ROOM':2, '4 ROOM':3, '5 ROOM':4, '2 ROOM':5,
            'EXECUTIVE': 6, 'MULTI GENERATION':7, 'MULTI-GENERATION':8}

street_name_map= {'ANG MO KIO AVE 1':1, 'ANG MO KIO AVE 3':2, 'ANG MO KIO AVE 4':3, 'ANG MO KIO AVE 10':4, 'ANG MO KIO AVE 5':5,
            'ANG MO KIO AVE 8': 6, 'ANG MO KIO AVE 6':7, 'ANG MO KIO AVE 9':8, 'ANG MO KIO AVE 2':9, 'BEDOK RESERVOIR RD':10,
            'BEDOK NTH ST 3':11, 'BEDOK STH RD':12, 'NEW UPP CHANGI RD':13, 'BEDOK NTH RD':14, 'BEDOK STH AVE 1':15}

flat_model_map= {'IMPROVED':1, 'NEW GENERATION':2, 'MODEL A':3, 'STANDARD':4, 'SIMPLIFIED':5, 'MODEL A-MAISONETTE':6,
                'APARTMENT':7, 'MAISONETTE':8, 'TERRACE':9, '2-ROOM':10, 'IMPROVED-MAISONETTE': 11, 'MULTI GENERATION': 12,
                'PREMIUM APARTMENT': 13, 'New Generation': 15, 'Adjoined flat': 23, 'Premium Maisonette': 27, '2-room': 28,
                'Model A2': 29, 'DBSS': 30, 'Type S1': 31, 'Type S2': 32, 'Premium Apartment Loft': 33, '3Gen': 34}


if option=="About Project":

    col1, col2 = st.columns(2)

    with col1: 
        st.markdown('<p style="color:blue; font-size:24px;">Project Objective:</p>', unsafe_allow_html=True)

        st.markdown('<p style="color:black;">The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.</p>', unsafe_allow_html=True)
    
    with col2:

        st.image(r"C:\Users\Good Day\Desktop\Project 6\images.jpg", use_column_width=True)
             
                
# Price Prediction
if option=="Price Prediction":

    st.markdown('<p style="color:blue;">PRICE PREDICTION</p>', unsafe_allow_html=True)

    with st.form("price"):

        # Splitting two colummns
        column_width=[2,0.5,2]
        col1,col2,col3=st.columns(column_width)

        with col1:    
           
            # customer=st.text_in('**Customer ID**')
            town=st.selectbox('**Town Name**', ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA',
                            'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA',
                            'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG','SERANGOON', 'TAMPINES', 'TOA PAYOH',
                            'WOODLANDS', 'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL'])
            
            block=st.selectbox('**Block No**', ['309', '216', '211', '202', '235','85C', '269B', '335A', '27A', '23A'])
            
            flat_type=st.selectbox('**Flat Type**', ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM','EXECUTIVE', 'MULTI GENERATION'])
            
            flat_model=st.selectbox('**Flat Model**', ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED', 'MODEL A-MAISONETTE',
                                'APARTMENT', 'MAISONETTE', 'TERRACE', '2-ROOM', 'IMPROVED-MAISONETTE', 'MULTI GENERATION',
                                'PREMIUM APARTMENT', 'New Generation', 'Adjoined flat', 'Premium Maisonette', '2-room',
                                'Model A2', 'DBSS', 'Type S1', 'Type S2', 'Premium Apartment Loft', '3Gen'])
            
            street_name=st.selectbox('**Street Name**', ['ANG MO KIO AVE 1', 'ANG MO KIO AVE 3', 'ANG MO KIO AVE 4', 'ANG MO KIO AVE 10', 'ANG MO KIO AVE 5',
                                'ANG MO KIO AVE 8', 'ANG MO KIO AVE 6', 'ANG MO KIO AVE 9', 'ANG MO KIO AVE 2', 'BEDOK RESERVOIR RD',
                                'BEDOK NTH ST 3', 'BEDOK STH RD', 'NEW UPP CHANGI RD', 'BEDOK NTH RD', 'BEDOK STH AVE 1'])

        with col3:

            month=st.number_input('**Enter the Year**', step=1, format='%d')
            
            floor_area=st.number_input('**Enter Floor Area (In Sq.m)**', step=1, format='%d')
            
            lease_commense_date=st.number_input('**Lease Commense Year**', step=1, format='%d')
            
            storey_range=st.text_input('**Enter Storey Range (Ex. 5 TO 10)**')

            remaining_lease=st.number_input('**Enter Remaining Lease Year**', step=1, format='%d')

            flat_old=st.number_input('**Enter Flat Old**', step=1, format='%d')


        # Every form must have a submit button.
        submitted = st.form_submit_button("Predict Price")
        if submitted:

            if month and floor_area and lease_commense_date and remaining_lease and storey_range and flat_old:


                with open(r'C:\Users\Good Day\Desktop\Project 6\deci_pred.pkl', 'rb') as file:
                    model=pickle.load(file)

                # Map the categorical variables to their numerical values
                
                town = town_map[town]
                block= block_map[block]
                flat_type= flat_type_map[flat_type]
                flat_model= flat_model_map[flat_model]
                street_name= street_name_map[street_name]


                # Converting storey range to avg storey as learn in machine learning
                storey_range=storey_range.strip().replace('TO','-')
                start, end=storey_range.split('-')

                start=int(start)
                end=int(end)

                storey_range=(start+end)/2
                
                # Prepare user input data
                user_input_data = np.array([month, town, flat_type, block, street_name, np.log(float(storey_range)),
                                             floor_area, flat_model, np.log(float(lease_commense_date)), remaining_lease, flat_old])
                
                # Reshape to 2D array
                user_input_data=user_input_data.reshape(1,-1)

                # Make prediction
                y_pred = model.predict(user_input_data)

                # Resale price
                resale_price=round(y_pred[0],0)

                # Display the Resale price

                st.write(f'**The resale price of flat with above Features: {resale_price}**')


            else:

                st.warning("**Please fill in all the data**")
                
                
                    

            

        


