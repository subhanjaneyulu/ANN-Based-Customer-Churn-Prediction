import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 


##load the traing model 
model = tf.keras.models.load_model('model.h5')

## load the encoder and scalers 

#load the encoder and scaler 
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open("scaler",'rb') as file:
    scaler=pickle.load(file)

##streamlit app 

st.title('Customer Churn Prediction')

##input boxes 
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('credit_score')
estimated_salary = st.number_input('Estimated salary')
tenure = st.slider('Tenure',0,10)
number_of_products = st.selectbox(
    "Number Of Products",
    [1,2,3,4]
)

st.write("Selected:", number_of_products)

has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member', [0,1]) 

##prepare input data 
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[number_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

##one-hot encode 'Geopgraphy' 

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out())


#combine one_hot encoded column with input_data 
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1) 

#scale the input_data 
input_data_scaled = scaler.transform(input_data)


##predict churn
prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

st.write(f"Churn Probability : {prediction_proba:.4f}")

if prediction_proba > 0.5:
    st.success('The Customer is likely to Churn')
else:
    st.info('The Customer is not likely to Churn')

