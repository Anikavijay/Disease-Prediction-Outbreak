import numpy as np # type: ignore
import os  #Used interact with file system
import pickle  #Help to load pretrained machine learning models
import streamlit as st  # type: ignore #Used for web application
from streamlit_option_menu import option_menu # type: ignore #It is third party library to create stylish side bar menu

#Set Page Configuration
st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',
                   page_icon='doctor')

#Load Saved Models
diabetes_model = pickle.load(open(r'C:\Users\LENOVO\Desktop\predictions\saved_models\diab_model.sav', 'rb'))
heart_model = pickle.load(open(r'C:\Users\LENOVO\Desktop\predictions\saved_models\heart_model.sav', 'rb'))
diabetes_scaler = pickle.load(open(r'C:\Users\LENOVO\Desktop\predictions\saved_models\scaler_diab.sav', 'rb'))
heart_scaler = pickle.load(open(r'C:\Users\LENOVO\Desktop\predictions\saved_models\scaler_heart.sav', 'rb'))
parkinsons_model=pickle.load(open(r'C:\Users\LENOVO\Desktop\predictions\saved_models\parkinsons_model.sav','rb'))
parkinsons_scaler=pickle.load(open(r'C:\Users\LENOVO\Desktop\predictions\saved_models\scaler_parkinsons.sav','rb'))
  
#Sidebar Menu
with st.sidebar:
    selected=option_menu('Prediction of Disease Outbreak System',
    ['Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction'],
                        menu_icon='hospital-fill',icons=['activity','heart','person'],default_index=0)

#Diabetes Prediction
if(selected=='Diabetes Prediction'):
    st.title("Diabetes Prediction")
    col1,col2,col3=st.columns(3)
    with col1:
         Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
         Glucose = st.text_input("Glucose Level")
    with col3:
         BloodPressure = st.text_input("Blood Pressure")
    with col1:
         SkinThickness = st.text_input("Skin Thickness")
    with col2:
         Insulin = st.text_input("Insulin Level")
    with col3:
         BMI = st.text_input("BMI")
    with col1:  
         DiabetesPedigreeFunction= st.text_input("Diabetes Pedigree Function")
    with col2:
         Age = st.text_input("Age")
    
    if st.button("Predict Diabetes"):
        try:
            input_data = np.array([[float(Pregnancies), float(Glucose), float(BloodPressure),
                                    float(SkinThickness), float(Insulin), float(BMI),
                                    float(DiabetesPedigreeFunction), float(Age)]])
            input_data_scaled = diabetes_scaler.transform(input_data)
            prediction = diabetes_model.predict(input_data_scaled)
            
            if prediction[0] == 1:
                st.error("The person is diabetic")
            else:
                st.success("The person is not diabetic")
        except ValueError:
            st.error("Please enter valid numerical values.")

#Heart Disease Prediction
if(selected=='Heart Disease Prediction'):
    st.title("Heart Disease Prediction")
    col1,col2,col3=st.columns(3)
    with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Sex (1 = Male, 0 = Female)")
    with col3:
        cp = st.text_input("Chest Pain Type (0-3)")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol = st.text_input("Serum Cholesterol in mg/dl")
    with col3:
        fbs = st.text_input("Fasting Blood Sugar>120 mg/dl")
    with col1:
        restecg = st.text_input("Resting ECG Results (0-2)")
    with col2:
        thalach = st.text_input("Maximum Heart Rate Achieved")
    with col3:
        exang = st.text_input("Exercise Induced Angina (1=Yes, 0=No)")
    with col1:
        oldpeak = st.text_input("ST Depression Induced by Exercise")
    with col2:
        slope = st.text_input("Slope of Peak Exercise ST Segment (0-2)")
    with col3:
        ca = st.text_input("Major Vessels Colored by Flourosopy (0-3)")
    with col1:
        thal = st.text_input("Thalassemia (0-3)")
    
    if st.button("Predict Heart Disease"):
        try:
            input_data = np.array([[float(age), float(sex), float(cp), float(trestbps), float(chol),
                                    float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak),
                                    float(slope), float(ca), float(thal)]])
            input_data_scaled = heart_scaler.transform(input_data)
            prediction = heart_model.predict(input_data_scaled)
            
            if prediction[0] == 1:
                st.error("The person has heart disease")
            else:
                st.success("The person does not have heart disease")
        except ValueError:
            st.error("Please enter valid numerical values.")

#Parkinson's Prediction
if (selected=='Parkinsons Prediction'):
    st.title("Parkinson's Disease Prediction")
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
     fo=st.text_input('MDVP:Fo(Hz)')
    with col2:
     fhi=st.text_input('MDVP:Fhi(Hz)')
    with col3:
     flo=st.text_input('MDVP:Flo(Hz)')
    with col4:
     Jitter_percent=st.text_input('MDVP:Jitter(%)')
    with col5:
     Jitter_Abs=st.text_input('MDVP:Jitter(Abs)')
    with col1:
     RAP=st.text_input('MDVP:RAP')
    with col2:
     PPQ=st.text_input('MDVP:PPQ')
    with col3:
     DDP=st.text_input('Jitter:DDP')
    with col4:
     Shimmer=st.text_input('MDVP:Shimmer')
    with col5:
     Shimmer_dB=st.text_input('MDVP:Shimmmer(dB)')
    with col1:
     APQ3=st.text_input('Shimmer:APQ3')
    with col2:
     APQ5=st.text_input('Shimmer:APQ5')
    with col3:
     APQ=st.text_input('MDVP:APQ')
    with col4:
     DDA=st.text_input('Shimmer:DDA')
    with col5:
     NHR=st.text_input('NHR')
    with col1:
     HNR=st.text_input('HNR')
    with col2:
     RPDE=st.text_input('RPDE')
    with col3:
     DFA=st.text_input("DFA")
    with col4:
     spread1=st.text_input('Spread1')
    with col5:
     spread2=st.text_input('Spread2')
    with col1:
     D2=st.text_input('D2')
    with col2:
     PPE=st.text_input('PPE')
    
    if st.button("Predict Parkinson's Disease"):
        try:
            input_data = np.array([[float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                                    float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                                    float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE),
                                    float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]])
            input_data_scaled = parkinsons_scaler.transform(input_data)
            prediction = parkinsons_model.predict(input_data_scaled)
            
            if prediction[0] == 1:
                st.error("The person has Parkinson's disease")
            else:
                st.success("The person does not have Parkinson's disease")
        except ValueError:
            st.error("Please enter valid numerical values.")




















 