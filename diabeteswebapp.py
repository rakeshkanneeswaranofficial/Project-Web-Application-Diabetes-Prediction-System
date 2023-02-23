#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 03:15:29 2023

@author: rakeshkanneeswaran
"""

import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('/Users/rakeshkanneeswaran/Desktop/trained_model.sav','rb')) 

#creating functions for prediction
def diabetes_prediction(input_data1):

    #changing input data into numpy array

    input_data_as_numpy_array = np.asarray(input_data1)
    #reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    # standrdize the input data
    prediction = loaded_model.predict(input_data_reshaped)
    if (prediction[0] == 0):
       return("the preson is not diabetic")
    else:
       return("The person is diabetic")  
   
    
def main():
    
    #giving title
    st.title(":red[Diabetes mellitus Prediction Web Application]")
    st.header(":blue[Using Mchine learning Support Vector Machine Algorithm]")
    #Getting input data from user 
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Level")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insuline")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age")
    
    #Code for Prediction
    diagnosis = " "
    #Creating button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)    
    
if __name__ == '__main__':
  main()    
    
    
    
  
    



print("everthing is ok")