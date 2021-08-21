#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import streamlit as st
import pickle as p

p_out = open("C:/Users/Rohit Chavan/model_xgb.pkl", "rb")
model = p.load(p_out)

def welcome():
    return "Welcome All"
def user_input_features():
    st.header('User Input Features')
    opened_by = st.text_input('Opened by', )
    resolved_by=st.text_input('Resolved by',)
    caller_id=st.text_input('Caller ID',)
    sys_created_by=st.text_input('SYS Created by',)
    subcategory=st.text_input('Subcategory',)
    location=st.text_input('Location',)
    sys_updated_by=st.text_input('SYS Updated By',)
    opened_hour=st.text_input('Opened Hour',)
    assigned_to=st.text_input('Assigned To',) 
    Data = {'Opened by': opened_by, 'Resolved by': resolved_by,
            'Caller ID': caller_id,'SYS Created by': sys_created_by,
            'Subcategory': subcategory, 'Location': location,
             'SYS Updated_by': sys_updated_by,'Opened Hour': opened_hour,
    'Assigned to': assigned_to, 
   }
    features = pd.DataFrame(Data,index=[0])
    return features
def main():
    from PIL import Image
    img = Image.open("C:/Users/Rohit Chavan/image.png.jpeg")
    img2=Image.open("D:/streamlitlogo1.png")
    
    img4=Image.open("D:/incidentimg3.jpg")
    
    st.image(img, width=800)
    st.sidebar.image(img4, width=300)
    st.sidebar.title("**About**")
   
    st.sidebar.subheader("Impact of Incidents")
    st.sidebar.write("**Hello There! **")
    st.sidebar.write("This is a Machine Learning model")
    st.sidebar.write("Which uses XGboost to Predict the Impact of Incidents")
    st.sidebar.write("Curious..? Want to try this..? Follow this Steps")
    st.sidebar.write("1.Input Values ")
    st.sidebar.write("2.Hit **Predict!** ")
    st.sidebar.write("After it will go directly to Machine Learning Model and It will Predict the ** Impact of Incident!!**")
    
    st.sidebar.image(img2, width=100)
    st.sidebar.title("Made With Streamlit by")
    st.sidebar.header("Team 5 ")
    st.sidebar.write("***Vivek***",",","***Pallavi***")
    st.sidebar.write("***Suraj***",",","***Navya***",",","***Rohit***")
    st.sidebar.header("Guided by:-")
    st.sidebar.write("***Vinod***",",","***Kavi Priya***")
   
    
    
    
    
    
    
    
    
    
    
    html_temp = """
    <div style="background-color:Teal;padding:10px">
    <h2 style="color:white;text-align:center">Streamlit Incidents Impact ML App </h2>
    </div>
    """
    level = st.slider("Select the level", 1, 3)
    st.text('Selected: {}'.format(level))
    st.markdown(html_temp, unsafe_allow_html=True)
    df = user_input_features()
    
    
   
           
    st.subheader('User Input parameters')
    st.write(df)
    result = ""
    if st.button("Predict"):
        result = model.predict(df)
    st.success('The prediction is {}'.format(result))
    st.markdown('[1]-The Impact is **High**')
    st.markdown('[2]-The Impact is **Medium**')
    st.markdown('[3]-The Impact is **Low**')
    
    
    

if __name__=='__main__':
    main()


