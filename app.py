import streamlit as st
import pickle
import xgboost
import numpy as np



pipe = pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))
st.title("Laptop Predictor")

company= st.selectbox('Brand',df['Company'].unique())

type1 = st.selectbox('Type',df['TypeName'].unique())
# ram
ram = st.selectbox('Ram(in Gb)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('weight of the Laptop')
# touchscreen
touchscreen =st.selectbox('Touchscreen',['No','Yes'])

ips= st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')
# resolution
resolution =st.selectbox('Screen Resolution',['1920x1080','1366x768',
                                              '1600x900','3840x2160','3200x1800'
                                              ,'2800x1800','2560x1600','2560x1440',
                                              '2304x1440'])

# cpu
cpu= st.selectbox('Brand',df['Cpu brand'].unique())

# hard drive
hdd= st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd= st.selectbox('SDD(in GB)',[0,8,128,256,512,1024])
gpu= st.selectbox('GPU',df['Gpu_brand'].unique())

os= st.selectbox('OS',df['Os'].unique())

if st.button('Predict Price'):
    ppi=None
    if touchscreen =='Yes':
        touchscreen=1
    else:
        touchscreen=0
    if ips=='Yes':
        ips = 1
    else:
        ips = 0
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi=((X_res**2)+(Y_res**2))**0.5/screen_size
    # query = np.array([company,type1,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    # query = query.reshape(1, -1)

    import pandas as pd

    query = pd.DataFrame([[company, type1, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]],
                         columns=['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS', 'ppi',
                                  'Cpu brand', 'HDD', 'SSD', 'Gpu_brand', 'Os'])

    st.title("the predicted price of this configuration is "+str(int(np.exp(pipe.predict(query)[0]))))




