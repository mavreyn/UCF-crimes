'''
Interactive using PandasAI to query incident data from database
Mapping and geolocation visualizted here as well

Maverick Reynolds
08.02.2023
'''

# Zero shot for classification
# K means as well
# Text search
# Use it to filter and all the rest of the visualization below!
# Queries to another file

import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import numpy as np

from embeddings import *
import json

with open('title_embeddings.json', 'r') as f:
    embeddings_dict = json.load(f)


def main():
    st.set_page_config(layout='wide')

    st.title('UCF Crimes Interactive Querying & Database Visualization')
    st.write('This web-app shows information gathered from the UCF Daily Crime Log from XXXXX to XXXXXX. It presents a Pandas dataframe of the data, a map of the incidents, and a few other visualizations. It also allows querying of the UCF crimes database using the PandasAI library. Ask a question using natural language and GPT will return the answer based on the dataset.')
    st.write('Made by Maverick Reynolds')
            
    df = pd.read_csv('UCFCrimes_Database_8-13.csv', index_col=0)

    hist_values = np.histogram(pd.to_datetime(df['report_dt']).dt.hour, bins=24, range=(0,24))[0]
        
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(hist_values)

    with col2:
        st.map(df.rename(columns={'lng': 'lon'})[['lat', 'lon']])
    
    query = st.text_input('Ask a question about the data')
    st.subheader('Query Response:')
    if query:
        llm = OpenAI()
        pdai = PandasAI(llm, conversational=True, verbose=True)
        response = pdai(df, query)
        st.write(response)

    st.write(df, use_column_width=True)
        

if __name__ == '__main__':
    main()
