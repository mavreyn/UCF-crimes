'''
Interactive using PandasAI to query incident data from database
Mapping and geolocation visualizted here as well

Maverick Reynolds
08.02.2023
'''

import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import numpy as np
from datetime import datetime
import dateparser

from embeddings import *
import json



def to_pydt(s):
    return dateparser.parse(s).replace(tzinfo=None)


def main():
    with open('title_embeddings.json', 'r') as f:
        embeddings_dict = json.load(f)

    st.set_page_config(layout='wide')

    # Main information about the app on the page
    st.title('UCF Crimes Interactive Querying & Database Visualization')
    st.markdown('**Made by Maverick Reynolds, Jack Sweeney, and Ethan Frakes**')
    st.write('This web-app shows information gathered from the UCF Daily Crime Log from January 27, 2023 to August 1, 2023. The project was developed over many months and is in its final stage of completion. The user may choose to filter data by tools in the sidebar (including title, date, and location) and the dashboard will update accordingly. The user may also choose to ask a question about the data using natural language and GPT (PandasAI) will return the answer based on the filtered dataset. Maps and other visualizations are also shown on the dashboard')
    st.markdown('---')

    # Make the sidebar
    st.sidebar.title('Settings :gear:')

    # Title search
    st.sidebar.subheader('Filter by title 📝')
    title_query = st.sidebar.text_input('Enter natural language query over crime titles', value='')
    if title_query:
        nresults = st.sidebar.slider('Num results', value=5, min_value=1, max_value=30)
    
    # Location
    st.sidebar.subheader('Filter by location 📍')
    location = st.sidebar.selectbox('Select a location', ['ALL', 'MAIN CAMPUS', 'UCF DOWNTOWN', 'ROSEN COLLEGE OF HOSPITALITY MANAGEMENT'])

    # Date search
    st.sidebar.subheader('Filter by date 📅')
    filter_dates = st.sidebar.checkbox('Enable date filter')
    if filter_dates:
        start_date = st.sidebar.date_input('Start date', value=datetime(2023, 1, 27))
        start_date = pd.Timestamp(start_date)
        end_date = st.sidebar.date_input('End date', value=datetime(2023, 8, 1))
        end_date = pd.Timestamp(end_date)

    # Apply filters
    df = pd.read_csv('UCFCrimes_Database_8-13.csv', index_col=0)
    if title_query:
        df = search_incident_titles(df, title_query, embeddings_dict, st.secrets['openai_api_key'], n=nresults)
    if location != 'ALL':
        df = df[df['campus'] == location]
    if filter_dates:
        if start_date > end_date:
            st.sidebar.error('End date must be after start date')
        else:
            df = df[(df['report_dt'].apply(to_pydt) >= start_date) & (df['report_dt'].apply(to_pydt) <= end_date)]

    # Show Dataframe
    st.subheader('Filtered Dataframe')
    st.write(df, use_column_width=True)

    hist_values = np.histogram(pd.to_datetime(df['report_dt']).dt.hour, bins=24, range=(0,24))[0]
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Incidents v. Time of Day')
        st.bar_chart(hist_values)

    with col2:
        st.write(len(df))
        st.map(df.rename(columns={'lng': 'lon'})[['lat', 'lon']])
    
    st.markdown('---')

    # Using PandasAI to make a query
    st.subheader('Make a Query')
    query = st.text_input('Use PandasAI to ask a question about the data')
    if query:
        llm = OpenAI(api_token=st.secrets['openai_api_key'])
        pdai = PandasAI(llm, conversational=True, verbose=True)
        response = pdai(df, query)
        st.write(response)


if __name__ == '__main__':
    main()
