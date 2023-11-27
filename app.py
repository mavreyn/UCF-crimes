'''
Interactive using PandasAI to query incident data from database
Mapping and geolocation visualizted here as well

Maverick Reynolds
08.02.2023
'''

import numpy as np
import json
import streamlit as st
import pandas as pd

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from embeddings import *
import get_emojis


CATEGORY_DICT = {
    1: "Theft and Property Offenses",
    2: "Drug-Related Offenses",
    3: "Assault and Battery",
    4: "Traffic Offenses",
    5: "Stalking and Harassment",
    6: "Fraud and White-Collar Crimes",
    7: "Weapons and Firearm Offenses",
    8: "Miscellaneous Offenses"
}

CAMPUSES = ['MAIN CAMPUS',
            'UCF DOWNTOWN',
            'ROSEN COLLEGE OF HOSPITALITY MANAGEMENT',
            'UNSPECIFIED CAMPUS']

def main():
    #plt.style.use('dark_background')
    plt.style.use('ggplot')

    with open('title_embeddings.json', 'r') as f:
        embeddings_dict = json.load(f)

    st.set_page_config(layout='wide')

    # Main information about the app on the page
    st.title('UCF Incident Visualization and Interactive 👮📊')
    st.markdown('**Made by Maverick Reynolds, Jack Sweeney, and Ethan Frakes** | [GitHub](https://github.com/mavreyn/UCF-crimes)')
    st.write('UCF Crimes is a system developed to help students, faculty, and staff at the University of Central Florida stay informed about incidents on campus. This web-applet allows users to interact with the data gathered over the months by filtering and querying the data. In the sidebar, users can search by title (using NLP and language embeddings) or filter by location and the dashboard will update accordingly. Below the incidents table and map are some visualizations of the data, including a pie chart of incidents by category, a bar chart of incidents by day of week, and a line graph of incidents by week of year. which update based on the filters applied.')
    st.write('We conclude from our data that the most common offenses were related to theft and property, followed by traffic offenses and drug-related incidents. We also found evidence that the distribution of incidents (by category as determined by our language model) is independent of the campus location (main or downtown), meaning that the distribution of incidents is similar across campuses. We also note that more incidents seem to be reported later in the day, possibly due to more people simply being awake during those hours. Other conclusions and associations may be found as we continue to expand the project and gather more data over time.')
    st.markdown('---')

    # Read Dataframe
    df = pd.read_csv('UCFCrimes_Database_8-13.csv', index_col=0)
    # Add categorization to data (do before filtering)
    with open('categorization.json', 'r') as f:
        categorization = json.load(f)

    # Make the sidebar
    st.sidebar.image('UCFC_Logo_Dark_Fv3_Under.png', width=100)
    st.sidebar.title('Settings & Filters :gear:')

    # Title search
    st.sidebar.subheader('Filter by Title 📝')
    title_query = st.sidebar.text_input('Enter natural language query over crime titles', value='')
    if title_query:
        nresults = st.sidebar.slider('Num results', value=5, min_value=1, max_value=30)
    
    # Location
    st.sidebar.subheader('Filter by location 📍')
    location = st.sidebar.selectbox('Select a location', ['ALL', 'MAIN CAMPUS', 'UCF DOWNTOWN', 'ROSEN COLLEGE OF HOSPITALITY MANAGEMENT'])

    st.sidebar.markdown('---')

    # Using PandasAI to make a query
    st.sidebar.subheader('Ask PandasAI 🐼')
    query = st.sidebar.text_input('Use PandasAI to ask a question about the data (mostly experimental but still fun)')
    if query:
        llm = OpenAI(api_token=st.secrets['openai_api_key'])
        pdai = PandasAI(llm, conversational=True, verbose=True)
        response = pdai(df, query)
        st.sidebar.write(response)

    # Get Emojis
    st.sidebar.subheader('Get Emojis 😎')
    emoji_retrieval_title = st.sidebar.text_input('Enter a title to get the emoji string for it (as seen on the IG and Discord posts)', value='')
    if emoji_retrieval_title:
        st.sidebar.write(f'Emojis: {get_emojis.get_emojis(emoji_retrieval_title)}')

    # Apply filters
    df['category'] = categorization
    if title_query:
        df = search_incident_titles(df, title_query, embeddings_dict, st.secrets['openai_api_key'], n=nresults)
    if location != 'ALL':
        df = df[df['campus'] == location]

# ======================================= SHOW DATA ======================================= #

    col1, col2 = st.columns(2)
    with col1:
        # Show Dataframe
        st.subheader(f'Showing {len(df)} Incidents')
        col1.write(df.drop('category', axis='columns'), use_column_width=True)

    with col2:
        st.map(df.rename(columns={'lng': 'lon'})[['lat', 'lon']])
    
# ======================================= VISUALIZATIONS ======================================= #
    
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(25, 10))

    # Pie chart of incidents by category
    category_counts = df['category'].map(CATEGORY_DICT).value_counts()
    axs[0, 0].set_title('Incidents by Category', fontweight='bold')
    axs[0, 0].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', shadow=False, textprops={'fontsize': 10})
    axs[0, 0].axis('equal')


    # Pie chart by campus
    campus_counts = df['campus'].value_counts()
    axs[0, 1].set_title('Incidents by Campus', fontweight='bold')
    axs[0, 1].pie(campus_counts, labels=campus_counts.index, autopct='%1.1f%%', shadow=False, textprops={'fontsize': 9}, startangle=250)
    axs[0, 1].axis('equal')


    # Independence test between campus and category
    axs[1, 0].set_title('Fraction of Incidents by Category and Campus', fontweight='bold')
    if location == 'ALL':
        main_cat_counts = list(df[df['campus']=='MAIN CAMPUS']['category'].map(CATEGORY_DICT).value_counts().items())
        downtown_cat_counts = list(df[df['campus']=='UCF DOWNTOWN']['category'].map(CATEGORY_DICT).value_counts().items())
        # Proportion of total
        total_main = sum([x[1] for x in main_cat_counts])
        total_downtown = sum([x[1] for x in downtown_cat_counts])
        total_main = total_main if total_main != 0 else 1
        total_downtown = total_downtown if total_downtown != 0 else 1
        # Preprocessing
        bottom = np.zeros(2)
        width = 0.5
        for i in range(len(CATEGORY_DICT)):
            if CATEGORY_DICT[i+1] not in [x[0] for x in main_cat_counts]:
                main_cat_counts.append((CATEGORY_DICT[i+1], 0))
            if CATEGORY_DICT[i+1] not in [x[0] for x in downtown_cat_counts]:
                downtown_cat_counts.append((CATEGORY_DICT[i+1], 0))
        # Build the bar chart
        for i in range(len(CATEGORY_DICT)):
            main_entry = main_cat_counts[i][1]/total_main
            downtown_entry = downtown_cat_counts[i][1]/total_downtown

            axs[1, 0].bar(('Main Campus', 'Downtown Campus'), (main_entry, downtown_entry), width, bottom=bottom, label=main_cat_counts[i][0])
            bottom += (main_entry, downtown_entry)
        axs[1, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1))


    # Bar chart of incidents by day of week
    df['weekday'] = pd.to_datetime(df['report_dt']).dt.day_name()
    day_counts = dict(df['weekday'].value_counts().items())
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    axs[1, 1].set_title('Fraction of Incidents by Day of Week (report_dt)', fontweight='bold')
    # Preprocessing
    for i in range(len(weekday_order)):
        if weekday_order[i] not in day_counts:
            day_counts[weekday_order[i]] = 0
    axs[1, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axs[1, 1].bar(weekday_order, [day_counts[day]/len(df) for day in weekday_order], color='green')


    # Line graph of incidents per week
    axs[0, 2].set_title('Incidents by Week of Year', fontweight='bold')
    df['report_dt_pddt'] = pd.to_datetime(df['report_dt'])
    df['week'] = df['report_dt_pddt'].dt.strftime('%Y-%U')
    weekly_counts = df['week'].value_counts().sort_index()
    # Some tick marks
    axs[0, 2].set_xticks(np.arange(0, len(weekly_counts), 4))
    axs[0, 2].set_yticks(np.arange(0, max(weekly_counts)+1, 5))
    axs[0, 2].plot(weekly_counts, color='orange', marker='o')


    # Bar chart of incidents by hour of day
    axs[1, 2].set_title('Incidents by Hour of Day (report_dt)', fontweight='bold')
    df['hour'] = pd.to_datetime(df['report_dt']).dt.hour
    axs[1, 2].hist(df['hour'], bins=24)

    # Plot the subplots
    st.pyplot(fig)


if __name__ == '__main__':
    main()
