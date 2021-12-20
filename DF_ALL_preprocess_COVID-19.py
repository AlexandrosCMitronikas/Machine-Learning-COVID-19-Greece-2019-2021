# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:16:49 2021

@author: AlexandrosCMitronikas
"""

import numpy as np
from sklearn import preprocessing
import pandas as pd
import sklearn
import scipy
from pandas import read_csv


#load COVID-19 dataset
df = pd.read_excel (r'D:\Personal\BSc Computer Science\6th Semester\Dissertation_Covid-19_Veropoulos\datasets\COVID-19_GREECE.xlsx')
#load mobility index dataset
df1 = pd.read_excel (r'D:\Personal\BSc Computer Science\6th Semester\Dissertation_Covid-19_Veropoulos\datasets\GoogleMAI_GREECE.xlsx')
#load GovResponseMeasures dataset
df2 = pd.read_excel (r'D:\Personal\BSc Computer Science\6th Semester\Dissertation_Covid-19_Veropoulos\datasets\GovResponseMeasures_GREECE.xlsx')


#merging datasets
df_all = pd.merge(df, df1,  on='date')
df_all = pd.merge(df_all, df2,  on='date')

  
# Count total NaN at each column in a DataFrame
print(" \nCount total NaN at each column in a DataFrame : \n\n",
    df_all.isnull().sum())

#removal of first NaN of chosen column
df_all = df_all[df_all['Reproduction rate'].notna()]

df_all = df_all[df_all['New tests'].notna()]


#all indexes with N/A replace to 0
df_all=df_all.fillna(0)


print(" \nCount total NaN at each column in a DataFrame : \n\n",
      df.isnull().sum())

df_all.columns

df_wPolicies = df_all[['New cases', 'Total cases', 'New deaths',
       'Total deaths', 'New cases / million', 'Total cases / million',
       'New deaths / million', 'Total deaths / million', 'Reproduction rate',
       'New tests', 'Total tests', 'New tests / thousand',
       'Total test / thousand', 'Tests per case', 'Positivity rate',
       'People vaccinated', 'People fully vaccinated', 'New vaccinations',
       'Total vaccinations', '% Total vaccinations', '% People vaccinated',
       '% People fully vacinated', 'stringency_index','retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline', 'C1_School closing',
       'C1_Duration', 'C2_Workplace closing', 'C2_Duration',
       'C3_Cancel public events', 'C3_Duration',
       'C4_Restrictions on gatherings', 'C4_Duration',
       'C5_Close public transport', 'C5_Duration',
       'C6_Stay at home requirements', 'C6_Duration',
       'C7_Restrictions on internal movement', 'C7_Duration',
       'C8_International travel controls', 'C8_Duration',
       'H1_Public information campaigns', 'H1_Duration', 'H2_Testing policy',
       'H2_Duration', 'H3_Contact tracing', 'H3_Duration',
       'H6_Facial Coverings', 'H6_Duration', 'H7_Vaccination policy',
       'H7_Duration', 'H8_Protection of elderly people', 'H8_Duration',
       'GovernmentResponseIndex', 'ContainmentHealthIndex']]

df_noPolicies = df_all[['New cases', 'Total cases', 'New deaths',
       'Total deaths', 'New cases / million', 'Total cases / million',
       'New deaths / million', 'Total deaths / million', 'Reproduction rate',
       'New tests', 'Total tests', 'New tests / thousand',
       'Total test / thousand', 'Tests per case', 'Positivity rate',
       'People vaccinated', 'People fully vaccinated', 'New vaccinations',
       'Total vaccinations', '% Total vaccinations', '% People vaccinated',
       '% People fully vacinated', 'stringency_index','retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']]

x_ForRr1 = df_all[['New deaths / million', 'Reproduction rate', '% People vaccinated', 
                    'New vaccinations', '% Total vaccinations', 'stringency_index' ]]

x_ForRr5 = df_all[['New deaths / million', 'Reproduction rate', 
                    'New vaccinations', '% People fully vacinated', 'stringency_index' ]]

