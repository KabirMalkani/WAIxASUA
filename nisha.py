# -*- coding: utf-8 -*-

import pandas as pd
import itertools
import seaborn as sns
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from IPython.display import display
from statistics import mode

#read in csvs
dfFullData = pd.read_csv(r"C:\Users\nisha_000\Documents\GitHub\WAIxASUA\full_data.csv", index_col="Index_")
dfCensus = pd.read_csv(r"C:\Users\nisha_000\Documents\GitHub\WAIxASUA\neighbourhood-profiles-2016-csv (4).csv",index_col = 'Characteristic')
dfWellBeing = pd.read_csv(r"C:\Users\nisha_000\Documents\GitHub\WAIxASUA\wellbeing_toronto.csv",index_col = "Neighbourhood Id")

#drop some unwanted columns for the census, and transpose
dfCensus = dfCensus.drop(columns = ["_id",'Category','Topic','Data Source']).T

#convert df into floats to manipulate easily
for col in dfCensus.columns:
    try:
        dfCensus[col] = dfCensus[col].astype(float)
    except:
        pass

#return mean and median for given columns
def MeanMedianMode(subset):
    colNumber = np.arange(start = 1, stop =len(subset.columns) + 1)
    meanArray = np.empty(0)
    medianArray = np.empty(0)

    for i in range(0,len(subset.index)):
        freq = subset.values[i].astype(int)
        expandedRow = np.repeat(colNumber, freq)
        meanArray = np.append(meanArray,expandedRow.mean())
        medianArray = np.append(medianArray, np.median(expandedRow))
    return [meanArray,medianArray]

#initialize the columns we want

subsetAgePop = dfCensus[["Children (0-14 years)", "Youth (15-24 years)","Working Age (25-54 years)","Pre-retirement (55-64 years)"]]
#removed for redundancy
#subsetSeniors= pd.DataFrame(dfCensus.apply(lambda x:x["Seniors (65+ years)"] + x["Older Seniors (85+ years)"], axis =1), columns = ['Seniors and Older Seniors'])
subsetAgePop = subsetAgePop.div(dfCensus["Population, 2016"],axis = 0)

subsetIncomeDecile = dfCensus[dfCensus.columns[pd.Series(dfCensus.columns).str.contains('decile')]]
subsetTotalResponseIncome = dfCensus["Total - Economic family income decile group for the population in private households - 100% data"]
subsetIncomeDecile = subsetIncomeDecile.drop(columns = "Total - Economic family income decile group for the population in private households - 100% data")
oldIncomeCols = subsetIncomeDecile.columns

subsetIncomeDecile.insert(1, "Income Decile 1-2 as % of total response", subsetIncomeDecile.iloc[:,[0,1]].sum(axis = 1))
subsetIncomeDecile.insert(2,"Income Decile 3-4 as % of total response", subsetIncomeDecile.iloc[:,[2,3]].sum(axis = 1))
subsetIncomeDecile.insert(3,"Income Decile 5-6 as % of total response",subsetIncomeDecile.iloc[:,[4,5]].sum(axis = 1))
subsetIncomeDecile.insert(4,"Income Decile 7-8 as % of total response", subsetIncomeDecile.iloc[:,[6,7]].sum(axis = 1))
#removed due to redundancy
#subsetIncomeDecile.insert(5,"Income Decile 9-10",subsetIncomeDecile.iloc[:,[8,9]].sum(axis = 1)/subsetTotalResponseIncome)

subsetIncomeDecile = subsetIncomeDecile.drop(columns = oldIncomeCols)
subsetIncomeDecile = subsetIncomeDecile.div(subsetTotalResponseIncome,axis = 0)

subsetTotalImmigration = dfCensus["Total - Citizenship for the population in private households - 25% sample data"]

subsetImmigrantType= dfCensus[["  Economic immigrants", "  Immigrants sponsored by family", "  Refugees", "  Other immigrants"]].div(subsetTotalImmigration,axis =0)
subsetImmigrationStatus = dfCensus[["  Non-immigrants","  Immigrants", "  Non-permanent residents"]].div(subsetTotalImmigration,axis =0)
subsetImmigrationGen = dfCensus[["  First generation", "  Second generation", "  Third generation or more"]].div(subsetTotalImmigration,axis =0)
subsetCitizenship = dfCensus[["  Canadian citizens"]].div(subsetTotalImmigration,axis =0)


subsetHighestEducation = dfCensus[["  Postsecondary certificate, diploma or degree","  No certificate, diploma or degree","  Secondary (high) school diploma or equivalency certificate"]]
subsetTotalEducation = subsetHighestEducation.sum(axis = 1)
subsetHighestEducation = subsetHighestEducation.div(subsetTotalEducation,axis =0)

subsetTotalPop = dfCensus["Population, 2016"]
subsetNeighbourhoodID= dfCensus["Neighbourhood Number"]

#need to sum this
subsetMale = pd.DataFrame(dfCensus[dfCensus.columns[pd.Series(dfCensus.columns).str.contains('Male:')]])
oldCols = subsetMale.columns
subsetMale.insert(0,"Male as % of total Population",subsetMale.sum(axis =1))
subsetMale = subsetMale.drop(columns= oldCols)
subsetMale = subsetMale.div(subsetTotalPop,axis = 0)

#create City profile
dfCityProfile2016 = dfCensus
oldCols = dfCensus.columns

#call function to get the results wanted

dfCityProfile2016["neighbourhood ID"] = subsetNeighbourhoodID
dfCityProfile2016["Total Pop 2016"] = subsetTotalPop

#delete the old columns we dont want
dfCityProfile2016 = dfCityProfile2016.drop(columns = oldCols)

#string all the columns we want together
dfCityProfile2016 = pd.concat([dfCityProfile2016,subsetImmigrationStatus,subsetImmigrationGen,subsetCitizenship,subsetHighestEducation, subsetMale,subsetAgePop],axis = 1)

#clean the other dataset
dfCityProfile2014 = dfWellBeing.drop(columns = ["Reference Period","Total Population"])

#dfCityProfile2016.to_csv("City Profiles 2016.csv")
#dfCityProfile2014.to_csv("City Profiles 2014.csv")

#join the datasets
dfCityProfile = dfCityProfile2014.join(dfCityProfile2016.set_index('neighbourhood ID'))

#save this new dataset as a csv file
#dfCityProfile.to_csv("City Profiles.csv")

crimeData = pd.merge(dfFullData, dfCityProfile, left_on="Hood_ID", right_index=True)
crimeData = crimeData.drop(columns=["Unnamed: 0","Neighbourhood_y"])
crimeData.to_csv("All Data.csv")
