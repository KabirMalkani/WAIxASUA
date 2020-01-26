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

dfCrime = pd.read_csv(r"C:\Users\nisha_000\Documents\GitHub\WAIxASUA\MCI_2014_to_2018.csv")
dfCensus = pd.read_csv(r"C:\Users\nisha_000\Documents\GitHub\WAIxASUA\neighbourhood-profiles-2016-csv.csv",index_col = 'Characteristic')
dfWellBeing = pd.read_csv(r"C:\Users\nisha_000\Documents\GitHub\WAIxASUA\wellbeing_toronto.csv")

#drop unwanted columns
dfCensus = dfCensus.drop(columns = ["_id",'Category','Topic','Data Source',"City of Toronto"]).T

#dfCityProfile2016 = dfCensus.join(dfWellBeing.set_index('Neighbourhood'))

#convert df into floats
for col in dfCensus.columns:
    try:
        dfCensus[col] = dfCensus[col].astype(float)
    except:
        pass

# return mean and median for given columns
def MeanMedianMode(subset):
    colNumber = np.arange(start = 1, stop =len(subset.columns) + 1)
    meanArray = np.empty(0)
    medianArray = np.empty(0)
    modeArray = []
    for i in range(0,len(subset.index)):
        freq = subset.values[i].astype(int)
        expandedRow = np.repeat(colNumber, freq)
        meanArray = np.append(meanArray,expandedRow.mean())
        medianArray = np.append(medianArray, np.median(expandedRow))

        #modeArray.append(mode(expandedRow))
        #print(mode(expandedRow))

    return [meanArray,medianArray]

#initialize the columns we want
subsetMale = dfCensus[dfCensus.columns[pd.Series(dfCensus.columns).str.startswith('Male')]]
subsetFemale = dfCensus[dfCensus.columns[pd.Series(dfCensus.columns).str.startswith('Female')]]
subsetIncomeDecile = dfCensus[dfCensus.columns[pd.Series(dfCensus.columns).str.contains('decile')]]
subsetImmigration = dfCensus[["  Canadian citizens", "  Not Canadian citizens", "  Non-immigrants", "  Immigrants", "  Non-permanent residents"]]
subsetImmigrationGen = dfCensus[["  First generation", "  Second generation"]]
subsetHighestEducation = dfCensus[["      Certificate of Apprenticeship or Certificate of Qualification", "    College, CEGEP or other non-university certificate or diploma", "    University certificate or diploma below bachelor level", "    University certificate, diploma or degree at bachelor level or above"]]

#create City profile
dfCityProfile2016 = dfCensus
oldCols = dfCensus.columns

subsetMaleMean, subsetMaleMedian = (MeanMedianMode(subsetMale))
#call function to get the results wanted
#*5-3 to get thl actua; bucket ages
dfCityProfile2016["Male Mean Age"] = np.round(subsetMaleMean*5-3,1)
dfCityProfile2016["Male Median Age"] = subsetMaleMedian*5-3


subsetFemaleMean, subsetFemaleMedian = (MeanMedianMode(subsetFemale))

dfCityProfile2016["Female Mean Age"] = np.round(subsetFemaleMean*5-3,1)
dfCityProfile2016["Female Median Age"] = subsetFemaleMedian*5-3


subsetIncomeDecileMean, subsetIncomeDecileMedian = (MeanMedianMode(subsetIncomeDecile))

dfCityProfile2016["Income Decile Mean"] = np.round(subsetIncomeDecileMean,1)
dfCityProfile2016["Income Decile Median"] = subsetIncomeDecileMedian

#delete the old columns we dont want
dfCityProfile2016 = dfCityProfile2016.drop(columns = oldCols)

#string all the columns we want together
dfCityProfile2016 = pd.concat([dfCityProfile2016,subsetImmigration,subsetImmigrationGen,subsetHighestEducation],axis = 1)

dfCityProfile2014 = dfWellBeing.drop(columns = ["Neighbourhood Id","Total Population"])

dfCityProfile2016.to_csv("City Profiles 2016.csv")
dfCityProfile2014.to_csv("City Profiles 2014.csv")
