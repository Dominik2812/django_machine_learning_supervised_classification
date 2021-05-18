import numpy as np
import pandas as pd
import random

# read the model data
baseData = pd.read_csv('./iris.csv') 
#check which col contains the nominal values and remove it
print(baseData)

features =[col for col in baseData.columns]
features.remove('classification')
baseDataValues= baseData.loc[:, features]

#################################################
############# create the testDataset ############
#################################################

deviationFactor=0.1 # the bigger, the more the data deviates from it original value
fractionFactor=0.9 # partition in percent of the original Dataset size
baseDataValues=baseDataValues.to_numpy()
randomChoice=[]
for _ in range(int(baseDataValues.shape[0]*fractionFactor)):
    rand_sample=random.randint(0,baseDataValues.shape[0]-1)
    randomChoice.append(baseDataValues[rand_sample])
baseDataValues=np.array(randomChoice)
testArray=baseDataValues*(1+ deviationFactor*np.random.random_sample((baseDataValues.shape[0],baseDataValues.shape[1])))


frame=pd.DataFrame(testArray, columns=features)
print(frame)

#name of your artificial test dataset
frame.to_csv('./iris-testData-10perc.csv')
