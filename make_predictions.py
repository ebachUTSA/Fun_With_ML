# -*- coding: utf-8 -*-
#Eric Bachura

### BEGIN IMPORT PORTION OF SCRIPT ###
import time
import random
import pandas as pd
import numpy as np

from joblib import load
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold


### END IMPORT PORTION OF SCRIPT ###

### BEGIN FUNCTION DEFINITION PORTION OF SCRIPT ###
def getRegressorPerformanceMetrics(yH,yHPred):
    '''
    This is a helper function for calculating the performance metrics for a regression model
        Parameters:
            yH (numpy array):                  the incoming y values from the holdout, these are the True values
            yHPred (numpy array):              the incoming y values the model predicted based on holdout x features, these are the predicted values
        Returns:
            a set of performance measures for a regression model
    '''
    mse = mean_squared_error(yH,yHPred) #calculate MSE using the model's predictions on the holdout against the actual holdout y values
    rmse = mean_squared_error(yH,yHPred,squared=False) #calculate RMSE using the model's predictions on the holdout against the actual holdout y values
    mae = mean_absolute_error(yH,yHPred) #calculate MAE using the model's predictions on the holdout against the actual holdout y values
    medianae = median_absolute_error(yH,yHPred) #calculate MedianAE using the model's predictions on the holdout against the actual holdout y values
    r2 = r2_score(yH,yHPred) #calculate R-sqaured value using the model's predictions on the holdout against the actual holdout y values
    explainedvariance = explained_variance_score(yH,yHPred) #calculate models explained variance using the model's predictions on the holdout against the actual holdout y values
    return (mse,rmse,mae,medianae,r2,explainedvariance) #return the performance metrics

def getPredictions(modelPath, data, xfeatureList):
    X = np.array(data.loc[:,xfeatureList]) #extract the X features from the training data
    model = load(modelPath)
    yHPred = model.predict(X) #obtain the trained model's predictions of what the y values should be based on the holdout X features (data it has never seen)
    return yHPred
### END FUNCTION DEFINITION PORTION OF SCRIPT ###

### BEGIN MAIN EXECUTION PORTION OF SCRIPT ###
baseDir = 'C:/Development/Fun_With_ML/' #set this to the directory you want to use as a base directory for everything, typically this is the folder where this script resides and in which you have a virtual environment set up (e.g. - the venv or menv folder)
fName = 'sample.xlsx' #variable designating the name of our data file, in this case I've created a sample set of data and called it sample.xlsx (it's an excel spreadsheet with 10k observations and a y that is a complex function of all of the x variables of which there are 7)
modelName = 'best_nns.joblib'
#NOTE: In the sample data I've provided, the input features are X1 to X7 and the yfeatures are either yr (for the regression values) or yc (for the classification values)
xFeatures = ('x1','x2','x3','x4','x5','x6','x7')
yFeatureC = 'yc'
yFeatureR = 'yr'

print("Loading data...")
df = pd.read_excel(f"{baseDir}/data/{fName}") #using pandas to read in the excel file with the data, if it was csv you'd use the read_csv method of pandas, or if it was sql you'd use read_sql
print("Data loaded!")

modelPath = f"{baseDir}/output/{modelName}"

predY = getPredictions(modelPath=modelPath,data=df,xfeatureList=xFeatures)
df['predY'] = predY

df.to_excel(f"{baseDir}/output/predictions_from_{modelName}.xlsx")

#### END SCRIPT ###