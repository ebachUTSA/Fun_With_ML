# -*- coding: utf-8 -*-
#Eric Bachura

### BEGIN IMPORT PORTION OF SCRIPT ###
import time
import pandas as pd
import numpy as np

from joblib import dump
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from src.utils import selectSQLPandas


### END IMPORT PORTION OF SCRIPT ###

### BEGIN FUNCTION DEFINITION PORTION OF SCRIPT ###
def getFeatures(dataT,dataH,xfeatureList,yfeature):
    '''
    This is a helper function for extracting numpy arrays representing X and y sets of features out of pandas dataframes
        Parameters:
            dataT (pandas df):                  the incoming pandas dataframe containing the training data features (X and y)
            dataH (pandas df):                  the incoming pandas dataframe containing the holdout/eval data features (X and y)
            xfeatureList (list of strings):     a list containing the case appropriate names of the X features to use (case sensitive as they will be pulled from the dataframes)
            yfeature (string):                  a string representing the case appropriate name of the y feature (case sensitive as it will be pulled from the dataframes)
        Returns:
            numpy arrays for the X training features, the X holdout feature, the y training feature, and the y holdout feature
    '''
    X = np.array(dataT.loc[:,xfeatureList]) #extract the X features from the training data
    XH = np.array(dataH.loc[:,xfeatureList]) #extract the X features from the holdout data
    y = np.array(dataT[yfeature].values) #extract the y feature out of the training data
    yH = np.array(dataH[yfeature].values) #extract the y feature out of the holdout data
    return (X, XH, y, yH) #return the numpy arrays of all of the necessary training and holdout data

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

def getClassifierPerformanceMetrics(yH,yHPred,yHPred_proba):
    '''
    This is a helper function for calculating the performance metrics for a classification model
        Parameters:
            yH (numpy array):                  the incoming y values from the holdout, these are the True values (labels)
            yHPred (numpy array):              the incoming y values the model predicted based on holdout x features, these are the predicted values and are the labels
            yHPred_proba (numpy array):        the incoming y values the representing the probabilities assocaited with model class predictions
        Returns:
            a set of performance measures for a classification model
    '''
    roc_auc = roc_auc_score(yH,yHPred_proba[:, 1]) #calculate the AUC score
    accuracy = accuracy_score(yH,yHPred) #calculate the accuracy
    return (roc_auc,accuracy) #return the performance metrics

def rf_scorer(dataT, dataH, xfeatureList, yfeature, seed=123, NT=1000, m=None, criterion='squared_error'):
    '''
    This is a wrapper function for the scikit-learn random forest regressor
    REF: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    NOTE: This wrapper function does not include all possible settings for the RandomForest Regressor class in Scikit-Learn and is intended to be a convenience function
        Parameters:
            dataT (pandas df):                  the incoming pandas dataframe containing the training data features (X and y)
            dataH (pandas df):                  the incoming pandas dataframe containing the holdout/eval data features (X and y)
            xfeatureList (list of strings):     a list containing the case appropriate names of the X features to use (case sensitive as they will be pulled from the dataframes)
            yfeature (string):                  a string representing the case appropriate name of the y feature (case sensitive as it will be pulled from the dataframes)
            seed (int):                         the number to use for the random_state value of the model, this allows for reproducibility
            NT (int):                           the number of trees to use in the forest
            m (int/string/None):                the number of features to consider during training, None = all possible freatures
            criterion (string):                 the function to use when evaluating the quality of a split
        Returns:
            trained randomforest regression model as well as the regression performance metric measures of r squared, RMSE, MSE, MAE, MedianAE, and Explained Variance
    '''
    X, XH, y, yH = getFeatures(dataT,dataH,xfeatureList,yfeature) #use the helper function to obtain the X and y feature sets
    model = RandomForestRegressor(n_estimators=NT, random_state=seed, oob_score=False, max_features=m, criterion=criterion) #instantiate the model, for this wrapper function I turn off out of bag samples (oob) explicitely so that the user knows, by defaul it is turned off anyway
    model.fit(X,y) #train the model using the training X features and evaluating against the training y feature
    yHPred = model.predict(XH) #obtain the trained model's predictions of what the y values should be based on the holdout X features (data it has never seen)
    mse,rmse,mae,medianae,r2,explainedvariance = getRegressorPerformanceMetrics(yH,yHPred)
    return (model,r2, rmse, mse, mae, medianae, explainedvariance) #return the model and the performance metrics

def svm_scorer(dataT, dataH, xfeatureList, yfeature, kernel='rbf', c=1.0, epsilon=0.1):
    '''
    This is a wrapper function for the scikit-learn Support Vector Machine regressor
    REF: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
    NOTE: This wrapper function does not include all possible settings for the Support Vector Machine Regressor class in Scikit-Learn and is intended to be a convenience function
        Parameters:
            dataT (pandas df):                  the incoming pandas dataframe containing the training data features (X and y)
            dataH (pandas df):                  the incoming pandas dataframe containing the holdout/eval data features (X and y)
            xfeatureList (list of strings):     a list containing the case appropriate names of the X features to use (case sensitive as they will be pulled from the dataframes)
            yfeature (string):                  a string representing the case appropriate name of the y feature (case sensitive as it will be pulled from the dataframes)
            kernel (string):                    the function to used in training (known as the kernel)
            c (float):                          regularization value, should be positive
            epsilon (float):                    the value specifying how close a predicted value has to be to truth before it isn't considered in training
        Returns:
            trained support vector machine regression model as well as the regression performance metric measures of r squared, RMSE, MSE, MAE, MedianAE, and Explained Variance
    '''
    X, XH, y, yH = getFeatures(dataT,dataH,xfeatureList,yfeature) #use the helper function to obtain the X and y feature sets
    model = SVR(kernel=kernel, C=c, epsilon=epsilon) #train the model, 
    model.fit(X,y)
    yHPred = model.predict(XH)
    mse,rmse,mae,medianae,r2,explainedvariance = getRegressorPerformanceMetrics(yH,yHPred)
    return (model,r2, rmse, mse, mae, medianae, explainedvariance)

def nn_scorer(dataT, dataH, xfeatureList, yfeature, hiddenlayers=(100,100),solver='adam',activator='relu',max_iterations=200):
    '''
    This is a wrapper function for the scikit-learn Neural Network regressor
    REF: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
    NOTE: This wrapper function does not include all possible settings for the Neural Network Regressor class in Scikit-Learn and is intended to be a convenience function
        Parameters:
            dataT (pandas df):                  the incoming pandas dataframe containing the training data features (X and y)
            dataH (pandas df):                  the incoming pandas dataframe containing the holdout/eval data features (X and y)
            xfeatureList (list of strings):     a list containing the case appropriate names of the X features to use (case sensitive as they will be pulled from the dataframes)
            yfeature (string):                  a string representing the case appropriate name of the y feature (case sensitive as it will be pulled from the dataframes)
            hiddenLayers (collection of ints):  the hidden layer definition, typically a tuple but could be a list, each entry in the tuple/list is an int representing the number of nodes the layer, e.g. (18,18) would indicate two hidden layers with 18 nodes each
            solver (string):                    the solving function to use (this function informs weight optimization)
            activator (string):                 the activation function to use (this function determines how the nodes activate)
            max_iterations (int):               the maximum number of iterations the model will go through during training
        Returns:
            trained neural network regression model as well as the regression performance metric measures of r squared, RMSE, MSE, MAE, MedianAE, and Explained Variance
    '''
    X, XH, y, yH = getFeatures(dataT,dataH,xfeatureList,yfeature) #use the helper function to obtain the X and y feature sets
    model = MLPRegressor(hidden_layer_sizes=hiddenlayers,solver=solver,activation=activator, max_iter=max_iterations)
    model.fit(X,y)
    yHPred = model.predict(XH)
    mse,rmse,mae,medianae,r2,explainedvariance = getRegressorPerformanceMetrics(yH,yHPred)
    return (model,r2, rmse, mse, mae, medianae, explainedvariance)

def rf_classifier(dataT, dataH, xfeatureList, yfeature, seed=123, NT=1000, m=None, criterion='gini'):
    '''
    This is a wrapper function for the scikit-learn random forest classifier
    REF: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    NOTE: This wrapper function does not include all possible settings for the RandomForest Classifier class in Scikit-Learn and is intended to be a convenience function
        Parameters:
            dataT (pandas df):                  the incoming pandas dataframe containing the training data features (X and y)
            dataH (pandas df):                  the incoming pandas dataframe containing the holdout/eval data features (X and y)
            xfeatureList (list of strings):     a list containing the case appropriate names of the X features to use (case sensitive as they will be pulled from the dataframes)
            yfeature (string):                  a string representing the case appropriate name of the y feature (case sensitive as it will be pulled from the dataframes)
            seed (int):                         the number to use for the random_state value of the model, this allows for reproducibility
            NT (int):                           the number of trees to use in the forest
            m (int/string/None):                the number of features to consider during training, None = all possible freatures
            criterion (string):                 the function to use when evaluating the quality of a split
        Returns:
            trained randomforest classifier model as well as the classifier performance metric measures of AUC and Accuracy
    '''
    X, XH, y, yH = getFeatures(dataT,dataH,xfeatureList,yfeature) #use the helper function to obtain the X and y feature sets
    model = RandomForestClassifier(n_estimators=NT, random_state=seed, oob_score=False, max_features=m, criterion=criterion)
    model.fit(X,y)
    yHPred = model.predict(XH)
    yHPred_proba = model.predict_proba(XH)
    roc_auc, accuracy = getClassifierPerformanceMetrics(yH,yHPred,yHPred_proba)
    return (model,roc_auc,accuracy)

def svm_classifier(dataT, dataH, xfeatureList, yfeature, kernel='rbf', c=1.0, prob=True):
    '''
    This is a wrapper function for the scikit-learn Support Vector Machine classifier
    REF: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    NOTE: This wrapper function does not include all possible settings for the Support Vector Machine Classifier class in Scikit-Learn and is intended to be a convenience function
        Parameters:
            dataT (pandas df):                  the incoming pandas dataframe containing the training data features (X and y)
            dataH (pandas df):                  the incoming pandas dataframe containing the holdout/eval data features (X and y)
            xfeatureList (list of strings):     a list containing the case appropriate names of the X features to use (case sensitive as they will be pulled from the dataframes)
            yfeature (string):                  a string representing the case appropriate name of the y feature (case sensitive as it will be pulled from the dataframes)
            kernel (string):                    the function to used in training (known as the kernel)
            c (float):                          regularization value, should be positive
            prob (bool):                        a bool indicating whether or not to calculate/approximate probabilities (slows down training, does NOT always perfectly align with classifications) ... has to do with how SVM's work
        Returns:
            trained Support Vector Machine Classifier model as well as the classifier performance metric measures of AUC and Accuracy
    '''
    X, XH, y, yH = getFeatures(dataT,dataH,xfeatureList,yfeature) #use the helper function to obtain the X and y feature sets
    model = SVC(kernel=kernel, C=c, probability=prob)
    model.fit(X,y)
    yHPred = model.predict(XH)
    yHPred_proba = model.predict_proba(XH)
    roc_auc, accuracy = getClassifierPerformanceMetrics(yH,yHPred,yHPred_proba)
    return (model,roc_auc,accuracy)

def nn_classifier(dataT, dataH, xfeatureList, yfeature, hiddenlayers=(100,100),solver='adam',activator='relu',max_iterations=200):
    '''
    This is a wrapper function for the scikit-learn Neural Network classifier
    REF: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    NOTE: This wrapper function does not include all possible settings for the Neural Network Classifier class in Scikit-Learn and is intended to be a convenience function
        Parameters:
            dataT (pandas df):                  the incoming pandas dataframe containing the training data features (X and y)
            dataH (pandas df):                  the incoming pandas dataframe containing the holdout/eval data features (X and y)
            xfeatureList (list of strings):     a list containing the case appropriate names of the X features to use (case sensitive as they will be pulled from the dataframes)
            yfeature (string):                  a string representing the case appropriate name of the y feature (case sensitive as it will be pulled from the dataframes)
            hiddenLayers (collection of ints):  the hidden layer definition, typically a tuple but could be a list, each entry in the tuple/list is an int representing the number of nodes the layer, e.g. (18,18) would indicate two hidden layers with 18 nodes each
            solver (string):                    the solving function to use (this function informs weight optimization)
            activator (string):                 the activation function to use (this function determines how the nodes activate)
            max_iterations (int):               the maximum number of iterations the model will go through during training
        Returns:
            trained Neural Network Classifier model as well as the classifier performance metric measures of AUC and Accuracy
    '''
    X, XH, y, yH = getFeatures(dataT,dataH,xfeatureList,yfeature) #use the helper function to obtain the X and y feature sets
    model = MLPClassifier(hidden_layer_sizes=hiddenlayers,solver=solver,activation=activator, max_iter=max_iterations)
    model.fit(X,y)
    yHPred = model.predict(XH)
    yHPred_proba = model.predict_proba(XH)
    roc_auc, accuracy = getClassifierPerformanceMetrics(yH,yHPred,yHPred_proba)
    return (model,roc_auc,accuracy)
### END FUNCTION DEFINITION PORTION OF SCRIPT ###

### BEGIN MAIN EXECUTION PORTION OF SCRIPT ###
baseDir = 'C:/Development/Fun_With_ML/' #set this to the directory you want to use as a base directory for everything, typically this is the folder where this script resides and in which you have a virtual environment set up (e.g. - the venv or menv folder)
fName = 'sample.xlsx' #variable designating the name of our data file, in this case I've created a sample set of data and called it sample.xlsx (it's an excel spreadsheet with 10k observations and a y that is a complex function of all of the x variables of which there are 7)

#NOTE: In the sample data I've provided, the input features are X1 to X7 and the yfeatures are either yr (for the regression values) or yc (for the classification values)
xFeatures = ('HARV_Positiv', 'HARV_Negativ', 'HARV_Pstv', 'HARV_Affil', 'HARV_Ngtv', 'HARV_Hostile', 'HARV_Strong', 'HARV_Power', 'HARV_Weak', 'HARV_Submit', 'HARV_Active', 'HARV_Passive', 'HARV_Pleasur', 'HARV_Pain', 'HARV_Feel', 'HARV_Arousal', 'HARV_EMOT', 'HARV_Virtue', 'HARV_Vice', 'HARV_Ovrst', 'HARV_Undrst', 'HARV_Academ', 'HARV_Doctrin', 'HARV_Econ2', 'HARV_Exch', 'HARV_ECON', 'HARV_Exprsv', 'HARV_Legal', 'HARV_Milit', 'HARV_Polit2', 'HARV_POLIT', 'HARV_Relig', 'HARV_Role', 'HARV_COLL', 'HARV_Work', 'HARV_Ritual', 'HARV_SocRel', 'HARV_Race', 'HARV_Kin2', 'HARV_MALE', 'HARV_Female', 'HARV_Nonadlt', 'HARV_HU', 'HARV_ANI', 'HARV_PLACE', 'HARV_Social', 'HARV_Region', 'HARV_Route', 'HARV_Aquatic', 'HARV_Land', 'HARV_Sky', 'HARV_Object', 'HARV_Tool', 'HARV_Food', 'HARV_Vehicle', 'HARV_BldgPt', 'HARV_ComnObj', 'HARV_NatObj', 'HARV_BodyPt', 'HARV_ComForm', 'HARV_COM', 'HARV_Say', 'HARV_Need', 'HARV_Goal', 'HARV_Try', 'HARV_Means', 'HARV_Persist', 'HARV_Complet', 'HARV_Fail', 'HARV_NatrPro', 'HARV_Begin', 'HARV_Vary', 'HARV_Increas', 'HARV_Decreas', 'HARV_Finish', 'HARV_Stay', 'HARV_Rise', 'HARV_Exert', 'HARV_Fetch', 'HARV_Travel', 'HARV_Fall', 'HARV_Think', 'HARV_Know', 'HARV_Causal', 'HARV_Ought', 'HARV_Perceiv', 'HARV_Compare', 'HARV_Eval2', 'HARV_EVAL', 'HARV_Solve', 'HARV_Abs2', 'HARV_ABS', 'HARV_Quality', 'HARV_Quan', 'HARV_NUMB', 'HARV_ORD', 'HARV_CARD', 'HARV_FREQ', 'HARV_DIST', 'HARV_Time2', 'HARV_TIME', 'HARV_Space', 'HARV_POS', 'HARV_DIM', 'HARV_Rel', 'HARV_COLOR', 'HARV_Self', 'HARV_Our', 'HARV_You', 'HARV_Name', 'HARV_Yes', 'HARV_No', 'HARV_Negate', 'HARV_Intrj', 'HARV_IAV', 'HARV_DAV', 'HARV_SV', 'HARV_IPadj', 'HARV_IndAdj', 'HARV_PowGain', 'HARV_PowLoss', 'HARV_PowEnds', 'HARV_PowAren', 'HARV_PowCon', 'HARV_PowCoop', 'HARV_PowAuPt', 'HARV_PowPt', 'HARV_PowDoct', 'HARV_PowAuth', 'HARV_PowOth', 'HARV_PowTot', 'HARV_RcEthic', 'HARV_RcRelig', 'HARV_RcGain', 'HARV_RcLoss', 'HARV_RcEnds', 'HARV_RcTot', 'HARV_RspGain', 'HARV_RspLoss', 'HARV_RspOth', 'HARV_RspTot', 'HARV_AffGain', 'HARV_AffLoss', 'HARV_AffPt', 'HARV_AffOth', 'HARV_AffTot', 'HARV_WltPt', 'HARV_WltTran', 'HARV_WltOth', 'HARV_WltTot', 'HARV_WlbGain', 'HARV_WlbLoss', 'HARV_WlbPhys', 'HARV_WlbPsyc', 'HARV_WlbPt', 'HARV_WlbTot', 'HARV_EnlGain', 'HARV_EnlLoss', 'HARV_EnlEnds', 'HARV_EnlPt', 'HARV_EnlOth', 'HARV_EnlTot', 'HARV_SklAsth', 'HARV_SklPt', 'HARV_SklOth', 'HARV_SklTot', 'HARV_TrnGain', 'HARV_TrnLoss', 'HARV_TranLw', 'HARV_MeansLw', 'HARV_EndsLw', 'HARV_ArenaLw', 'HARV_PtLw', 'HARV_Nation', 'HARV_Anomie', 'HARV_NegAff', 'HARV_PosAff', 'HARV_SureLw', 'HARV_If', 'HARV_NotLw', 'HARV_TimeSpc', 'HARV_FormLw', 'LR_WC', 'LR_UniqueWC', 'LR_TTR', 'LR_RTTR', 'LR_CTTR', 'LR_Herdan', 'LR_Summer', 'LR_Dugast', 'LR_Maas', 'LR_MSTTR', 'LR_MATTR', 'LR_MTLD', 'LR_HDD')
yFeatureC = 'is_resume'
yFeatureR = 'yr'

print("Loading data...")
df = selectSQLPandas('select * from craigslist_ml_training_data') #using pandas to read in the excel file with the data, if it was csv you'd use the read_csv method of pandas, or if it was sql you'd use read_sql
print("Data loaded!")

xFeatures = list(df.columns)
columns_to_remove = ['postid','is_resume','Raw_WC']
for col in columns_to_remove:
    xFeatures.remove(col)


fold = 0 #variable initiated at 0 to indicate the initial fold value
kf = KFold(n_splits = 5, shuffle = True, random_state = 123) # here we are initializing a k-fold object that splits our data for us into however many splits we specify, in each case we are essentially saying, use 80% of the data (4/5) to train on a fold and 20% to evaluate, we are shuffling each split and setting a random state for reproducibility
print("Entering k-fold iterations...")
for foldsplit in kf.split(df): #iterate over the folds
    train = df.iloc[foldsplit[0]] #get the training portion of the split
    test =  df.iloc[foldsplit[1]] #get the testing portion of the split
    
    '''
    Uncomment which model you want to try and feel free to adjust/change
    the performance metrics you want printed out for the scorer/regresser models
    I have left all hyperparameter settings to default, feel free to adjust
    and play around with different structures.
    
    Also note, that RandomForest has
    an advantage with this sample due to the fact that the scales of the input
    features vary significantly (some go into the hundreds, others are only decimals)
    
    If the data was normalized, you would see a fairer comparison, try to
    remember why that is (mentioned it in class).
    
    Also, notice that all models have an RMSE (or really any metric) that is
    terrible for a regression model, consider why that might be.
    '''
    
    # start = time.time()
    # rfc_model,roc_auc,accuracy = rf_classifier(train,test,xFeatures,yFeatureC)
    # stop = time.time()
    # print(f"RFC fold{fold}: AUC={roc_auc}, Accuracy={accuracy}, Time Taken={stop-start}")

    # start = time.time()
    # svmc_model,roc_auc,accuracy = svm_classifier(train,test,xFeatures,yFeatureC)
    # stop = time.time()
    # print(f"SVMC fold{fold}: AUC={roc_auc}, Accuracy={accuracy}, Time Taken={stop-start}")

    start = time.time()
    nnc_model,roc_auc,accuracy = nn_classifier(train,test,xFeatures,yFeatureC)
    stop = time.time()
    print(f"NNC fold{fold}: AUC={roc_auc}, Accuracy={accuracy}, Time Taken={stop-start}")

    # start = time.time()
    # rfs_model,r2, rmse, mse, mae, medianae, explainedvariance = rf_scorer(train,test,xFeatures,yFeatureR)
    # stop = time.time()
    # print(f"RFS fold{fold}: RMSE={rmse}, Time Taken={stop-start}")

    # start = time.time()
    # svms_model,r2, rmse, mse, mae, medianae, explainedvariance = svm_scorer(train,test,xFeatures,yFeatureR)
    # stop = time.time()
    # print(f"SVMS fold{fold}: RMSE={rmse}, Time Taken={stop-start}")

    # start = time.time()
    # nns_model,r2, rmse, mse, mae, medianae, explainedvariance = nn_scorer(train,test,xFeatures,yFeatureR)
    # stop = time.time()
    # print(f"NNS fold{fold}: RMSE={rmse}, Time Taken={stop-start}")
    
    
    fold += 1 #increment the split
    
'''
Use the following if you wanted to save a model
in this case, model is the variable representing the model
and fileLoc is the location on disk you want to save
I recommend giving the fileLoc an extension of .joblib to clearly
indicate that the file is a joblib binary
'''
#dump(model, fileLoc)

#### END SCRIPT ###