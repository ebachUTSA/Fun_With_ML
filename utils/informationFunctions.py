import math

def getEntropy(df,col='Target'):
    '''
    This function calculates the entropy of the passed in col (column)
    for the passed in df (dataframe/pandas)
    
    To do this you need to:
    1 - Identify all of the unique states (values) - for dataframes we can all df[col].unique()
    2 - For each unique value, you need to calculate it's proportion (all proportions should sum to 1)
    3 - To obtain entropy of the total system, do the following
    4 - Multiply the proportion by log2 proportion
    5 - Do the above (step 4), for each of the unique values (states)
    6 - Sum the results of the above two steps (4,5)
    7 - Negate that sum
    '''
    totalCount = df.shape[0] #len(df[col]) len([x for _,x in df[col].iterrows()])
    uniqueVals = df[col].unique() #returns a list of the unique values for the col in the df
    entropy = 0 #initialize my entropy value at 0 (we will be doing a series of summations)
    for uniqueVal in uniqueVals: #looping construct to iterate over the uniqueValues found in the passed in col of the passed in df
        valCount = len(df[(df[col]==uniqueVal)]) #count how many values in the col of the df are equal to the uniqueVal
        entropy += (valCount/totalCount)*math.log2((valCount/totalCount)) #THIS is the entropy member (calculation of the member entropy)
    entropy *= -1 #flip the sign
    return entropy #return the entropy value for col in df

def getInformationGain(df,splitcol,target):
    '''
    This function returns the information gain value that would be obtained
    by splitting on the passed in splitcol (the column to use as a decision branch)
    This function necessarily calls the getEntropy function
    '''
    startingEntropy = getEntropy(df,target) #calculating the current state of entropy of the target
    uniqueVals = df[splitcol].unique() #obtaining all of the unique values that exist in the split column (the column to split on)
    splits = [] #we are initializing our empty list to hold all split entropy values
    for uniqueVal in uniqueVals: #looping through the unique values in the column to split on
        split = df[df[splitcol] == uniqueVal] #obtain the dataframe corresponding to only those values in the split column that are of the current unique value
        splits.append(split) #push that dataframe reference into the splits list
    newEntropy = 0
    for split in splits:
        prob = (split.shape[0] / df.shape[0])
        newEntropy += prob * getEntropy(split,target)
    infoGain = startingEntropy - newEntropy
    return infoGain