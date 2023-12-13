import re
from string import punctuation
from lexicalrichness import LexicalRichness as lr
import pandas as pd
import numpy as np
import librosa
from src.config import Config
config = Config()

### CONSTANTS ###
DICTIONARIES_LOC = f'{config.base_directory}/data' #location of dictionary files, relative to this script
DICTIONARIES = ['hv4.dic'] #names of dictionary files
LR_FEATURES = ['LR_WC','LR_UniqueWC','LR_TTR','LR_RTTR','LR_CTTR','LR_Herdan','LR_Summer','LR_Dugast','LR_Maas','LR_MSTTR','LR_MATTR','LR_MTLD','LR_HDD']
SPECTRAL_FEATURES = ['chroma_stft_avg','rmse_avg','spec_cent_avg','spec_bw_avg','rolloff_avg','zcr_avg','mfcc1_avg','mfcc2_avg','mfcc3_avg','mfcc4_avg','mfcc5_avg','mfcc6_avg','mfcc7_avg','mfcc8_avg','mfcc9_avg','mfcc10_avg','mfcc11_avg','mfcc12_avg','mfcc13_avg','mfcc14_avg','mfcc15_avg','mfcc16_avg','mfcc17_avg','mfcc18_avg','mfcc19_avg','mfcc20_avg','mfcc21_avg','mfcc22_avg','mfcc23_avg','mfcc24_avg','mfcc25_avg','mfcc26_avg','mfcc27_avg','mfcc28_avg','mfcc29_avg','mfcc30_avg','chroma_stft_std','rmse_std','spec_cent_std','spec_bw_std','rolloff_std','zcr_std','mfcc1_std','mfcc2_std','mfcc3_std','mfcc4_std','mfcc5_std','mfcc6_std','mfcc7_std','mfcc8_std','mfcc9_std','mfcc10_std','mfcc11_std','mfcc12_std','mfcc13_std','mfcc14_std','mfcc15_std','mfcc16_std','mfcc17_std','mfcc18_std','mfcc19_std','mfcc20_std','mfcc21_std','mfcc22_std','mfcc23_std','mfcc24_std','mfcc25_std','mfcc26_std','mfcc27_std','mfcc28_std','mfcc29_std','mfcc30_std']
### FUNCTION DEFINITION SECTION ###
def importDictionaries():
    myDVars = {}
    myDVarParts = {}
    myDVarsList = []
    maxval = -1 #set to negative 1 to force 0 based index alignment in data structures
    for dictionary in DICTIONARIES:
        fName = DICTIONARIES_LOC+'/'+dictionary
        with open(fName) as dic:
            catFlag = False
            for line in dic:
                if (re.match(r'^%',line)):
                    if (not catFlag):
                        catFlag = True
                    else:
                        catFlag = False
                    continue
                myLineList = line.replace('\n','').split('\t')
                if catFlag:
                    #We are in the feature specification portion of the dictionary
                    myDVars[int(myLineList[0])+maxval] = 0
                    myDVarsList.append(myLineList[-1])
                else:
                    token = re.sub(f"[{re.escape(punctuation)}]", "", myLineList[0]).lower()  # Remove punctuation
                    if token not in myDVarParts:
                        myDVarParts[token] = []
                    for varnum in myLineList[1:]:
                        if varnum != '':
                            myDVarParts[token].append(int(varnum)+maxval)
        maxval = max(myDVars.keys())
    return [myDVars, myDVarParts, myDVarsList]

def getDicFeatures(text, myDVars, myDVarParts, myDVarsList, result):
    wordCount = 0
    myDVars2 = myDVars.copy()
    text = preprocessText(text)
    for word in text:
        wordCount += 1
        word = word.strip().lower()
        if (myDVarParts.get(word)):
            for varNum in myDVarParts[word]:
                myDVars2[varNum] += 1
    result['Raw_WC'].append(wordCount)
    for myKey,myVal in myDVars2.items():
        if wordCount == 0:
            result[myDVarsList[myKey]].append(0)
        else:
            result[myDVarsList[myKey]].append(myVal/wordCount)
    return result

def getLRFeatures(text,result):
    lex = lr(text)
    result['LR_WC'].append(lex.words)
    result['LR_UniqueWC'].append(lex.terms)
    try:
        result['LR_TTR'].append(lex.ttr)
    except Exception as e:
        result['LR_TTR'].append(0)
    try:
        result['LR_RTTR'].append(lex.rttr)
    except Exception as e:
        result['LR_RTTR'].append(0)
    try:
        result['LR_CTTR'].append(lex.cttr)
    except Exception as e:
        result['LR_CTTR'].append(0)
    try:
        result['LR_MSTTR'].append(lex.msttr(segment_window=10))
    except Exception as e:
        result['LR_MSTTR'].append(0)
    try:
        result['LR_MATTR'].append(lex.mattr(window_size=10))
    except Exception as e:
        result['LR_MATTR'].append(0)
    try:
        result['LR_HDD'].append(lex.hdd(draws=15))
    except Exception as e:
        result['LR_HDD'].append(0)
    try:
        result['LR_MTLD'].append(lex.mtld(threshold=0.72))
    except Exception as e:
        result['LR_MTLD'].append(0)
    try:
        result['LR_Herdan'].append(lex.Herdan)
    except Exception as e:
        result['LR_Herdan'].append(0)
    try:
        result['LR_Summer'].append(lex.Summer)
    except Exception as e:
        result['LR_Summer'].append(0)
    try:
        result['LR_Dugast'].append(lex.Dugast)
    except Exception as e:
        result['LR_Dugast'].append(0)
    try:
        result['LR_Maas'].append(lex.Maas)
    except Exception as e:
        result['LR_Maas'].append(0)
    return result

def preprocessText(text):
    text = text.lower()  # Lowercase text
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)  # Remove punctuation
    text = text.split()  # Split text into list
    return text

def generateTextFeatures(df,textCol='text'):
    myDVars, myDVarParts, myDVarsList = importDictionaries()
    result = {'Raw_WC':[]} #initialize the result dictionary
    for myKey,myVal in myDVars.items():
        result[myDVarsList[myKey]]=[] #add each dictionary variable into the result dictionary
    for lrFeature in LR_FEATURES:
        result[lrFeature] = [] #add each lr variable into the result dictionary
    for _,row in df.iterrows():
        result = getDicFeatures(row[textCol],myDVars, myDVarParts, myDVarsList,result) #get the dictionary based features
        result = getLRFeatures(row[textCol],result)
    
    resultdf = pd.DataFrame(result)
    return pd.concat([df,resultdf],axis=1)

def generateAudioFeatures_fulldf(df,fname_col: str='fname',begin_ts_col: str='begin_ts',end_ts_col: str='end_ts') -> pd.DataFrame:
    result = {}
    for spectralFeature in SPECTRAL_FEATURES:
        result[spectralFeature] = []
    for _,row in df.iterrows():
        result = generateAudioFeatures(row[fname_col],row[begin_ts_col],row[end_ts_col],result)
    resultdf = pd.DataFrame(result)
    return pd.concat([df,resultdf],axis=1)

def generateAudioFeatures(fname,begin_ts,end_ts,result):
    duration = end_ts-begin_ts
    y, sr = librosa.load(fname, sr=None, mono=True, offset=begin_ts, duration=duration)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    result['chroma_stft_avg'].append(float(np.mean(chroma_stft)))
    result['rmse_avg'].append(float(np.mean(rmse)))
    result['spec_cent_avg'].append(float(np.mean(spec_cent)))
    result['spec_bw_avg'].append(float(np.mean(spec_bw)))
    result['rolloff_avg'].append(float(np.mean(rolloff)))
    result['zcr_avg'].append(float(np.mean(zcr)))
    result['mfcc1_avg'].append(float(np.mean(mfcc[0])))
    result['mfcc2_avg'].append(float(np.mean(mfcc[1])))
    result['mfcc3_avg'].append(float(np.mean(mfcc[2])))
    result['mfcc4_avg'].append(float(np.mean(mfcc[3])))
    result['mfcc5_avg'].append(float(np.mean(mfcc[4])))
    result['mfcc6_avg'].append(float(np.mean(mfcc[5])))
    result['mfcc7_avg'].append(float(np.mean(mfcc[6])))
    result['mfcc8_avg'].append(float(np.mean(mfcc[7])))
    result['mfcc9_avg'].append(float(np.mean(mfcc[8])))
    result['mfcc10_avg'].append(float(np.mean(mfcc[9])))
    result['mfcc11_avg'].append(float(np.mean(mfcc[10])))
    result['mfcc12_avg'].append(float(np.mean(mfcc[11])))
    result['mfcc13_avg'].append(float(np.mean(mfcc[12])))
    result['mfcc14_avg'].append(float(np.mean(mfcc[13])))
    result['mfcc15_avg'].append(float(np.mean(mfcc[14])))
    result['mfcc16_avg'].append(float(np.mean(mfcc[15])))
    result['mfcc17_avg'].append(float(np.mean(mfcc[16])))
    result['mfcc18_avg'].append(float(np.mean(mfcc[17])))
    result['mfcc19_avg'].append(float(np.mean(mfcc[18])))
    result['mfcc20_avg'].append(float(np.mean(mfcc[19])))
    result['mfcc21_avg'].append(float(np.mean(mfcc[20])))
    result['mfcc22_avg'].append(float(np.mean(mfcc[21])))
    result['mfcc23_avg'].append(float(np.mean(mfcc[22])))
    result['mfcc24_avg'].append(float(np.mean(mfcc[23])))
    result['mfcc25_avg'].append(float(np.mean(mfcc[24])))
    result['mfcc26_avg'].append(float(np.mean(mfcc[25])))
    result['mfcc27_avg'].append(float(np.mean(mfcc[26])))
    result['mfcc28_avg'].append(float(np.mean(mfcc[27])))
    result['mfcc29_avg'].append(float(np.mean(mfcc[28])))
    result['mfcc30_avg'].append(float(np.mean(mfcc[29])))
    result['chroma_stft_std'].append(float(np.std(chroma_stft)))
    result['rmse_std'].append(float(np.std(rmse)))
    result['spec_cent_std'].append(float(np.std(spec_cent)))
    result['spec_bw_std'].append(float(np.std(spec_bw)))
    result['rolloff_std'].append(float(np.std(rolloff)))
    result['zcr_std'].append(float(np.std(zcr)))
    result['mfcc1_std'].append(float(np.std(mfcc[0])))
    result['mfcc2_std'].append(float(np.std(mfcc[1])))
    result['mfcc3_std'].append(float(np.std(mfcc[2])))
    result['mfcc4_std'].append(float(np.std(mfcc[3])))
    result['mfcc5_std'].append(float(np.std(mfcc[4])))
    result['mfcc6_std'].append(float(np.std(mfcc[5])))
    result['mfcc7_std'].append(float(np.std(mfcc[6])))
    result['mfcc8_std'].append(float(np.std(mfcc[7])))
    result['mfcc9_std'].append(float(np.std(mfcc[8])))
    result['mfcc10_std'].append(float(np.std(mfcc[9])))
    result['mfcc11_std'].append(float(np.std(mfcc[10])))
    result['mfcc12_std'].append(float(np.std(mfcc[11])))
    result['mfcc13_std'].append(float(np.std(mfcc[12])))
    result['mfcc14_std'].append(float(np.std(mfcc[13])))
    result['mfcc15_std'].append(float(np.std(mfcc[14])))
    result['mfcc16_std'].append(float(np.std(mfcc[15])))
    result['mfcc17_std'].append(float(np.std(mfcc[16])))
    result['mfcc18_std'].append(float(np.std(mfcc[17])))
    result['mfcc19_std'].append(float(np.std(mfcc[18])))
    result['mfcc20_std'].append(float(np.std(mfcc[19])))
    result['mfcc21_std'].append(float(np.std(mfcc[20])))
    result['mfcc22_std'].append(float(np.std(mfcc[21])))
    result['mfcc23_std'].append(float(np.std(mfcc[22])))
    result['mfcc24_std'].append(float(np.std(mfcc[23])))
    result['mfcc25_std'].append(float(np.std(mfcc[24])))
    result['mfcc26_std'].append(float(np.std(mfcc[25])))
    result['mfcc27_std'].append(float(np.std(mfcc[26])))
    result['mfcc28_std'].append(float(np.std(mfcc[27])))
    result['mfcc29_std'].append(float(np.std(mfcc[28])))
    result['mfcc30_std'].append(float(np.std(mfcc[29])))
    return result

def generateAudioFeatures_old(fname,begin_ts,end_ts):
    duration = end_ts-begin_ts
    spectralFeatures = {}
    y, sr = librosa.load(fname, sr=None, mono=True, offset=begin_ts, duration=duration)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    spectralFeatures['chroma_stft_avg'] = float(np.mean(chroma_stft))
    spectralFeatures['rmse_avg'] = float(np.mean(rmse))
    spectralFeatures['spec_cent_avg'] = float(np.mean(spec_cent))
    spectralFeatures['spec_bw_avg'] = float(np.mean(spec_bw))
    spectralFeatures['rolloff_avg'] = float(np.mean(rolloff))
    spectralFeatures['zcr_avg'] = float(np.mean(zcr))
    spectralFeatures['mfcc1_avg'] = float(np.mean(mfcc[0]))
    spectralFeatures['mfcc2_avg'] = float(np.mean(mfcc[1]))
    spectralFeatures['mfcc3_avg'] = float(np.mean(mfcc[2]))
    spectralFeatures['mfcc4_avg'] = float(np.mean(mfcc[3]))
    spectralFeatures['mfcc5_avg'] = float(np.mean(mfcc[4]))
    spectralFeatures['mfcc6_avg'] = float(np.mean(mfcc[5]))
    spectralFeatures['mfcc7_avg'] = float(np.mean(mfcc[6]))
    spectralFeatures['mfcc8_avg'] = float(np.mean(mfcc[7]))
    spectralFeatures['mfcc9_avg'] = float(np.mean(mfcc[8]))
    spectralFeatures['mfcc10_avg'] = float(np.mean(mfcc[9]))
    spectralFeatures['mfcc11_avg'] = float(np.mean(mfcc[10]))
    spectralFeatures['mfcc12_avg'] = float(np.mean(mfcc[11]))
    spectralFeatures['mfcc13_avg'] = float(np.mean(mfcc[12]))
    spectralFeatures['mfcc14_avg'] = float(np.mean(mfcc[13]))
    spectralFeatures['mfcc15_avg'] = float(np.mean(mfcc[14]))
    spectralFeatures['mfcc16_avg'] = float(np.mean(mfcc[15]))
    spectralFeatures['mfcc17_avg'] = float(np.mean(mfcc[16]))
    spectralFeatures['mfcc18_avg'] = float(np.mean(mfcc[17]))
    spectralFeatures['mfcc19_avg'] = float(np.mean(mfcc[18]))
    spectralFeatures['mfcc20_avg'] = float(np.mean(mfcc[19]))
    spectralFeatures['mfcc21_avg'] = float(np.mean(mfcc[20]))
    spectralFeatures['mfcc22_avg'] = float(np.mean(mfcc[21]))
    spectralFeatures['mfcc23_avg'] = float(np.mean(mfcc[22]))
    spectralFeatures['mfcc24_avg'] = float(np.mean(mfcc[23]))
    spectralFeatures['mfcc25_avg'] = float(np.mean(mfcc[24]))
    spectralFeatures['mfcc26_avg'] = float(np.mean(mfcc[25]))
    spectralFeatures['mfcc27_avg'] = float(np.mean(mfcc[26]))
    spectralFeatures['mfcc28_avg'] = float(np.mean(mfcc[27]))
    spectralFeatures['mfcc29_avg'] = float(np.mean(mfcc[28]))
    spectralFeatures['mfcc30_avg'] = float(np.mean(mfcc[29]))
    spectralFeatures['chroma_stft_std'] = float(np.std(chroma_stft))
    spectralFeatures['rmse_std'] = float(np.std(rmse))
    spectralFeatures['spec_cent_std'] = float(np.std(spec_cent))
    spectralFeatures['spec_bw_std'] = float(np.std(spec_bw))
    spectralFeatures['rolloff_std'] = float(np.std(rolloff))
    spectralFeatures['zcr_std'] = float(np.std(zcr))
    spectralFeatures['mfcc1_std'] = float(np.std(mfcc[0]))
    spectralFeatures['mfcc2_std'] = float(np.std(mfcc[1]))
    spectralFeatures['mfcc3_std'] = float(np.std(mfcc[2]))
    spectralFeatures['mfcc4_std'] = float(np.std(mfcc[3]))
    spectralFeatures['mfcc5_std'] = float(np.std(mfcc[4]))
    spectralFeatures['mfcc6_std'] = float(np.std(mfcc[5]))
    spectralFeatures['mfcc7_std'] = float(np.std(mfcc[6]))
    spectralFeatures['mfcc8_std'] = float(np.std(mfcc[7]))
    spectralFeatures['mfcc9_std'] = float(np.std(mfcc[8]))
    spectralFeatures['mfcc10_std'] = float(np.std(mfcc[9]))
    spectralFeatures['mfcc11_std'] = float(np.std(mfcc[10]))
    spectralFeatures['mfcc12_std'] = float(np.std(mfcc[11]))
    spectralFeatures['mfcc13_std'] = float(np.std(mfcc[12]))
    spectralFeatures['mfcc14_std'] = float(np.std(mfcc[13]))
    spectralFeatures['mfcc15_std'] = float(np.std(mfcc[14]))
    spectralFeatures['mfcc16_std'] = float(np.std(mfcc[15]))
    spectralFeatures['mfcc17_std'] = float(np.std(mfcc[16]))
    spectralFeatures['mfcc18_std'] = float(np.std(mfcc[17]))
    spectralFeatures['mfcc19_std'] = float(np.std(mfcc[18]))
    spectralFeatures['mfcc20_std'] = float(np.std(mfcc[19]))
    spectralFeatures['mfcc21_std'] = float(np.std(mfcc[20]))
    spectralFeatures['mfcc22_std'] = float(np.std(mfcc[21]))
    spectralFeatures['mfcc23_std'] = float(np.std(mfcc[22]))
    spectralFeatures['mfcc24_std'] = float(np.std(mfcc[23]))
    spectralFeatures['mfcc25_std'] = float(np.std(mfcc[24]))
    spectralFeatures['mfcc26_std'] = float(np.std(mfcc[25]))
    spectralFeatures['mfcc27_std'] = float(np.std(mfcc[26]))
    spectralFeatures['mfcc28_std'] = float(np.std(mfcc[27]))
    spectralFeatures['mfcc29_std'] = float(np.std(mfcc[28]))
    spectralFeatures['mfcc30_std'] = float(np.std(mfcc[29]))
    return spectralFeatures

def getAudioDuration(fname):
    return librosa.get_duration(filename=fname)
### SCRIPT EXECUTION SECTION ###
