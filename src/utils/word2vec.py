from gensim.models import KeyedVectors
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from src.config import Config
config = Config()

# def loadModel(modelPath=None):
#     if modelPath is None:
#         modelPath = f"{config.base_directory}/data/google_en_w2v.bin"
#     # Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
#     model = KeyedVectors.load_word2vec_format(modelPath, binary=True)
#     return model

def similarWords(incoming_words,topn=100,positive=True,ignore_out_of_vocabulary=False):
    model = config.w2v_model
    if not isinstance(incoming_words,list):
        incoming_words = list(incoming_words)
    wordstouse = []
    for word in incoming_words:
        if word in model:
            wordstouse.append(word)
    if len(wordstouse) != len(incoming_words) and not ignore_out_of_vocabulary:
        return [] #could probably come up with a better way to do this
    if positive:
        words = model.most_similar(positive=wordstouse, topn=topn)
    else:
        words = model.most_similar(negative=wordstouse, topn=topn)
    return words

def wordEquation(positive_words,negative_words,topn=100,ignore_out_of_vocabulary=False):
    model = config.w2v_model
    if not isinstance(positive_words,list):
        positive_words = list(positive_words)
    pos_wordstouse = []
    for word in positive_words:
        if word in model:
            pos_wordstouse.append(word)
    if not isinstance(negative_words,list):
        negative_words = list(negative_words)
    neg_wordstouse = []
    for word in negative_words:
        if word in model:
            neg_wordstouse.append(word)
    if (len(neg_wordstouse) != len(negative_words) or len(pos_wordstouse) != len(positive_words)) and not ignore_out_of_vocabulary:
        return [] #could probably come up with a better way to do this
    words = model.most_similar(positive=pos_wordstouse, negative=neg_wordstouse, topn=topn)
    return words

def wordDistance(word1,word2):
    model = config.w2v_model
    for word in (word1,word2):
        if word not in model:
            return None
    word_similarity = model.similarity(word1,word2)
    return word_similarity

def leastSimilar(words):
    model = config.w2v_model
    if not isinstance(words,list):
        return None
    for word in words:
        if word not in model:
            return None
    odd_one_out = model.doesnt_match(words)
    return odd_one_out

def clean_text(text):
    result = []
    badChars = '~!@#$%^&*()_+=`[]\\;,./{}|:"<>?'
    # text = text.lower() #after testing the w2v, casing does appear to matter, at least for proper nouns
    text = text.replace('\n',' ')
    while '  ' in text:
        text = text.replace('  ',' ')
    for c in badChars:
        text = text.replace(c,'')
    cleaned_text = text.split(' ')
    for word in cleaned_text:
        if word not in stop_words:
            result.append(word)
    return result

def compareTexts(text1,text2):
    texts = [text1,text2]
    cleaned_texts = []
    distances = []
    for text in texts:
        cleaned_texts.append(clean_text(text))
    for word1 in cleaned_texts[0]:
        for word2 in cleaned_texts[1]:
            wd = wordDistance(word1,word2)
            if wd is not None:
                distances.append(wd)
    return sum(distances)/len(distances)
            