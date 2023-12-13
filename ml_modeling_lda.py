import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import spacy
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import numpy as np
from joblib import dump
import pyLDAvis
import pyLDAvis.gensim
from src.utils import insertSQLPandas, selectSQLPandas
from src.config import Config
config = Config()

def main():
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner']) #could also use en_core_web_sm

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(data):
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data, min_count=5, threshold=10) # higher threshold fewer phrases.
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        data_words_nostops = remove_stopwords(data)
        return [bigram_mod[doc] for doc in data_words_nostops]

    def make_trigrams(data):
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data, min_count=5, threshold=10) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data], threshold=10)  
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        data_words_nostops = remove_stopwords(data)
        return [trigram_mod[bigram_mod[doc]] for doc in data_words_nostops]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def get_lda_model(corpus, dictionary, k, a, b, multicore=True):
        if multicore:
            lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                                    id2word=dictionary,
                                                    num_topics=k, 
                                                    random_state=100,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha=a,
                                                    eta=b)
        else:
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=dictionary,
                                                    num_topics=k, 
                                                    random_state=100,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha=a,
                                                    eta=b)
        return lda_model
    
    def get_per_topic_coherence(cm,lda_model,num_words_in_topic=10):
        topics = lda_model.show_topics(num_words=num_words_in_topic)
        coherence_per_topic = cm.get_coherence_per_topic()
        topics_str = ['\n '.join(t) for t in topics]
        data_topic_score = pd.DataFrame( data=zip(topics_str, coherence_per_topic), columns=['Topic', 'Coherence'] )
        return data_topic_score

    baseDir = config.base_directory
    
    '''
    !!!NOTE(s)!!!
    You must define the text column and the sql variables below
    
    NOTE: By default the way this is written it will use multicore
    if you do not want that to happen, change th useMulticore variable
    below to False!
    
    NOTE: The output of this process places everything into two
    subdirectories in your baseDir output folder, you need to
    creat these!!!:
    models
    visuals
    '''
    useMulticore = True
    textcol = '' #define your text column!
    sql = '' #define your sql statement to get the data you want!!!
    
    df = selectSQLPandas(sql)
    data = df[textcol].values.tolist()
    data_words = list(sent_to_words(data))
    print("Data loaded!")

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words)

    # df['abstract_cleaned'] = df.apply(lambda row: (re.sub("[^A-Za-z0-9' ]+", ' ', row['abstract'])),axis=1)
    # df['abstract_cleaned'] = df.apply(lambda row: (word_tokenize(row['abstract_cleaned'])), axis = 1)


    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    print("Corpus is ready!")
    # Topics range
    min_topics = 2
    max_topics = 20
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.1))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    alpha.append('auto') #cannot be used with lda multicore

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.1))
    beta.append('symmetric')
    beta.append('auto')

    # Validation sets
    num_of_docs = len(corpus)
    print(f"Number of docs is {num_of_docs}")
    corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
                    # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
                    #gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75), 
                    corpus]

    corpus_title = ['100% Corpus']

    model_results = {'id': [],
                        'Topics': [],
                        'Alpha': [],
                        'Beta': [],
                        'c_v': [],
                        'c_uci': [],
                        'c_npmi': [],
                        'u_mass': []
                    }

    # Can take a long time to run
    # iterate through validation corpuses
    count = 1
    print("Modeling beginning!")
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterate through beta values
                for b in beta:
                    model_results = {'id': [],
                        'Topics': [],
                        'Alpha': [],
                        'Beta': [],
                        'c_v': [],
                        'c_uci': [],
                        'c_npmi': [],
                        'u_mass': []
                    }
                    # get the coherence score for the given parameters
                    if a == 'auto':
                        multicore=False
                    else:
                        multicore=useMulticore
                    lda_model = get_lda_model(corpus=corpus_sets[i], dictionary=id2word, k=k, a=a, b=b, multicore=multicore)
                    coherence_c_v = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
                    coherence_c_uci = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_uci')
                    coherence_c_npmi = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_npmi')
                    coherence_u_mass = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='u_mass')
                    c_v = coherence_c_v.get_coherence()
                    c_uci = coherence_c_uci.get_coherence()
                    c_npmi = coherence_c_npmi.get_coherence()
                    u_mass = coherence_u_mass.get_coherence()
                    
                    # Save the model results
                    model_results['id'].append(count)
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['c_v'].append(c_v)
                    model_results['c_uci'].append(c_uci)
                    model_results['c_npmi'].append(c_npmi)
                    model_results['u_mass'].append(u_mass)
                    metric_df = pd.DataFrame.from_dict(model_results)
                    insertSQLPandas(metric_df,'topicModel_metrics')
                    dump(lda_model,f"{baseDir}/output/models/lda_model_{count}.joblib")
                    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
                    dump(LDAvis_prepared,f"{baseDir}/output/visuals/pyLDAvis_{count}.joblib")
                    pyLDAvis.save_html(LDAvis_prepared,f"{baseDir}/output/visuals/pyLDAvis_{count}.html")
                    print(f"{count}\t{c_v}\t{c_uci}\t{c_npmi}\t{u_mass}")
                    count+=1

if __name__ == '__main__':
    main()