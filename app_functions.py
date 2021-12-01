# import libaries
import nltk
import spacy
import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pandas as pd


# global variables

# get unique English stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords = list(stopwords)
stopwords.extend(['the', 'eg.', 'eg'])
stopwords = set(stopwords)

# initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# regular expression pattern for removing non-word character
regex = re.compile(r'[^\w\s]+')

def lemmatize_text(text):
    # first split text into sentences based on the period
    sentences = text.split('. ')
    lemmas = []   # a list of lemmatized sentences
    
    for sent in sentences:
        # Parse the text using the loaded 'en' model object `nlp`
        doc = nlp(sent)
        
        # Extract the lemma for each token and join
        lemmas.append(" ".join([token.lemma_ for token in doc]))
    
    # return the lemmatized text
    return lemmas

def clean_text(text):
    '''Remove none-word characters'''
    
    global regex
    
    # remove any extra spaces
    text_temp = re.sub(r'\s{2,}', ' ', text, flags=re.MULTILINE)
    
    # Remove Emails
    text_temp = re.sub('\S*@\S*\s?', '', text_temp)

    # Remove new line characters
    text_temp = re.sub('\s+', ' ', text_temp)

    # Remove distracting single quotes
    text_temp = re.sub("\'", "", text_temp)
    
    # remove any none-word characters
    sentences = []
    for sent in text_temp.split('. '):
        sentences.append(' '.join([regex.sub('', word) for word in sent.split(' ')]))
    
    return '. '.join(sentences)

def process_text(text):
    '''Perform text cleaning and processing'''
    
    # global variables
    global stopwords
    
    # convert text to lowercase and perform lemmatization
    lemmatized_text = lemmatize_text(text.lower())
    
    # remove stopwords
    tokens = [word for word in lemmatized_text if not word in stopwords]
    
    # remove any non-words character
    processed_text = clean_text(' '.join(tokens))
    
    return processed_text


'''******** Functions for Topic Extraction Model ********'''

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(tests, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def remove_stopwords_for_lda(texts):
    # global variables
    global stopwords
    
    stop_words = list(stopwords)
    stop_words.extend(['of', 'be', 'and', 'in', 'to', 'also'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def extract_topics(text):
    # split text into sentences
    sentences = text.split('. ')
    
    # convert sentences to words
    data_words = list(sent_to_words(sentences))
    data_words = remove_stopwords_for_lda(data_words)
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentecne clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    data_words_bigrams = make_bigrams(data_words, bigram_mod)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words_bigrams)

    # Create Corpus
    texts = data_words_bigrams

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # run LDA model 5 times on different k-value 
    models = []
    coherences = []
    perplexities = []
    k_values = [x for x in range(1, 6)]
    for k in k_values:
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus = corpus,
            id2word = id2word,
            num_topics = k,
            random_state = 100,
            update_every=1,
            chunksize=100,
            passes = 10,
            alpha = 'auto',
            per_word_topics=True)

        # Compute Perplexity
        # a measure of how good the model is; the lower the better.
        # since log return negative value, for values < 0, the higher the better, e.g. -6 is preferred over -7
        perplexity = lda_model.log_perplexity(corpus)

        # Compute Coherence Score: the higher the better
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        
        # save model and scores
        models.append(lda_model)
        coherences.append(coherence_lda)
        perplexities.append(perplexity)
    
    # set the best model with highest coherence score
    best_model = models[coherences.index(max(coherences))]
    
    # get terms for extracted topic
    topic_terms = []
    for topic in best_model.show_topics(formatted=False):
        # get terms for each topic
        terms = [t[0] for t in topic[1]]
        topic_terms.append(', '.join(terms))
    
    # build a dataframe of extracted topics
    topics = pd.DataFrame(dict({'Topic Id':[x for x in range(1,len(topic_terms)+1)],
                               'Terms': topic_terms}))
    
    # build a dataframe of scores
    scores = pd.DataFrame(dict({'Number of Topics': k_values, 'Coherence': coherences, 'Perplexity': perplexities}))
    
    # return topic dataframe, perplexity and coherence scores
    return topics, scores