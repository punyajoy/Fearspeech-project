import re
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') 

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from nltk.tokenize import word_tokenize,sent_tokenize
lemmatizer = WordNetLemmatizer()  

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    #corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons])


### Some text preprocessing functions

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
REPLACE_IP_ADDRESS = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
URL = re.compile(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')

def preprocess_func(text):
        remove_words=['<allcaps>','</allcaps>','<hashtag>','</hashtag>','<elongated>','<emphasis>','<repeated>','\'','s']
        word_list=text_processor.pre_process_doc(text)
        word_list=list(filter(lambda a: a not in remove_words, word_list)) 
        sent=" ".join(word_list)
        sent = re.sub(r"[<\*>]", " ",sent)
        return sent

def text_lemmatize(text):
    token_words=text.split()
    lemma_sentence=[]
    for word in token_words:
        lemma_sentence.append(lemmatizer.lemmatize(word))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)

def text_prepare(text):
    text = text.replace('\n', ' ').lower() #lowercase text
    text = URL.sub('', text)
    text = REPLACE_IP_ADDRESS.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub(' ',text) #replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ',text) #give space symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([w for w in text.split() if not w in STOPWORDS])# delete stopwords from text
    return text


### Bag of Words (BoW)
from scipy import sparse as sp_sparse
def my_bag_of_words(text, words_to_index, dict_size):
    result_vector = np.zeros(dict_size)
    for word in text.split(' '):
        if word in words_to_index:
            result_vector[words_to_index[word]] +=1
    return result_vector

def bag_of_words(x_train, x_val, x_test):
    words_counts = {}
    for comments in x_train:
        for word in comments.split():
            if word not in words_counts:
                words_counts[word] = 0
            words_counts[word] += 1

    DICT_SIZE = len(words_counts)
    POPULAR_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[:DICT_SIZE]
    WORDS_TO_INDEX = {key: rank for rank, key in enumerate(POPULAR_WORDS, 0)}
    INDEX_TO_WORDS = {index:word for word, index in WORDS_TO_INDEX.items()}
    ALL_WORDS = WORDS_TO_INDEX.keys()
            
    x_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in x_train])
    x_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in x_val])
    x_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in x_test])
    return x_train_mybag, x_val_mybag, x_test_mybag


### Tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_features(x_train, x_val, x_test):
    """
        X_train, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test set and return the result
    
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5, token_pattern='(\S+)')
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_val_tfidf = tfidf_vectorizer.transform(x_val)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    
    return x_train_tfidf, x_val_tfidf, x_test_tfidf, tfidf_vectorizer.vocabulary_


### Glove Word Embeddings

'''
Uncomment if you need to load a new embedding, Glove is already loaded as pickle file
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

glove_model = loadGloveModel("/home/punyajoy/HULK/Fearspeech_project/embeddings/glove.840B.300d.txt")
'''

import pickle
glove_embeddings = pickle.load(open('/home/punyajoy/HULK/Fearspeech_project/glove.pkl', "rb"))
def post_features(post, embeddings = glove_embeddings, emb_size=300):
    words = post.split()
    words=[w for w in words if w.isalpha() and w in embeddings]
    if len(words)==0:
        return np.hstack([np.zeros(emb_size)])
    M=np.array([embeddings[w] for w in words])
    return M.mean(axis=0)

def glove_features(x_train, x_val, x_test):
    x_train_glove = np.array([post_features(x) for x in x_train])
    x_val_glove = np.array([post_features(x) for x in x_val])
    x_test_glove = np.array([post_features(x) for x in x_test])
    return x_train_glove, x_val_glove, x_test_glove