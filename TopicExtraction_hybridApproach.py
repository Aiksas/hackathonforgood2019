# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 09:56:30 2019

@author: saskia.lensink
"""

#####################################################
##  Topic classifier
#####################################################


# Goal of this script: extracting the topic of a text that needs translating.
# We have only limited time, and unlabeled data. Although topic extraction is
# an often-used technique in NLP, its output consists of a jumble of words 
# that describe the different topic present in a document or set of documents 
# --- an output that still needs interpretation by a human an cannot be 
# implemented easily in an application that automatically assigns topics.

# We do already have some idea on which topics could be relevant, but we do
# not have time to annotate documents manually, to be able to use a supervised 
# topic labeling technique such as Naive Bayes to categorize new documents.
# Therefore, we came up with a hybrid approach.

# We predefined a set of categories that are considered relevant by TWB
# (e.g. 'Hygiene', 'Shelter', etc), and enriched these categories with 
# synonyms, hyponyms taken from Wordnet (https://wordnet.princeton.edu/).
# We then used Non-Negative Matrix Factorization to extract topics from the
# document, which gives us an output of several words related to this latent
# topics. 

# After extracting these words relating to the latent categories, we ran 
# a simple comparison between these words and the enriched 'dictionary' of 
# predefined categories. If there is an exact match, we know which category to 
# assign to the document. For example, if the set of words related to the 
# latent categories are <'health', 'bandage', 'hospital'>, we have an exact
# match with the category label 'health', and therefore we assign this text
# to the category'Health'. Note that if there are multiple matches with 
# different categories, we go for the majory vote. 

# When there are no exact matches between the output of the topic extraction
# and the predefined labels, we made use of a summed similarity score between 
# the words in the output of the topic extraction algorithm, and the words
# in the enriched dictionary of predefined labels and their related terms. 
# This similarity score is based on the cosine distance words take up in a 
# multidimensional vector space. For more information, see e.g. 
# https://radimrehurek.com/gensim/tut3.html. 
# 

# Saskia E. Lensink, saskia.lensink@cgi.com
# Team Lazy Pandas






# load packages
import pandas as pd
import os
import nltk
import numpy as np
# import re
from collections import Counter
# from gensim.models import Word2Vec
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer

# for future improvement: spaCy  -- somehow pip failed at installing it



# change to your own wd
os.chdir("C:/Users/saskia.lensink/OneDrive/HackatonTWB")
# load pre-trained word embeddings
word_vectors = api.load("glove-wiki-gigaword-100")  


# # 1. Load data

text = pd.read_csv("docs/201403_women_and_migration_SKR.docx.sdlxliff.csv", sep=";")
## Result = Education
# text = pd.read_csv("docs/APRIL_2019_-_send_out_for_translation.docx_sdlxliff.csv", sep=";")
## result = Education
# text = pd.read_csv("docs/Disability_Inclusion_and_Etiquette_Training_presentation_text.docx.sdlxliff.csv", ";")
## result = health
# text = pd.read_csv("docs/(GDPC)_First_Aid__Vietnamese_vLookup_Results_for_tra1.csv", sep=";")
## result = food

###############################################
##### TEXT PREPROCESSING  #####################
###############################################


# Convert to lower case 
# text = text.lower()
text['source'] = text['source'].str.lower()

# Convert numbers into strings
# text = re.sub(r'\d+', '', text)
text['source'] = text['source'].str.replace(r'\d+', '')
# print(text)

# remove '\n' if present
# text = re.sub('\\n', ' ', text)
text['source'] = text['source'].str.replace(r'\d+', '')

# remove () if present
text['source'] = text['source'].str.replace(r'\([^)]*\)', '')
text['source'] = text['source'].str.replace(r'\(', '')
# text = re.sub(r'\([^)]*\)', '', text)
# text = re.sub(r'\(', '', text)



## Tokenizing

# tokenize into sentences and put sentences into df
def tokenize_sentences(df, column):
    df['tokenized'] = df[column].apply(sent_tokenize)
    # df = sent_tokenize(df)
    # df = pd.DataFrame(df)
    df.columns = ['number', 'sentence', 'tokenized']
    return(df)
    
text = tokenize_sentences(text, 'source')


def tokenize(df, column):
    """ Split text into single words """
    df['tokenized'] = df[column].apply(word_tokenize)
    return(df)

text = tokenize(text, 'sentence')
#print(text.head())


# remove punctuation 
text['tokenized'].str.replace('[^\w\s]','')
# print(text.tokenized.head())


## POS tagging and lemmatizing ##
def get_pos(word):
    """ Map POS tag to first character lemmatize() accepts """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize with the appropriate POS tag
text['lemmatized'] = text['tokenized'].apply(lambda x: [lemmatizer.lemmatize(word, get_pos(word)) for word in x])
# text_lemmatized = [lemmatizer.lemmatize(word, get_pos(word)) for word in text]
#print(text.lemmatized.head())

# remove stop words
stop_words = set(stopwords.words('english'))

def remove_stop_words(df, column):
    """ function to remove English stop words  """
    # df = [word for word in text_lemmatized if word not in stop_words]
    df[column] = df[column].apply(lambda x: [item for item in x if item not in stop_words])
    return(df)
    
text_lemmatized = remove_stop_words(text, 'lemmatized')
# print(text.lemmatized.head())


text['lemmatized'] = text['lemmatized'].str.join(' ')
# print(text.lemmatized.head())
# text_lemmatized = ' '.join(text_lemmatized)
# text_lemmatized = [text_lemmatized]

## Topic extraction using Non-Negative Matrix Factorization ##
cv = CountVectorizer(stop_words="english", max_features=1000, ngram_range=(1,3), min_df=2)
nmf = NMF(n_components=5) 

transformed = cv.fit_transform(text['lemmatized'])
nmf.fit(transformed)

# Get topics out 
topics = []
for topic_idx, topic in enumerate(nmf.components_):
    label = '{}: '.format(topic_idx)
    content = label + " ".join([cv.get_feature_names()[i] + ','
    for i in topic.argsort()[:-6:-1]])
    topics.append(content)
    
# transform topics into pandas dataframe for easier look-up
topics = np.asarray(topics)
topics = pd.DataFrame(topics)
topics.columns = ['topics']

# cleaning text 
topics['topics'] = topics['topics'].str.replace(r'\d+', '')
topics['topics'] = topics['topics'].str.replace(',', '')
topics['topics'] = topics['topics'].str.replace(':', '')

# putting into array with single words
topics = tokenize(topics, 'topics')
topics = np.concatenate(topics['tokenized'])

# remove doubles
topics = list(set(topics))



###### Similarities
 
# See if the topic words of the documents are mentioned in the labels
# There's two ways of doing this:
# 1) checking for an exact match
# 2) checking for a semantically similar match    

def sum_similarity_scores(label, topics, model): #topics=topics, model=model):
    """ Function that loops through all extracted topics, gets the
    similarity score of that topic with a label, sums all similarity
    scores and then outputs the total similarity score """    
    test_score = []
    for t in topics:
        if t in model.vocab:
            if label in model.vocab:
                score = model.similarity(label, t)
            else:
                score = 0
            test_score.append(score)
    return(np.sum(test_score)) 


def average_similarity_scores(label, topics, model): #topics=topics, model=model):
    """ Function that loops through all extracted topics, gets the
    similarity score of that topic with a label, averages all similarity
    scores and then outputs the average similarity score """    
    test_score = []
    for t in topics:
        if t in model.vocab:
            if label in model.vocab:
                score = model.similarity(label, t)
            else:
                score = 0
            test_score.append(score)
    return(np.mean(test_score)) 



# read in labels
labels = pd.read_csv('Dic_4.csv', sep=";")
relevant_labels = list(labels.label)
relevant_labels = [label.lower() for label in relevant_labels]
scores = dict.fromkeys(relevant_labels, 0)

# get max score
for l in relevant_labels:
    score = sum_similarity_scores(l, topics, word_vectors)
    scores[l] = score
    
#  check all scores
print(scores)

# get most likely label
label_text = max(scores, key=scores.get)
print(label_text)


    

    