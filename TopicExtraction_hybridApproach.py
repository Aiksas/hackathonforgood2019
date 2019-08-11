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

# google news model
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# good model, but takes a long time to load as it contains very large set of embeddings

# read in labels and their synsets
labels = pd.read_csv('Dic_4.csv', sep=";")
labels['synsets'] = labels['synsets'].str.replace('[^\w\s]','')


# to do: rewrite and loop over different categories, now still hard-coded
# as this was written when I was tired at the end of long day in Hackaton...
def check_identical_words(topics):
    """ Function that looks for similar words within the topics and the labels and 
    their synsets """
    categories = []
    for word in topics:
        if labels.loc[labels['label'] == 'Nutrition', 'synsets'].str.contains(word).any():
            category = 'Nutrition'
            categories.append(category)
        elif labels.loc[labels['label'] == 'Protection', 'synsets'].str.contains(word).any():
            category = 'Protection'
            categories.append(category)
        elif labels.loc[labels['label'] == 'Logistics', 'synsets'].str.contains(word).any():
            category = 'Logistics'
            categories.append(category)
        elif labels.loc[labels['label'] == 'Health', 'synsets'].str.contains(word).any():
            category = 'Health'
            categories.append(category)
        elif labels.loc[labels['label'] == 'Shelter', 'synsets'].str.contains(word).any():
            category = 'Shelter'
            categories.append(category)
        elif labels.loc[labels['label'] == 'Water/Sanitation/Hygiene', 'synsets'].str.contains(word).any():
            category = 'Water/Sanitation/Hygiene'
            categories.append(category)  
        elif labels.loc[labels['label'] == 'Camp Coordination and Camp Management', 'synsets'].str.contains(word).any():
            category = 'Camp Coordination and Camp Management'
            categories.append(category)   
        elif labels.loc[labels['label'] == 'Education', 'synsets'].str.contains(word).any():
            category = 'Education'
            categories.append(category)  
        elif labels.loc[labels['label'] == 'Emergency Telecommunications', 'synsets'].str.contains(word).any():
            category = 'Emergency Telecommunications'
            categories.append(category) 
        elif labels.loc[labels['label'] == 'Food security', 'synsets'].str.contains(word).any():
            category = 'Food security'
            categories.append(category) 
        else:
            category = 'unknown'
            categories.append(category)
    #categories = list(set(categories))
    return(categories)

categories = check_identical_words(topics)
most_common_label = [word for word, word_count in Counter(categories).most_common(1)]

# to do: rewrite and loop over different categories, now still hard-coded
# as this was written when I was tired at the end of long day in Hackaton...
def get_similar_words(df):
    """ Function that determines summed similarity of first 4 words in topic
    per label, and then assigns label with highest score """
    
    Nutrition_score = (word_vectors.similarity('nutrition', topics[0]) + 
                       word_vectors.similarity('nutrition', topics[1]) + 
                       word_vectors.similarity('nutrition', topics[2]) + 
                       word_vectors.similarity('nutrition', topics[3]))
    
    Protection_score = (word_vectors.similarity('protection', topics[0]) + 
                       word_vectors.similarity('protection', topics[1]) + 
                       word_vectors.similarity('protection', topics[2]) + 
                       word_vectors.similarity('protection', topics[3]))
    
    Logistics_score = (word_vectors.similarity('logistics', topics[0]) + 
                       word_vectors.similarity('logistics', topics[1]) + 
                       word_vectors.similarity('logistics', topics[2]) + 
                       word_vectors.similarity('logistics', topics[3]))
    
    Health_score = (word_vectors.similarity('health', topics[0]) + 
                       word_vectors.similarity('health', topics[1]) + 
                       word_vectors.similarity('health', topics[2]) + 
                       word_vectors.similarity('health', topics[3]))
    
    Shelter_score = (word_vectors.similarity('shelter', topics[0]) + 
                       word_vectors.similarity('shelter', topics[1]) + 
                       word_vectors.similarity('shelter', topics[2]) + 
                       word_vectors.similarity('shelter', topics[3]))
    
    Hygiene_score = (word_vectors.similarity('hygiene', topics[0]) + 
                       word_vectors.similarity('hygiene', topics[1]) + 
                       word_vectors.similarity('hygiene', topics[2]) + 
                       word_vectors.similarity('hygiene', topics[3]))
    
    Camp_score = (word_vectors.similarity('camp', topics[0]) + 
                       word_vectors.similarity('camp', topics[1]) + 
                       word_vectors.similarity('camp', topics[2]) + 
                       word_vectors.similarity('camp', topics[3]))
    
    Food_score = (word_vectors.similarity('food', topics[0]) + 
                       word_vectors.similarity('food', topics[1]) + 
                       word_vectors.similarity('food', topics[2]) + 
                       word_vectors.similarity('food', topics[3]))

    Education_score = (word_vectors.similarity('education', topics[0]) + 
                       word_vectors.similarity('education', topics[1]) + 
                       word_vectors.similarity('education', topics[2]) + 
                       word_vectors.similarity('education', topics[3]))
    
    Telecommunication_score = (word_vectors.similarity('telecommunication', topics[0]) + 
                       word_vectors.similarity('telecommunication', topics[1]) + 
                       word_vectors.similarity('telecommunication', topics[2]) + 
                       word_vectors.similarity('telecommunication', topics[3]))
    
    scores = {'Nutrition':Nutrition_score, 'Protection':Protection_score, 'Logistics':Logistics_score, 
              'Health':Health_score, 'Shelter':Shelter_score, 'Hygiene':Hygiene_score, 
              'Camp':Camp_score, 'Food':Food_score, 'Education':Education_score, 
              'Telecommunication':Telecommunication_score} 
    
    label_text = max(scores, key=scores.get)
    
    return(label_text)


# Assign the category of the text to the variable 'label_text'
if most_common_label != ['unknown']:
    label_text = most_common_label
else:
    label_text = get_similar_words(topics)
    
    
    
# print the category of the text
print(label_text)



    