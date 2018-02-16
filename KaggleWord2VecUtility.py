#!/usr/bin/env python

import re
import nltk

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words("english")

def stemm(stemmer,w):
    try:
        temp = stemmer.stem(w)
    except:
        temp = w
    return temp


def lemm(lemm,w):
    try:
        temp = lemm.lemmatize(w)
    except:
        temp = w
    return temp

class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=True, stem = False, lem= False ):

        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review,"html.parser").get_text()


        #
        # 2. Remove caractéres não alfa-numéricos
        review_text = re.sub("[^a-zA-Z]"," ", review_text)


        #
        # 3. Tokeniza e/ou utiliza técnicas de pre-processamento
        if(stem):
            words = [stemm(stemmer, w) for w in tokenizer.tokenize(review_text) if w not in stopwords]
        elif(lem):
            words = [lemmatizer(lemm, w) for w in tokenizer.tokenize(review_text) if w not in stopwords]
        elif(stopwords):
            words = [ w for w in tokenizer.tokenize(review_text) if w not in stopwords]
        else:
            words = tokenizer.tokenize(review_text)
        #
        # 4. Retorna a lista de palavras
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=True ):
        #
        # 1. Usa o tokenizador passado para tokenizar a sentença
        raw_sentences = tokenizer.tokenize(review.strip())
        #

        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                # 2. Chama review_to_wordlist para ter a lista de palavras
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                  remove_stopwords, stem=False ))
        #
        # Retorna a lista de sentenças, onde cada sentença é uma lista de palavras
        return sentences
