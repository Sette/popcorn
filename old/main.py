

import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import nltk
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from KaggleWord2VecUtility import KaggleWord2VecUtility
from nltk.metrics import ConfusionMatrix



import pandas as pd
import numpy as np
import os
from nltk.corpus import stopwords
import nltk.data
import nltk
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from KaggleWord2VecUtility import KaggleWord2VecUtility
from nltk.metrics import ConfusionMatrix
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import *
from nltk.tokenize import *
from sklearn import svm

#Define o tokenizador a ser utilizado
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
tokenFunc = tokenizer.tokenize
stopwords = nltk.corpus.stopwords.words("english")
vectorizer = TfidfVectorizer(tokenizer= tokenFunc, ngram_range=(1,3), min_df = 0, stop_words=stopwords,)
transformer = TfidfTransformer()


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print ("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model


def main():
    # Leitura de arquivos e armazenamento em um DF

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData_min.tsv'), header=0,
                     delimiter="\t", quoting=3)
    '''
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    '''
    #Opcional ---- Cross over com o conjunto de treinamento
    train,test = train_test_split(train, test_size=0.33, random_state=42)

    (X_train, Y_train) = train.review, train.sentiment

    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,
                                  delimiter="\t", quoting=3)

    # Verifica o numero de registros carregados
    print("Lidos %d registros de treinamento com label, %d registros de teste, %s registros de treinamento sem Label " \
          % (train["review"].size,
             test["review"].size,unlabeled_train["review"].size))

    # ******Cria lista de sentenças encontradas na base de treinamento com label e tambem a sem label
    # ****** Limpa as sentenças
    sentences = []  # Initialize an empty list of sentences

    # Pré-processamento

    print("Criando lista de sentenças do conjunto de treinamento com label")
    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    ''''
    print("Criando lista de sentenças do conjunto de treinamento sem label")
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
    '''

    
    # Configura valores para o word2vec

    num_features = 300  # Word vector dimensionality
    '''
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    
    # Initialize and train the model (this will take some time)
    print("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    
    '''
    model = loadGloveModel("glove.6B.300d.txt")


    # ****** Criando vetor de palavras para treinamento
    #
    print("Criando vetor de palavras para treinamento")

    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)

    print("Criando vetor de palavras para teste")

    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)



    #Classificação com TFIDF e SVM

    '''
    
    classType = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)

    Classifier_SVM = Pipeline([('vect', vectorizer),('transf', transformer),('clf',  classType)])

    Classifier_SVM.fit(X_train, Y_train)

    predicted = Classifier_SVM.predict(test["review"])
    
    
    print("Accuracy: ", accuracy_score(list(test['sentiment']), predicted))
    print("F1: ", f1_score(list(test['sentiment']), predicted))
    print("Precision: ", precision_score(list(test['sentiment']), predicted))
    matriz = ConfusionMatrix(test['sentiment'], predicted)
    
    '''
    # ****** Fit a random forest to the training set, then make predictions
    #
    # Definindo Random Forest para Classificação com 100 arvores


    forest = RandomForestClassifier(n_estimators=100)

    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # Testa modelo com conjunto de teste
    result = forest.predict(testDataVecs)

    # Escrevendo resultados obtidos
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("saida.csv", index=False, quoting=3)



    #Fazendo cálculos de analise de resultados
    print("Accuracy: ", accuracy_score(list(test['sentiment']), result))
    print("F1: ", f1_score(list(test['sentiment']), result))
    print("Precision: ", precision_score(list(test['sentiment']), result))
    matriz = ConfusionMatrix(test['sentiment'], result)

    print(matriz)
    
    

if __name__ == '__main__':
    main()

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print ("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



def main():
    # Leitura de arquivos e armazenamento em um DF

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                     delimiter="\t", quoting=3)
    '''
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    '''
    #Opcional ---- Cross over com o conjunto de treinamento
    train,test = train_test_split(train, train_size=0.7, random_state=42)

    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,
                                  delimiter="\t", quoting=3)

    # Verifica o numero de registros carregados
    print("Lidos %d registros de treinamento com label, %d registros de teste, %d registros de treinamento sem Label " \
          % (train["review"].size,
             test["review"].size, unlabeled_train["review"].size))

    # ******Cria lista de sentenças encontradas na base de treinamento com label e tambem a sem label
    # ****** Limpa as sentenças
    sentences = []  # Initialize an empty list of sentences

    print("Criando lista de sentenças do conjunto de treinamento com label")
    for review in train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print("Criando lista de sentenças do conjunto de treinamento sem label")
    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)


    # Configura valores para o word2vec
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)

    # ****** Criando vetor de palavras para treinamento
    #
    print("Criando vetor de palavras para treinamento")

    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)

    print("Criando vetor de palavras para teste")

    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)

    # ****** Fit a random forest to the training set, then make predictions
    #
    # Definindo Random Forest para Classificação com 100 arvores
    forest = RandomForestClassifier(n_estimators=100)

    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # Testa modelo com conjunto de teste
    result = forest.predict(testDataVecs)

    # Escrevendo resultados obtidos
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("saida.csv", index=False, quoting=3)

    #Fazendo cálculos de analise de resultados
    print("Accuracy: ", accuracy_score(list(test['sentiment']), result))
    print("F1: ", f1_score(list(test['sentiment']), result))
    print("Precision: ", precision_score(list(test['sentiment']), result))
    matriz = ConfusionMatrix(test['sentiment'], result)

    print(matriz)


if __name__ == '__main__':
    main()
