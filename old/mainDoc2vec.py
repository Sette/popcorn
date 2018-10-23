

import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import nltk
from sklearn.model_selection import train_test_split
import gensim
from random import shuffle
from sklearn.metrics import confusion_matrix
from gensim.models.doc2vec import TaggedDocument
from bs4 import BeautifulSoup
import re
from KaggleWord2VecUtility import KaggleWord2VecUtility

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words("english")

def cleanCorpus(review):
    review_text = BeautifulSoup(review, "html.parser").get_text().lower()

    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)

    words = [w for w in tokenizer.tokenize(review_text) if w not in stopwords]

    return words


def trainModel(trainingDataset, alpha, min_alpha, num_epochs):
	model = gensim.models.Doc2Vec(dm=0, alpha=0.025, size=20, window=4, min_alpha=0.025, min_count=0)
	model.build_vocab(trainingDataset)
	alpha_delta = (alpha - min_alpha) / num_epochs
	for epoch in range(num_epochs):
		print ('Now training epoch %s'%epoch)
		shuffle(trainingDataset)
		model.alpha = alpha
		model.min_alpha = alpha  # fix the learning rate, no decay
		model.train(trainingDataset)
		alpha -= alpha_delta
	return model

def preprocessDataframe(dataframe):
    dataset = []
    for index,row in enumerate(dataframe.review):
        preprocDoc = cleanCorpus(row)

        dataset.append(TaggedDocument(words=preprocDoc, tags=[dataframe.sentiment[index]]))
    return dataset


def main():
    # Leitura de arquivos e armazenamento em um DF

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                     delimiter="\t", quoting=3)
    '''
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    '''
    #Opcional ---- Cross over com o conjunto de treinamento
    train,test = train_test_split(train, test_size=0.33, random_state=42)

    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,
                                  delimiter="\t", quoting=3)

    documentTrainDataset = preprocessDataframe(train)

    alpha = 0.025
    min_alpha = 0.001
    num_epochs = 20

    print(documentTrainDataset)


    documentModel = trainModel(documentTrainDataset, alpha, min_alpha, num_epochs)

    # model = gensim.models.Doc2Vec.load('mymodeldoc2vec')
    print("==============================Document model==============================")
    # evaluate the model
    tot_sim = 0.0
    documentTrueTags = []
    documentPredTags = []
    for doc in train:
        print("=============== doc ========== ")
        print(doc)
        documentTrueTags.append(doc.tags)
        predVec = documentModel.infer_vector(doc.words)
        print("===== pred sim tags ==========")
        predTags = documentModel.docvecs.most_similar([predVec], topn=5)
        for pt in predTags:
            print(pt)
        documentPredTags.append(predTags[0][0])
        print("=============================")

    print(documentTrueTags)
    print(documentPredTags)
    print(confusion_matrix(documentTrueTags, documentPredTags))


if __name__ == '__main__':
    main()
