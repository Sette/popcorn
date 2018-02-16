

import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import nltk
from sklearn.model_selection import train_test_split


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

    # Verifica o numero de registros carregados
    print("Read %d registros de treinamento com label, %d registros de teste, registros de treinamento sem Label " \
          % (train["review"].size,
             test["review"].size,unlabeled_train["review"].size))



if __name__ == '__main__':
    main()
