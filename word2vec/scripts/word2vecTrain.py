import gzip
import logging
import os
from argparse import ArgumentParser
import gensim
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import RegexpTokenizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)





def unifiedDLWord2VecModelTrain(corpusFileName):
    print('test')
    documents = []
    input_file = "C:\\Users\\jayan\\workspace\\unified-dl\\nlp-in-practice\\word2vec\\expCorpus\\" + corpusFileName + '.txt'

    nltk.download('stopwords')
    sw_nltk = stopwords.words('english')
    sw_nltk.append('.')
    print(sw_nltk)

    with open(input_file, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            # print(line.decode("utf-8"))
            line = line.decode("utf-8").replace('-', 'd81baa892b904120ac07c3c2f37d12a9')
            line = line.replace('/', 'd81baa892b904120ac07c3c2f37d12a8')
            line = line.replace(':', 'd81baa892b904120ac07c3c2f37d12a7')
            line = line.replace('#', 'd81baa892b904120ac07c3c2f37d12a6')
            documents.append([str(word.lower()).replace('d81baa892b904120ac07c3c2f37d12a9', '-')
                             .replace('d81baa892b904120ac07c3c2f37d12a8', '/')
                             .replace('d81baa892b904120ac07c3c2f37d12a7', ':')
                             .replace('d81baa892b904120ac07c3c2f37d12a6', '#') for word in word_tokenize(line) if word.lower() not in sw_nltk])
            #  words = word_tokenize(line)
            # print(words)
            pass
    print(documents)

    print('Complete parsing corpus file')
    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        vector_size=50,
        window=10,
        min_count=1,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=200)
    model.wv.save('C:\\Users\\jayan\\workspace\\unified-dl\\nlp-in-practice\\word2vec\\expVectors\\'+corpusFileName+".kv")

def unifiedDLVectorModelTrainExperiemnts():
    for i in range(1, 20):
        for j in range(21, 70):
            trainCorpus = 'TrainCorpus-unifiedDLSource1V' + str(i) + '-RdfRepo-unifiedDLSource2V' + str(j) + '-RdfRepo'
            # trainCorpus = 'TrainCorpus - unifiedDLSource1V' + str(i) + ' - RdfRepo - unifiedDLSource2V' + str(j) + ' - RdfRepo'
            unifiedDLWord2VecModelTrain(trainCorpus)
            print(trainCorpus + " Train Successful: True")
            print("-------------------------------------")
            print("-------------------------------------")
            print("-------------------------------------")
            print("-------------------------------------")


def unifiedDLVectorModelTrainMaster():
    # 'TrainCorpus-unifiedDLSource1V1-RdfRepo-unifiedDLSource2V1-RdfRepo.kv'
    #chaff Up ir_flare Up 76mm-fwd Up 762mg-fwd-port Up 762mg-fwd-stbd Up 762mg-aft-port Up 762mg-aft-stbd Up
    # line = 'chaff Up ir_flare Up 76mm-fwd Up 762mg-fwd-port Up 762mg-fwd-stbd Up 762mg-aft-port Up 762mg-aft-stbd Up##UNIFIEDDL-SEPARATOR##TrainCorpus-unifiedDLSource1V1-RdfRepo-unifiedDLSource2V1-RdfRepo.kv'
    # input_vector_path = 'unifieddlWord2VecModelFinalLatest.kv'
    parser = ArgumentParser()
    parser.add_argument("-t", type=str, help="word", dest="corpusFileName", required=True)
    args = parser.parse_args()
    if args.corpusFileName:
        line = args.corpusFileName
    unifiedDLWord2VecModelTrain(line)
    # TrainCorpus - unifiedDLSource1V1 - RdfRepo - unifiedDLSource2V21 - RdfRepo


if __name__ == '__main__':
    # unifiedDLVectorLookupMaster()
     unifiedDLVectorModelTrainMaster()
    # unifiedDLVectorModelTrainExperiemnts()