import gzip
import logging
import os
from argparse import ArgumentParser
from sys import exception

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


def show_file_contents(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)

def read_input_text(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)

def trainModel():
    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "../reviews_data.txt.gz")

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        vector_size=150,
        window=10,
        min_count=2,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)



    # save only the word vectors
    model.wv.save(os.path.join(abspath, "default"))

    model.wv.save('vectors.kv')

    w1 = "dirty"
    print("Most similar to {0}".format(w1), model.wv.most_similar(positive=w1))

    # look up top 6 words similar to 'polite'
    w1 = ["polite"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # look up top 6 words similar to 'france'
    w1 = ["france"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # look up top 6 words similar to 'shocked'
    w1 = ["shocked"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # look up top 6 words similar to 'shocked'
    w1 = ["beautiful"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # get everything related to stuff on the bed
    w1 = ["bed", 'sheet', 'pillow']
    w2 = ['couch']
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            negative=w2,
            topn=10))

    # similarity between two different words
    print("Similarity between 'dirty' and 'smelly'",
          model.wv.similarity(w1="dirty", w2="smelly"))

    # similarity between two identical words
    print("Similarity between 'dirty' and 'dirty'",
          model.wv.similarity(w1="dirty", w2="dirty"))

    # similarity between two unrelated words
    print("Similarity between 'dirty' and 'clean'",
          model.wv.similarity(w1="dirty", w2="clean"))

    print(model.wv['dirty'])
    print(len(model.wv['dirty']))

    print(model.wv['clean'])
    print(len(model.wv['clean']))



def loadModel():
    abspath = os.path.dirname(os.path.abspath(__file__))

    # model = gensim.models.Word2Vec()
    #
    # # save only the word vectors
    # model.wv.load(os.path.join(abspath, "default"))

    reloaded_word_vectors = KeyedVectors.load('vectors.kv')

    # vectors = numpy.load(os.path.join(abspath, "default.vectors.npy"))

    print(reloaded_word_vectors)
    print(reloaded_word_vectors['dirty'])
    #
    # print(vectors['clean'])
    # print(len(vectors['clean']))



def trainCustomModel():
    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "../sample.txt")

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input_text(data_file))
    logging.info("Done reading data file")

    print(documents)

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        vector_size=150,
        window=10,
        min_count=2,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)

    print(model.wv['spy'])

    # look up top 6 words similar to 'polite'
    w1 = ["spy"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))


def doc2vecTrain():
    data = ["I love machine learning. Its awesome.",
            "I love coding in python",
            "I love building chat-bots",
            "they chat amagingly well",
            "obama age 30",
            "michele age 40"]
    data = []
    input_file = "C:\\Users\\jayan\\workspace\\unified-dl\\nlp-in-practice\\word2vec\\doc2vec.txt"
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            data.append(str(line))
            pass

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    test_tagged_data = [word_tokenize(_d.lower(),preserve_line=True) for i, _d in enumerate(data)]

    print(test_tagged_data)
    print(tagged_data)

    max_epochs = 100
    vec_size = 10
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")
    model = Doc2Vec.load("d2v.model")
    # to find the vector of a document which is not in training data
    test_data1 = word_tokenize("boeing".lower())
    v1 = model.infer_vector(test_data1)

    test_data2 = word_tokenize("carry".lower())
    v2 = model.infer_vector(test_data2)
    print("V1_infer", v1)

    w1 = word_tokenize("boeing".lower())
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=50))

    print(model.wv.similarity('P-8A Poseidon', 'maritime patrol aircraft'))
    vector1 = np.array(v1)
    vector2 = np.array(v2)
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

    print(cosine_similarity)


    # to find most similar doc using tags
    # similar_doc = model.docvecs.most_similar('1')
    # print(similar_doc)
    pass


# def unifiedDLWord2VecModelTrain():
#     documents = []
#     input_file = "C:\\Users\\jayan\\workspace\\unified-dl\\nlp-in-practice\\word2vec\\Word2vecTraining-unifiedDLSource1V3L-unifiedDLSource2V3L.txt"
#
#     nltk.download('stopwords')
#     sw_nltk = stopwords.words('english')
#     sw_nltk.append('.')
#     print(sw_nltk)
#
#     with open(input_file, 'rb') as f:
#         lines = f.readlines()
#         for line in lines:
#             # print(line.decode("utf-8"))
#             line = line.decode("utf-8").replace('-', 'd81baa892b904120ac07c3c2f37d12a9')
#             line = line.replace('/', 'd81baa892b904120ac07c3c2f37d12a8')
#             line = line.replace(':', 'd81baa892b904120ac07c3c2f37d12a7')
#             line = line.replace('#', 'd81baa892b904120ac07c3c2f37d12a6')
#
#             documents.append([str(word.lower()).replace('d81baa892b904120ac07c3c2f37d12a9', '-')
#                              .replace('d81baa892b904120ac07c3c2f37d12a8', '/')
#                              .replace('d81baa892b904120ac07c3c2f37d12a7', ':')
#                              .replace('d81baa892b904120ac07c3c2f37d12a6', '#') for word in word_tokenize(line) if word.lower() not in sw_nltk])
#             # words = word_tokenize(line)
#             # print(words)
#             pass
#     print(documents)
#
#     # build vocabulary and train model
#     model = gensim.models.Word2Vec(
#         documents,
#         vector_size=50,
#         window=10,
#         min_count=1,
#         workers=10)
#     model.train(documents, total_examples=len(documents), epochs=200)
#     model.wv.save('unifieddlFinalUnifiedDLSource1V3L-2V3L.kv')


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

def getWordVector(vectorPath, word):
    wv = KeyedVectors.load('C:\\Users\\jayan\\workspace\\unified-dl\\nlp-in-practice\\word2vec\\expVectors\\' + vectorPath, mmap='r')
    print(wv)
    return wv[word]

def getVector(vectorPath):
    wv = KeyedVectors.load('C:\\Users\\jayan\\workspace\\unified-dl\\nlp-in-practice\\word2vec\\expVectors\\' + vectorPath, mmap='r')
    return wv

def unifiedDLVectorLookup(line,input_vector_path):
    sw_nltk = stopwords.words('english')
    sw_nltk.append('.')
    documents = []
    line = line.replace('-', 'd81baa892b904120ac07c3c2f37d12a9')
    line = line.replace('/', 'd81baa892b904120ac07c3c2f37d12a8')
    line = line.replace(':', 'd81baa892b904120ac07c3c2f37d12a7')
    line = line.replace('#', 'd81baa892b904120ac07c3c2f37d12a6')

    documents.append([str(word.lower()).replace('d81baa892b904120ac07c3c2f37d12a9', '-')
                     .replace('d81baa892b904120ac07c3c2f37d12a8', '/')
                     .replace('d81baa892b904120ac07c3c2f37d12a7', ':')
                     .replace('d81baa892b904120ac07c3c2f37d12a6', '#') for word in word_tokenize(line) if
                      word.lower() not in sw_nltk])

    # print(documents)

    vectorSum = [0] * 50
    vectorCounter = 0
    wv = getVector(input_vector_path)
    for document in documents[0]:
        try:
            # vector = getWordVector(input_vector_path, document)
            if document in wv:
                vector = wv[document]
            else:
                vector = [0] * 50

            vectorSum = vectorSum + vector
            vectorCounter = vectorCounter + 1
        except:
            print(exception)
            pass

    if sum(vectorSum) != 0:
        for singleValue in vectorSum:
            print(singleValue)


def unifiedDLVectorLookupMaster():
    # 'TrainCorpus-unifiedDLSource1V1-RdfRepo-unifiedDLSource2V1-RdfRepo.kv'
    #chaff Up ir_flare Up 76mm-fwd Up 762mg-fwd-port Up 762mg-fwd-stbd Up 762mg-aft-port Up 762mg-aft-stbd Up
    # line = 'chaff Up ir_flare Up 76mm-fwd Up 762mg-fwd-port Up 762mg-fwd-stbd Up 762mg-aft-port Up 762mg-aft-stbd Up##UNIFIEDDL-SEPARATOR##TrainCorpus-unifiedDLSource1V1-RdfRepo-unifiedDLSource2V1-RdfRepo.kv'
    # input_vector_path = 'unifieddlWord2VecModelFinalLatest.kv'
    parser = ArgumentParser()
    parser.add_argument("-w", type=str, help="word", dest="word", required=True)
    args = parser.parse_args()
    if args.word:
        line = args.word
    w = line.split("##UNIFIEDDL-SEPARATOR##")

    # print(line)
    # print(w[0])
    unifiedDLVectorLookup(w[0], w[1])

def unifiedDLVectorModelTrainMaster():
    # 'TrainCorpus-unifiedDLSource1V1-RdfRepo-unifiedDLSource2V1-RdfRepo.kv'
    #chaff Up ir_flare Up 76mm-fwd Up 762mg-fwd-port Up 762mg-fwd-stbd Up 762mg-aft-port Up 762mg-aft-stbd Up
    # line = 'chaff Up ir_flare Up 76mm-fwd Up 762mg-fwd-port Up 762mg-fwd-stbd Up 762mg-aft-port Up 762mg-aft-stbd Up##UNIFIEDDL-SEPARATOR##TrainCorpus-unifiedDLSource1V1-RdfRepo-unifiedDLSource2V1-RdfRepo.kv'
    # input_vector_path = 'unifieddlWord2VecModelFinalLatest.kv'
    parser = ArgumentParser()
    parser.add_argument("-w", type=str, help="word", dest="word", required=True)
    args = parser.parse_args()
    if args.word:
        line = args.word
    # w = line.split("##UNIFIEDDL-SEPARATOR##")

    # print(line)
    # print(w[0])
    unifiedDLWord2VecModelTrain(line)

if __name__ == '__main__':
    unifiedDLVectorLookupMaster()
    # unifiedDLWord2VecModelTrain()