import math

import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords

from scripts.word2vec import getWordVector

# test = word_tokenize('(RUS)')
#
# print(test)
#
# input_vector_path = 'unifieddl.kv'
#
# vector = getWordVector(input_vector_path, "(")
#
# # print(vector)
# # print(vector*2)
# # print(vector+vector)
# # nltk.download('stopwords')
#
#
#
# line = 'http://infobeyondtech.com/unifieddl/TargetDamageAssessments#FirerCourse'
# sw_nltk = stopwords.words('english')
# sw_nltk.append('.')
# documents = []
# line = line.replace('-', 'd81baa892b904120ac07c3c2f37d12a9')
# line = line.replace('/', 'd81baa892b904120ac07c3c2f37d12a8')
# line = line.replace(':', 'd81baa892b904120ac07c3c2f37d12a7')
# line = line.replace('#', 'd81baa892b904120ac07c3c2f37d12a6')
#
# documents.append([str(word.lower()).replace('d81baa892b904120ac07c3c2f37d12a9', '-')
#                  .replace('d81baa892b904120ac07c3c2f37d12a8', '/')
#                  .replace('d81baa892b904120ac07c3c2f37d12a7', ':')
#                  .replace('d81baa892b904120ac07c3c2f37d12a6', '#') for word in word_tokenize(line) if
#                   word.lower() not in sw_nltk])
#
# vectorSum = [0] * 50
#
# vectorCounter = 0
#
# print(documents)
#
# for document in documents[0]:
#     try:
#         vector = getWordVector(input_vector_path, document)
#         vectorSum = vectorSum + vector
#         vectorCounter = vectorCounter + 1
#     except:
#         pass
#
# print(vectorSum)
X = [1, 4, -6, 8]
expo = np.exp(X)



print(np.sum(X))

