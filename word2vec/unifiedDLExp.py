import gensim
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords


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


if __name__ == '__main__':
    relationalSources = [
        "unifiedDLSource1V1",
        "unifiedDLSource1V2",
        "unifiedDLSource1V3",
        "unifiedDLSource1V4",
        "unifiedDLSource1V5",
        "unifiedDLSource1V6",
        "unifiedDLSource1V7",
        "unifiedDLSource1V8",
        "unifiedDLSource1V9",
        "unifiedDLSource1V10",
        "unifiedDLSource1V11",
        "unifiedDLSource1V12",
        "unifiedDLSource1V13",
        "unifiedDLSource1V14",
        "unifiedDLSource1V15",
        "unifiedDLSource1V16",
        "unifiedDLSource1V17",
        "unifiedDLSource1V18",
        "unifiedDLSource1V19",
        "unifiedDLSource1V20"]

    jsonSources = [
        "unifiedDLSource2V1",
        "unifiedDLSource2V2",
        "unifiedDLSource2V3",
        "unifiedDLSource2V4",
        "unifiedDLSource2V5",
        "unifiedDLSource2V6",
        "unifiedDLSource2V7",
        "unifiedDLSource2V8",
        "unifiedDLSource2V9",
        "unifiedDLSource2V10",
        "unifiedDLSource2V11",
        "unifiedDLSource2V12",
        "unifiedDLSource2V13",
        "unifiedDLSource2V14",
        "unifiedDLSource2V15",
        "unifiedDLSource2V16",
        "unifiedDLSource2V17",
        "unifiedDLSource2V18",
        "unifiedDLSource2V19",
        "unifiedDLSource2V20"]

    # print(relationalSources[0])

    for repo1 in relationalSources:
        for repo2 in jsonSources:
            corpusFileName = 'TrainCorpus' + '-' + repo1 + '-RdfRepo-' + repo2 + '-RdfRepo'
            unifiedDLWord2VecModelTrain(corpusFileName)
            print(repo1 + "---" + repo2 + " successful!!")
            pass
    pass
