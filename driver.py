#train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
#test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation
'''train a SGD classifier using unigram representation,
predict sentiments on imdb_te.csv, and write output to
unigram.output.txt'''
from typing import Any, Union

from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader

'''train a SGD classifier using bigram representation,
predict sentiments on imdb_te.csv, and write output to
bigram.output.txt'''

'''train a SGD classifier using unigram representation
with tf-idf, predict sentiments on imdb_te.csv, and write 
output to unigramtfidf.output.txt'''

'''train a SGD classifier using bigram representation
with tf-idf, predict sentiments on imdb_te.csv, and write 
output to bigramtfidf.output.txt'''

import pandas as pd
import numpy as np
from os import listdir
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

train_path = "aclImdb_v1.tar/aclImdb/train/" # use terminal to ls files under this directory
test_path = "imdb_te.csv" # test data for grade evaluation
train_file_name ="imdb_tr.csv"
test_file_name = "imdb_te.csv"

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''Implement this module to extract
    and combine text files under train_path directory into
    imdb_tr.csv. Each text file in train_path should be stored
    as a row in imdb_tr.csv. And imdb_tr.csv should have two
    columns, "text" and label'''

    fileList = [inpath + "pos/",inpath + "neg/"]
    stopwordsList = open("stopwords.en.txt", "r", encoding ="ISO-8859-1").read()
    stopwordsList = stopwordsList.split("\n")
    #stopwordsList= pd.DataFrame(pd.read_csv("stopwords.en.txt", header=0, encoding ="ISO-8859-1"))
    #stopwordsList = stopwordsList.str.split("\n")

    polarno = 1
    contentList = []
    for dirlist in fileList:
        #print("Directory path is ",dirlist)
        for file in  listdir(dirlist):
            textdata1 = open(dirlist + file, "r", encoding ="ISO-8859-1").read()
            textdata1 = textdata1.split()
            newList = []
            for data in textdata1:
                if data.lower() not in stopwordsList:
                    #textdata1.__delitem__(i)
                    newList.append(data)
            textdata1 = ' '.join(newList)
            contentList.append([textdata1,polarno])
            #print(textdata1)
        polarno = 0
    random.shuffle(contentList)
    pd.DataFrame(contentList,columns=["text","polarity"]).to_csv(train_file_name,sep=",",header=True,index=True,index_label="row_number")

    return None

if __name__ == "__main__":
    imdb_data_preprocess(inpath=train_path)
    dataTrain = pd.read_csv(train_file_name,header=0, encoding ="ISO-8859-1")
    Xtrain,Ytrain = dataTrain["text"],dataTrain["polarity"]
    dataTest = pd.read_csv(test_file_name,header=0, encoding ="ISO-8859-1")
    Xtest = dataTest["text"]

    unigram_data = CountVectorizer(ngram_range=(1,1))
    Xtrain_vec_uni = unigram_data.fit_transform(Xtrain)
    #unigram_data = unigram_data.fit(Xtrain)
    #Xtrain_vec_uni = unigram_data.transform(Xtrain)
    Xtest_vec_uni = unigram_data.transform(Xtest)
    sgduniclass = SGDClassifier(loss='hinge',penalty='l1')
    sgduniclass.fit(Xtrain_vec_uni,Ytrain)
    Ytest_vec_uni = sgduniclass.predict(Xtest_vec_uni)
    #print("Unigram scor is ",sgduniclass.score(Xtrain_vec_uni,Ytrain))
    np.savetxt("unigram.output.txt", Ytest_vec_uni, newline="\n",fmt="%d")


    uni_tfi_tran = TfidfTransformer()
    Xtrain_vec_uniti = uni_tfi_tran.fit_transform(Xtrain_vec_uni)
    #uni_tfi_tran = uni_tfi_tran.fit(Xtrain_vec_uni)
    #Xtrain_vec_uniti = uni_tfi_tran.transform(Xtrain_vec_uni)
    Xtest_vec_uniti = uni_tfi_tran.transform(Xtest_vec_uni)
    sgduniclass = SGDClassifier(loss='hinge',penalty='l1')
    sgduniclass.fit(Xtrain_vec_uniti,Ytrain)
    Ytest_vec_uniti = sgduniclass.predict(Xtest_vec_uniti)
    #print("Unigram tifi scor is ",sgduniclass.score(Xtrain_vec_uniti,Ytrain))
    np.savetxt("unigramtfidf.output.txt", Ytest_vec_uniti, newline="\n",fmt="%d")


    bigram_data = CountVectorizer(ngram_range=(1,2))
    Xtrain_vec_bi = bigram_data.fit_transform(Xtrain)
    #bigram_data = bigram_data.fit(Xtrain)
    #Xtrain_vec_bi = bigram_data.transform(Xtrain)
    Xtest_vec_bi = bigram_data.transform(Xtest)
    sgduniclass = SGDClassifier(loss='hinge',penalty='l1')
    sgduniclass.fit(Xtrain_vec_bi,Ytrain)
    Ytest_vec_bi = sgduniclass.predict(Xtest_vec_bi)
    np.savetxt("bigram.output.txt", Ytest_vec_bi, newline="\n",fmt="%d")
    #print("Bigram  scor is ",sgduniclass.score(Xtrain_vec_bi,Ytrain))

    bi_tfi_tran = TfidfTransformer()
    Xtrain_vec_biti = bi_tfi_tran.fit_transform(Xtrain_vec_bi)
    #bi_tfi_tran = bi_tfi_tran.fit(Xtrain_vec_bi)
    #Xtrain_vec_biti = bi_tfi_tran.transform(Xtrain_vec_bi)
    Xtest_vec_biti = bi_tfi_tran.transform(Xtest_vec_bi)
    sgduniclass = SGDClassifier(loss='hinge',penalty='l1')
    sgduniclass.fit(Xtrain_vec_biti,Ytrain)
    Ytest_vec_biti = sgduniclass.predict(Xtest_vec_biti)
    np.savetxt("bigramtfidf.output.txt", Ytest_vec_biti, newline="\n",fmt="%d")
    #print("Bigram tifi scor is ",sgduniclass.score(Xtrain_vec_biti,Ytrain))






