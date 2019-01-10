import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")

class InputHelper(object):
    
    def getTsvData(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<3:
                continue
            x1.append(l[0].lower())
            x2.append(l[1].lower())
            y.append(int(l[2]))#np.array([0,1]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)


    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<3:
                continue
            x1.append(l[0].lower())
            x2.append(l[1].lower())
            y.append(int(l[2])) #np.array([0,1]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)  
 
    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
                
    def dumpValidation(self,x1_text,x2_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x1_shuffled=x1_text[shuffled_index]
        x2_shuffled=x2_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x1_dev=x1_shuffled[dev_idx:]
        x2_dev=x2_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('validation.txt'+str(i),'w') as f:
            for text1,text2,label in zip(x1_dev,x2_dev,y_dev):
                f.write(str(label)+"\t"+text1+"\t"+text2+"\n")
            f.close()
        del x1_dev
        del y_dev
    
    # Data Preparatopn
    # ==================================================
    def loadMap(self, token2id_filepath):
        if not os.path.isfile(token2id_filepath):
            print "file not exist, building map"
            buildMap()

        token2id = {}
        id2token = {}
        with open(token2id_filepath) as infile:
            for row in infile:
                row = row.rstrip()#.decode("utf-8")
                token = row.split('\t')[0]
                token_id = int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token
        return token2id, id2token

    def saveMap(self, vocab):
        with open("char2id", "wb") as outfile:
            for idx in range(len(vocab)):
                outfile.write(vocab[idx] + "\t" + str(idx)  + "\r\n")
        print "saved map between token and id"
    
    def getEmbeddings(self, infile_path, embedding_size):
        char2id, id_char = self.loadMap("char2id")
        print len(char2id)
        row_index = 0
        emb_matrix = np.zeros((len(char2id.keys()), embedding_size))
        with open(infile_path, "rb") as infile:
            for row in infile:
                row = row.strip()
                row_index += 1
                items = row.split()
                char = items[0]
                emb_vec = [float(val) for val in items[1:]]
                if char in char2id:
                    emb_matrix[char2id[char]] = emb_vec
        return emb_matrix
    
    def getDataSets(self, training_paths, dev_paths, max_document_length, percent_dev, batch_size):
        x1_text, x2_text, y=self.getTsvData(training_paths)
        
        # Build vocabulary
        print("Building vocabulary")
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor.fit_transform(np.concatenate((x2_text,x1_text),axis=0))
        vocab =  vocab_processor.vocabulary_.__dict__['_reverse_mapping']
        self.saveMap(vocab)
        print("Length of loaded vocabulary ={}".format( len(vocab_processor.vocabulary_)))
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1*len(y_shuffled)*percent_dev//100
        del x1
        del x2

        if dev_paths == None:
            # Split train/dev set
            self.dumpValidation(x1_text,x2_text,y,shuffle_indices,dev_idx,0)
            # TODO: This is very crude, should use cross-validation
            x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
            x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
            y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
            print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))       
        else:
            x1_train, x2_train, y_train = x1_text, x2_text, y
            x1_dev, x2_dev, y_dev = self.getTsvData(dev_paths)
            x1 = np.asarray(list(vocab_processor.transform(x1_dev)))
            x2 = np.asarray(list(vocab_processor.transform(x2_dev)))

        sum_no_of_batches = sum_no_of_batches+(len(y_train)//batch_size)
        train_set=(x1_shuffled,x2_shuffled,y_shuffled)
        dev_set=(x1,x2,y_dev)    
        gc.collect()
        return train_set,dev_set,vocab_processor,sum_no_of_batches
    
    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp,x2_temp,y = self.getTsvTestData(data_path)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print len(vocab_processor.vocabulary_)

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1,x2, y, x1_temp, x2_temp

