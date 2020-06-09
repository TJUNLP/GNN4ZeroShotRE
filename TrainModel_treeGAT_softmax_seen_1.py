# -*- encoding:utf-8 -*-

# import tensorflow as tf
# config = tf.ConfigProto(allow_soft_placement=True)
# #最多占gpu资源的70%
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# #开始不会给tensorflow全部gpu资源 而是按需增加
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

import pickle, datetime, codecs, math, gc, random
import os.path
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
from spektral.layers import GraphConv
from ProcessData import ProcessData_gat_onlySeen
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from NNstruc.NN_GCN import Model_treeGAT_softmax_1

tf.compat.v1.disable_eager_execution()

def test_model3(nn_model, tag2sentDict_test):

    pairs_test, labels_test = ProcessData_gat_onlySeen. \
        Create4Classifier_softmax(tag2sentDict_test, shuffle=True, class_num=120)

    print('pairs_test len = ', len(pairs_test[0]), len(labels_test))

    train_x1_sent = np.asarray(pairs_test[0], dtype="int32")
    train_x1_fltr = np.array(pairs_test[1])
    train_y = np.asarray(labels_test, dtype="int32")

    inputs_test_x = [train_x1_sent, train_x1_fltr]
    inputs_test_y = [train_y]

    loss, acc = nn_model.evaluate(inputs_test_x, inputs_test_y, verbose=1, batch_size=1024)

    P = acc
    R = acc
    F = acc
    print('test class ... P =, R =, F = ', P, R, F)

    return P, R, F


def train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                    resultdir,
                    npoches=100, batch_size=50, retrain=False, inum=0):

    if retrain:
        nn_model.load_weights(modelfile)
        modelfile = modelfile + '.2nd.h5'

    nn_model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    checkpointer = ModelCheckpoint(filepath=modelfile + ".best_model.h5", monitor='val_loss', verbose=0,
                                   save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)

    # nn_model.fit(inputs_train_x, inputs_train_y,
    #              batch_size=batch_size,
    #              epochs=npoches,
    #              verbose=1,
    #              shuffle=True,
    #              validation_split=0.1,
    #
    #              callbacks=[reduce_lr, checkpointer, early_stopping])
    #
    # nn_model.save_weights(modelfile, overwrite=True)
    #
    # print('the test result-----------------------')
    # P, R, F = test_model(nn_model, pairs_test, labels_test, classifer_labels_test, target_vob)
    # print('P = ', P, 'R = ', R, 'F = ', F)

    nowepoch = 1
    increment = 1
    earlystop = 0
    maxF = 0.
    while nowepoch <= npoches:
        nowepoch += increment
        earlystop += 1

        inputs_train_x, inputs_train_y,\
        inputs_dev_x, inputs_dev_y = Dynamic_get_trainSet()

        nn_model.fit(inputs_train_x, inputs_train_y,
                     batch_size=batch_size,
                     epochs=increment,
                     validation_data=[inputs_dev_x, inputs_dev_y],
                     shuffle=True,
                     verbose=1,
                     callbacks=[reduce_lr])

        print('the test result-----------------------')
        # loss, acc = nn_model.evaluate(inputs_dev_x, inputs_dev_y, batch_size=batch_size, verbose=0)
        P, R, F = test_model3(nn_model, tagDict_test)
        if F > maxF:
            earlystop = 0
            maxF = F
            nn_model.save_weights(modelfile, overwrite=True)

        print(str(inum), nowepoch, earlystop, F, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>maxF=', maxF)

        if earlystop >= 10:
            break

    return nn_model


def infer_e2e_model(nnmodel, modelname, modelfile, resultdir, w2file=''):

    nnmodel.load_weights(modelfile)
    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    print('the test 3 result-----------------------')
    P, R, F = test_model3(nn_model, tagDict_test)
    print('P = ', P, 'R = ', R, 'F = ', F)


def SelectModel(modelname, node_count, wordvocabsize,
                tagvocabsize, posivocabsize, charvocabsize,
                word_W, posi_W, tag_W, char_W,
                input_sent_lenth,
                w2v_k, posi2v_k, tag2v_k, c2v_k,
                batch_size=32):

    nn_model = None

    if modelname is 'Model_treeGAT_softmax_1':
        nn_model = Model_treeGAT_softmax_1(node_count=node_count,
                                           wordvocabsize=wordvocabsize,
                                           w2v_k=w2v_k,
                                           word_W=word_W,
                                           l2_reg=5e-4)

    return nn_model


def Dynamic_get_trainSet(shuffle=True):

    pairs_train, labels_train = ProcessData_gat_onlySeen.\
        Create4Classifier_softmax(tagDict_train, shuffle, class_num=120)
    pairs_dev, labels_dev = ProcessData_gat_onlySeen.\
        Create4Classifier_softmax(tagDict_dev, shuffle, class_num=120)
    print('Dynamic_get_trainSet train len = ', len(pairs_train[0]), len(labels_train))
    print('Dynamic_get_trainSet dev len = ', len(pairs_dev[0]), len(labels_dev))

    train_x1_sent = np.asarray(pairs_train[0], dtype="int32")
    train_x1_fltr = np.array(pairs_train[1])
    print(train_x1_fltr.shape)
    train_y = np.asarray(labels_train, dtype="int32")

    dev_x1_sent = np.asarray(pairs_dev[0], dtype="int32")
    dev_y = np.asarray(labels_dev, dtype="int32")
    dev_x1_fltr = np.array(pairs_dev[1])

    inputs_train_x = [train_x1_sent, train_x1_fltr]
    inputs_train_y = [train_y]
    inputs_dev_x = [dev_x1_sent, dev_x1_fltr]
    inputs_dev_y = [dev_y]

    return inputs_train_x, inputs_train_y, inputs_dev_x, inputs_dev_y


if __name__ == "__main__":

    maxlen = 100

    modelname = 'Model_treeGAT_softmax_1'

    print(modelname)

    rel_prototypes_file = './data/WikiReading/rel_class_prototypes.txt.json.txt'
    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = ""
    t2v_file = './data/WikiReading/WikiReading.rel2v.by_glove.100d.txt'

    trainfile = './data/WikiReading/WikiReading_data.random.train.txt'
    testfile = './data/WikiReading/WikiReading_data.random.test.txt'

    resultdir = "./data/result/"

    datafname = 'WikiReading_data_treeGAT.Word.onlySeen'
    model_datafname = modelname + '+' + datafname
    datafile = "./model/model_data/" + datafname + ".pkl"

    modelfile = "next ...."

    hasNeg = False

    batch_size = 256 #512

    retrain = False
    Test = True
    GetVec = False

    if not os.path.exists(datafile):
        print("Precess data....")

        ProcessData_gat_onlySeen.get_data(trainfile, testfile,
                                          w2v_file, c2v_file, t2v_file, datafile,
                                          w2v_k=100, c2v_k=50, t2v_k=100, maxlen=maxlen)

    for inum in range(2, 3):

        tagDict_train, tagDict_dev, tagDict_test, \
        word_vob, word_id2word, word_W, w2v_k, \
        char_vob, char_id2char, char_W, c2v_k, \
        target_vob, target_id2word, \
        posi_W, posi_k, type_W, type_k, \
        max_s, max_posi, max_c = pickle.load(open(datafile, 'rb'))

        nn_model = SelectModel(modelname,
                               node_count=(max_s+6),
                               wordvocabsize=len(word_vob),
                               tagvocabsize=len(target_vob),
                               posivocabsize=max_posi + 1,
                               charvocabsize=len(char_vob),
                               word_W=word_W, posi_W=posi_W, tag_W=type_W, char_W=char_W,
                               input_sent_lenth=max_s,
                               w2v_k=w2v_k, posi2v_k=max_posi + 1, tag2v_k=type_k, c2v_k=c2v_k,
                               batch_size=batch_size)

        modelfile = "./model/" + model_datafname + "__" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Lstm data has extisted: " + datafile)
            print("Training EE model....")
            print(modelfile)
            train_e2e_model(nn_model, modelfile, inputs_train_x=[], inputs_train_y=[],
                            resultdir=resultdir, npoches=100, batch_size=batch_size, retrain=False, inum=inum)

        else:
            if retrain:
                print("ReTraining EE model....")
                train_e2e_model(nn_model, modelfile, inputs_train_x=[], inputs_train_y=[],
                                resultdir=resultdir, npoches=100, batch_size=batch_size, retrain=False, inum=inum)

        if Test:
            print("test EE model....")
            print(datafile)
            print(modelfile)
            infer_e2e_model(nn_model, model_datafname, modelfile, resultdir, w2file=modelfile)


        del nn_model
        gc.collect()


# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

