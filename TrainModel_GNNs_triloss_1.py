# -*- encoding:utf-8 -*-

import pickle, datetime, codecs, math, gc
import os.path
import numpy as np
from ProcessData import ProcessData_gcn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from NNstruc.NN_GCN_triplet import Model_LSTM_treeGCN_triloss_1
import tensorflow as tf

def test_model3(nn_model, tag2sentDict_test):

    predict_all = 0
    predict_right_all = 0
    totel_right_all = 0

    for ii, tag in enumerate(tag2sentDict_test.keys()):
        sents = tag2sentDict_test[tag]

        truth_tag_list = []
        data_s_all_0 = []
        data_e1_posi_all_0 = []
        data_e2_posi_all_0 = []
        char_s_all_0 = []
        data_tag_all = []
        totel_right = 0

        for s in range(1, len(sents)//2):
            totel_right += 1
            totel_right_all += 1

            for si, ty in enumerate(tag2sentDict_test.keys()):

                data_s, data_e1_posi, data_e2_posi, char_s = sents[s]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)
                data_tag_all.append([ty])

                if tag == ty:
                    truth_tag_list.append(si)

        pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0, data_tag_all]

        train_x1_sent = np.asarray(pairs[0], dtype="int32")
        train_x1_e1_posi = np.asarray(pairs[1], dtype="int32")
        train_x1_e2_posi = np.asarray(pairs[2], dtype="int32")
        train_x1_sent_cahr = np.asarray(pairs[3], dtype="int32")

        train_tag = np.asarray(pairs[4], dtype="int32")

        inputs_train_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x1_sent_cahr,
                          train_tag, train_tag, train_tag, train_tag]

        intermediate_layer_model = tf.keras.models.Model(inputs=nn_model.input,
                                                      outputs=nn_model.get_layer('right_cos').output)
        # intermediate_layer_model = keras.models.Model(inputs=nn_model.input,
        #                                               outputs=nn_model.get_layer('right_cos').get_output_at(0))

        predictions = intermediate_layer_model.predict(inputs_train_x, verbose=0, batch_size=batch_size)

        width = len(tag2sentDict_test.keys())
        assert len(predictions) // width == totel_right
        assert len(truth_tag_list) == totel_right

        predict = 0
        predict_right = 0
        best_F = 0

        threshold = 0.0
        while threshold <= 1.01:

            predict_class = 0
            predict_right_class = 0

            for i in range(len(predictions) // width) :
                left = i * width
                right = (i + 1) * width

                subpredictions = predictions[left:right]
                subpredictions = subpredictions.flatten().tolist()
                class_max = max(subpredictions)
                class_where = subpredictions.index(class_max)

                if class_max > threshold:
                    predict_class += 1

                    if class_where == truth_tag_list[i]:
                        predict_right_class += 1

            P = predict_right_class / max(predict_class, 0.000001)
            R = predict_right_class / totel_right
            F = 2 * P * R / max((P + R), 0.000001)

            if F > best_F:
                predict = predict_class
                predict_right = predict_right_class
                best_F = F


            threshold += 0.025

        # print(ii, '  best_F=', best_F)

        predict_all += predict
        predict_right_all += predict_right

    P = predict_right_all / max(predict_all, 0.000001)
    R = predict_right_all / totel_right_all
    F = 2 * P * R / max((P + R), 0.000001)
    print('P =, R =, F = ', P, R, F)

    return P, R, F


'''
def test_model3(nn_model, tag2sentDict_test):

    predict = 0
    predict_right = 0

    predict_class = 0
    predict_right_class = 0


    data_s_all_0 = []
    data_e1_posi_all_0 = []
    data_e2_posi_all_0 = []
    char_s_all_0 = []

    data_tag_all = []

    labels_all = []
    totel_right = 0

    tagDict_prototypes = ProcessData_Siamese_SentPair.\
        get_rel_prototypes(rel_prototypes_file, max_s, max_posi, word_vob, target_vob, char_vob, max_c)
    assert tagDict_prototypes.keys() == tag2sentDict_test.keys()


    truth_tag_list = []
    for tag in tag2sentDict_test.keys():
        sents = tag2sentDict_test[tag]

        for s in range(1, len(sents)//2):
            totel_right += 1

            for si, ty in enumerate(tagDict_prototypes.keys()):

                data_s, data_e1_posi, data_e2_posi, char_s = sents[s]
                data_s_all_0.append(data_s)
                data_e1_posi_all_0.append(data_e1_posi)
                data_e2_posi_all_0.append(data_e2_posi)
                char_s_all_0.append(char_s)

                data_tag_all.append([ty])

                if tag == ty:
                    labels_all.append(1)
                    truth_tag_list.append(si)
                else:
                    labels_all.append(0)


    pairs = [data_s_all_0, data_e1_posi_all_0, data_e2_posi_all_0, char_s_all_0, data_tag_all]

    train_x1_sent = np.asarray(pairs[0], dtype="int32")
    train_x1_e1_posi = np.asarray(pairs[1], dtype="int32")
    train_x1_e2_posi = np.asarray(pairs[2], dtype="int32")
    train_x1_sent_cahr = np.asarray(pairs[3], dtype="int32")

    train_tag = np.asarray(pairs[4], dtype="int32")

    inputs_train_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x1_sent_cahr,
                      train_tag, train_tag, train_tag, train_tag]

    intermediate_layer_model = keras.models.Model(inputs=nn_model.input,
                                                  outputs=nn_model.get_layer('right_cos').output)
    # intermediate_layer_model = keras.models.Model(inputs=nn_model.input,
    #                                               outputs=nn_model.get_layer('right_cos').get_output_at(0))

    predictions = intermediate_layer_model.predict(inputs_train_x, verbose=1, batch_size=batch_size)


    width = len(tag2sentDict_test.keys())
    assert len(predictions) // width == totel_right
    assert len(truth_tag_list) == totel_right
    predict_rank = 0

    P, R, F = 0., 0., 0.
    threshold = 0.0
    while threshold < 1.01:

        predict_class = 0
        predict_right_class = 0

        for i in range(len(predictions) // width) :
            left = i * width
            right = (i + 1) * width
            # subpredictions = predictions[left:right]
            # subpredictions = subpredictions.flatten().tolist()
            #
            # mindis = max(subpredictions)
            # mindis_where = subpredictions.index(mindis)
            #
            # if mindis > 0.5:
            #     predict += 1
            #
            #     if mindis_where == truth_tag_list[i]:
            #         predict_right += 1

            subpredictions = predictions[left:right]
            subpredictions = subpredictions.flatten().tolist()
            class_max = max(subpredictions)
            class_where = subpredictions.index(class_max)

            if class_max > threshold:
                predict_class += 1

                if class_where == truth_tag_list[i]:
                    predict_right_class += 1



        # P = predict_right / max(predict, 0.000001)
        # R = predict_right / totel_right
        # F = 2 * P * R / max((P + R), 0.000001)
        # print('predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)
        # print('test predict_rank = ', predict_rank / totel_right)
        # print('P =, R =, F = ', P, R, F)

        P = predict_right_class / max(predict_class, 0.000001)
        R = predict_right_class / totel_right
        F = 2 * P * R / max((P + R), 0.000001)
        # print('threshold-------------------------', threshold)
        # print('predict_right_class =, predict_class =, totel_right = ', predict_right_class, predict_class, totel_right)
        # print('test class ... P =, R =, F = ', P, R, F)
        print(str(P) + ' ' + str(R))
        threshold += 0.02

    return P, R, F
'''

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
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=0.00001)

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

        inputs_train_x, inputs_train_y = Dynamic_get_trainSet(isdev=False)
        inputs_dev_x, inputs_dev_y = Dynamic_get_trainSet(isdev=True)

        nn_model.fit(inputs_train_x, inputs_train_y,
                               batch_size=batch_size,
                               epochs=increment,
                               validation_data=[inputs_dev_x, inputs_dev_y],
                               shuffle=True,
                               # class_weight={0: 1., 1: 3.},
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

        if earlystop >= 15:
            break

    return nn_model


def infer_e2e_model(nnmodel, modelname, modelfile, resultdir, w2file=''):

    nnmodel.load_weights(modelfile)
    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    print('the test 3 result-----------------------')
    P, R, F = test_model3(nn_model, tagDict_test)
    print('P = ', P, 'R = ', R, 'F = ', F)


def SelectModel(modelname, node_count, wordvocabsize, tagvocabsize, posivocabsize,charvocabsize,
                     word_W, posi_W, tag_W, char_W,
                     w2v_k, posi2v_k, tag2v_k, c2v_k,maxword_length,
                     batch_size=32):
    nn_model = None

    if modelname is 'Model_LSTM_treeGCN_triloss_1':
        margin1 = 0.08
        margin2 = 0.1
        margin3 = 0.1
        nn_model = Model_LSTM_treeGCN_triloss_1(node_count=node_count,
                                                wordvocabsize=wordvocabsize,
                                                charvocabsize=charvocabsize,
                                                posivocabsize=posivocabsize,
                                                tagvocabsize=tagvocabsize,
                                                w2v_k=w2v_k, c2v_k=c2v_k, posi2v_k=posi2v_k, tag2v_k=tag2v_k,
                                                word_W=word_W, char_W=char_W, posi_W=posi_W,tag_W=tag_W,
                                                maxword_length=maxword_length,
                                                batch_size=batch_size,
                                                margin1=margin1, margin2=margin2, margin3=margin3)

    return nn_model


def Dynamic_get_trainSet(isdev):

    pairs_train = ProcessData_gcn.\
        CreateTriplet_RankClassify621(tagDict_train, tagDict_dev, tagDict_test, type_W, isdev=isdev)

    print('CreatePairs train len = ', len(pairs_train[0]))

    train_x1_sent = np.asarray(pairs_train[0], dtype="int32")
    train_x1_e1_posi = np.asarray(pairs_train[1], dtype="int32")
    train_x1_e2_posi = np.asarray(pairs_train[2], dtype="int32")
    train_x1_sent_cahr = np.asarray(pairs_train[3], dtype="int32")
    train_tag_p = np.asarray(pairs_train[4], dtype="int32")
    train_tag_n = np.asarray(pairs_train[5], dtype="int32")
    train_tag_a = np.asarray(pairs_train[6], dtype="int32")
    train_tag_n0 = np.asarray(pairs_train[8], dtype="int32")

    train_y0 = np.zeros(len(pairs_train[0]), dtype="int32")
    # train_y = np.asarray(labels_train, dtype="int32")
    # train_y_classifer = np.asarray(classifer_labels_train, dtype="int32")

    inputs_train_x = [train_x1_sent, train_x1_e1_posi, train_x1_e2_posi, train_x1_sent_cahr,
                      train_tag_p, train_tag_n, train_tag_a, train_tag_n0]
    inputs_train_y = [train_y0]

    return inputs_train_x, inputs_train_y


if __name__ == "__main__":

    maxlen = 100

    modelname = 'Model_LSTM_treeGCN_triloss_1'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    t2v_file = './data/WikiReading/WikiReading.rel2v.by_glove.100d.txt'

    trainfile = './data/WikiReading/WikiReading_data.random.train.txt'
    testfile = './data/WikiReading/WikiReading_data.random.test.txt'

    resultdir = "./data/result/"

    datafname = 'WikiReading_data_treeGCN.WordCharPosi.UnSeen.devsplit'
    model_datafname = modelname + '+' + datafname
    datafile = "./model/model_data/" + datafname + ".pkl"

    batch_size = 512 # 512
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")

        ProcessData_gcn.get_data(trainfile, testfile,
                                 w2v_file, t2v_file, datafile,
                                 w2v_k=100, c2v_k=50, t2v_k=100, maxlen=maxlen)

    for inum in range(0, 3):

        tagDict_train, tagDict_dev, tagDict_test, \
        word_vob, word_id2word, word_W, w2v_k, \
        char_vob, char_id2char, char_W, c2v_k, \
        target_vob, target_id2word, \
        posi_W, posi_k, type_W, type_k, \
        max_s, max_posi, max_c = pickle.load(open(datafile, 'rb'))

        relRankDict = ProcessData_gcn.get_rel_sim_rank(type_W)

        nn_model = SelectModel(modelname,
                               node_count=6+35+6+30+6+35,
                               wordvocabsize=len(word_vob),
                               tagvocabsize=len(target_vob),
                               posivocabsize=max_posi + 1,
                               charvocabsize=len(char_vob),
                               word_W=word_W, posi_W=posi_W, tag_W=type_W, char_W=char_W,
                               maxword_length=max_c,
                               w2v_k=w2v_k, posi2v_k=max_posi + 1, tag2v_k=type_k, c2v_k=c2v_k,
                               batch_size=batch_size)

        modelfile = "./model/" + modelname + "__" + datafname + "__" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Lstm data has extisted: " + datafile)
            print("Training EE model....")
            print(modelfile)
            train_e2e_model(nn_model, modelfile, inputs_train_x=[], inputs_train_y=[],
                            resultdir=resultdir, npoches=100, batch_size=batch_size, retrain=False, inum=inum)

        if Test:
            print("test EE model....")
            print(datafile)
            print(modelfile)
            infer_e2e_model(nn_model, modelname, modelfile, resultdir, w2file=modelfile)


        del nn_model
        gc.collect()


# CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

