# coding:utf-8

from tensorflow.keras.layers import Dropout, Embedding, Dense, Activation
from tensorflow.keras.layers import Lambda, LSTM, concatenate, Flatten, TimeDistributed
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Bidirectional, Dot
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from spektral.datasets import citation
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
import networkx as nx
import tensorflow as tf
import scipy.sparse as sp
from spektral.layers import GraphConv
from spektral.layers import GraphAttention, GlobalAttentionPool
from ProcessData import getGraph4Text
from tensorflow.python.ops.sparse_ops import sparse_tensor_to_dense
tf.compat.v1.disable_eager_execution()

graph_dict = getGraph4Text.GetGraph(max_context_l=35, max_e_1=6,
                                    max_context_m=30, max_e_2=6,
                                    max_context_r=35)

fltr = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
fltr = GraphConv.preprocess(fltr)
print('fltr.toarray.shape ... ', fltr.toarray().shape)


def Model_LSTM_treeGCN_softmax_1(node_count, wordvocabsize, charvocabsize, posivocabsize,
                            w2v_k, c2v_k, posi2v_k,
                            word_W, char_W, posi_W, maxword_length,
                            l2_reg=5e-4, batch_size=32):
    X_word_in = Input(shape=(node_count,), dtype='int32')
    # fltr_in = Input(shape=(node_count, node_count), sparse=True)
    fltr_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))

    word_embedding_layer = Embedding(input_dim=wordvocabsize + 1,
                                     output_dim=w2v_k,
                                     input_length=node_count,
                                     mask_zero=True,
                                     trainable=True,
                                     weights=[word_W])
    word_embedding_x = word_embedding_layer(X_word_in)
    word_embedding_x = Dropout(0.5)(word_embedding_x)

    char_input_sent_x1 = Input(shape=(node_count-6, maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(
        Embedding(input_dim=charvocabsize,
                  output_dim=c2v_k,
                  batch_input_shape=(batch_size, node_count-6, maxword_length),
                  trainable=True,
                  weights=[char_W]))
    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.5)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(node_count-6,), dtype='int32')
    input_e2_posi_x1 = Input(shape=(node_count-6,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                     output_dim=posi2v_k,
                                     input_length=node_count-6,
                                     mask_zero=False,
                                     trainable=False,
                                     weights=[posi_W])
    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)

    BiLSTM_layer = Bidirectional(LSTM(50, activation='tanh',
                                      return_sequences=True), merge_mode='concat')
    word_embedding_node0to5 = Lambda(lambda x: x[:, :6])(word_embedding_x)
    word_embedding_sent_x1 = Lambda(lambda x: x[:, 6:])(word_embedding_x)
    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.5)(BiLSTM_x1)
    word_embedding_x = Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([word_embedding_node0to5, BiLSTM_x1])
    graph_conv_1 = GraphConv(200,
                             activation='relu',
                             kernel_regularizer=l2(l2_reg),
                             use_bias=True)([word_embedding_x, fltr_in])
    dropout_1 = Dropout(0.5)(graph_conv_1)
    graph_conv_2 = GraphConv(200,
                             activation='relu',
                             kernel_regularizer=l2(l2_reg),
                             use_bias=True)([dropout_1, fltr_in])
    dropout_2 = Dropout(0.5)(graph_conv_2)

    feature_node0 = Lambda(lambda x: x[:, 0])(dropout_2)

    # pool = GlobalAttentionPool(200)(dropout_2)

    flatten = Flatten()(dropout_2)
    fc = Dense(100, activation='tanh')(flatten)
    fc = Dropout(0.5)(fc)

    # LSTM_backward = LSTM(200, activation='tanh', return_sequences=False,
    #                      go_backwards=True, dropout=0.5)(dropout_2)

    # present_node0 = concatenate([feature_node0, lstm_h], axis=-1)
    class_output = Dense(120)(fc)
    class_output = Activation('softmax', name='CLASS')(class_output)

    # Build model
    model = Model(inputs=[X_word_in, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1, fltr_in], outputs=class_output)
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])
    return model


def Model_LSTM_treeGCN_triloss_1(node_count, wordvocabsize, charvocabsize, posivocabsize, tagvocabsize,
                            w2v_k, c2v_k, posi2v_k, tag2v_k,
                            word_W, char_W, posi_W, tag_W, maxword_length,
                            l2_reg=5e-4, batch_size=32,
                                 margin1=0.1, margin2=0.1, margin3=0.1):

    X_word_in = Input(shape=(node_count,), dtype='int32')
    # fltr_in = Input(shape=(node_count, node_count), sparse=True)
    fltr_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))

    word_embedding_layer = Embedding(input_dim=wordvocabsize + 1,
                                     output_dim=w2v_k,
                                     input_length=node_count,
                                     mask_zero=True,
                                     trainable=True,
                                     weights=[word_W])
    word_embedding_x = word_embedding_layer(X_word_in)
    word_embedding_x = Dropout(0.5)(word_embedding_x)

    char_input_sent_x1 = Input(shape=(node_count-6, maxword_length,), dtype='int32')
    char_embedding_sent_layer = TimeDistributed(
        Embedding(input_dim=charvocabsize,
                  output_dim=c2v_k,
                  batch_input_shape=(batch_size, node_count-6, maxword_length),
                  trainable=True,
                  weights=[char_W]))
    char_embedding_sent_x1 = char_embedding_sent_layer(char_input_sent_x1)
    char_cnn_sent_layer = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_sent_x1 = char_cnn_sent_layer(char_embedding_sent_x1)
    char_embedding_sent_x1 = TimeDistributed(GlobalMaxPooling1D())(char_embedding_sent_x1)
    char_embedding_sent_x1 = Dropout(0.5)(char_embedding_sent_x1)

    input_e1_posi_x1 = Input(shape=(node_count-6,), dtype='int32')
    input_e2_posi_x1 = Input(shape=(node_count-6,), dtype='int32')

    embedding_posi_layer = Embedding(input_dim=posivocabsize,
                                     output_dim=posi2v_k,
                                     input_length=node_count-6,
                                     mask_zero=False,
                                     trainable=False,
                                     weights=[posi_W])
    embedding_e1_posi_x1 = embedding_posi_layer(input_e1_posi_x1)
    embedding_e2_posi_x1 = embedding_posi_layer(input_e2_posi_x1)

    BiLSTM_layer = Bidirectional(LSTM(50, activation='tanh',
                                      return_sequences=True, return_state=True), merge_mode='concat')
    word_embedding_node0to5 = Lambda(lambda x: x[:, :6])(word_embedding_x)
    word_embedding_sent_x1 = Lambda(lambda x: x[:, 6:])(word_embedding_x)
    embedding_x1 = concatenate([word_embedding_sent_x1, char_embedding_sent_x1,
                                embedding_e1_posi_x1, embedding_e2_posi_x1], axis=-1)
    BiLSTM_x1 = BiLSTM_layer(embedding_x1)
    BiLSTM_x1 = Dropout(0.5)(BiLSTM_x1)
    word_embedding_x = Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([word_embedding_node0to5, BiLSTM_x1])
    graph_conv_1 = GraphConv(200,
                             activation='relu',
                             kernel_regularizer=l2(l2_reg),
                             use_bias=True)([word_embedding_x, fltr_in])
    dropout_1 = Dropout(0.5)(graph_conv_1)
    graph_conv_2 = GraphConv(200,
                             activation='relu',
                             kernel_regularizer=l2(l2_reg),
                             use_bias=True)([dropout_1, fltr_in])
    dropout_2 = Dropout(0.5)(graph_conv_2)

    flatten = Flatten()(dropout_2)
    fc = Dense(100, activation='tanh')(flatten)
    fc = Dropout(0.5)(fc)

    input_tag_p = Input(shape=(1,), dtype='int32')
    input_tag_n = Input(shape=(1,), dtype='int32')
    input_tag_a = Input(shape=(1,), dtype='int32')
    input_tag_n0 = Input(shape=(1,), dtype='int32')

    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False,
                                    weights=[tag_W])

    tag_embedding_p = tag_embedding_layer(input_tag_p)
    tag_embedding_p = Flatten()(tag_embedding_p)
    tag_embedding_n = tag_embedding_layer(input_tag_n)
    tag_embedding_n = Flatten()(tag_embedding_n)
    tag_embedding_a = tag_embedding_layer(input_tag_a)
    tag_embedding_a = Flatten()(tag_embedding_a)
    tag_embedding_n0 = tag_embedding_layer(input_tag_n0)
    tag_embedding_n0 = Flatten()(tag_embedding_n0)

    right_cos = Dot(axes=-1, normalize=True, name='right_cos')([fc, tag_embedding_p])
    wrong_cos = Dot(axes=-1, normalize=True, name='wrong_cos')([fc, tag_embedding_n])
    # at_cos = Dot(axes=-1, normalize=True, name='at_cos')([tag_embedding_p, tag_embedding_n])
    anchor_cos = Dot(axes=-1, normalize=True, name='anchor_cos')([fc, tag_embedding_a])
    anchor_wrong_cos = Dot(axes=-1, normalize=True, name='anchor_wrong_cos')([fc, tag_embedding_n0])

    isequal_a_n0 = Lambda(lambda x: tf.keras.backend.equal(x[0], x[1]))\
        (tag_embedding_a, tag_embedding_n0)
    isequal_p_n = Lambda(lambda x: tf.keras.backend.equal(x[0], x[1]))\
        (tag_embedding_n, tag_embedding_p)

    if isequal_p_n is True and isequal_a_n0 is False:
        loss = Lambda(lambda x: tf.keras.backend.relu(0. + x[0] - x[1]) +
                                tf.keras.backend.relu(margin2 + x[2] - x[3]) +
                                tf.keras.backend.relu(0. + x[1] - x[3]))\
            ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])
    elif isequal_a_n0 is True and isequal_p_n is False:
        loss = Lambda(lambda x: tf.keras.backend.relu(margin1 + x[0] - x[1]) +
                                tf.keras.backend.relu(0. + x[2] - x[3]) +
                                tf.keras.backend.relu(0. + x[1] - x[3]))\
            ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])
    else:
        loss = Lambda(lambda x: tf.keras.backend.relu(margin1 + x[0] - x[1]) +
                                tf.keras.backend.relu(margin2 + x[2] - x[3]) +
                                tf.keras.backend.relu(margin3 + x[1] - x[3]))\
            ([wrong_cos, right_cos, anchor_wrong_cos, anchor_cos])

    mymodel = Model([X_word_in, input_e1_posi_x1, input_e2_posi_x1, char_input_sent_x1,
                     input_tag_p, input_tag_n, input_tag_a, input_tag_n0, fltr_in], loss)

    mymodel.compile(loss=lambda y_true, y_pred: y_pred, optimizer=Adam(lr=0.001))

    return mymodel

