# coding:utf-8

from tensorflow.keras.layers import Dropout, Embedding, Dense, Activation, Lambda, LSTM, concatenate, Flatten
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

# graph_dict = getGraph4Text.GetGraph(max_context_l=35, max_e_1=6,
#                                     max_context_m=35, max_e_2=6,
#                                     max_context_r=35)
graph_dict = getGraph4Text.GetGraph_withOneTag(max_context_l=35, max_e_1=6,
                                    max_context_m=35, max_e_2=6,
                                    max_context_r=35)

fltr = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
fltr = GraphConv.preprocess(fltr)
print('fltr.toarray.shape ... ', fltr.toarray().shape)


def Model_treeGCN_softmax_1(node_count, wordvocabsize, w2v_k, word_W,
                            l2_reg=5e-4):
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
    word_embedding_x = Dropout(0.25)(word_embedding_x)

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
    fc = Dense(512, activation='relu')(flatten)
    fc = Dropout(0.5)(fc)

    # LSTM_backward = LSTM(200, activation='tanh', return_sequences=False,
    #                      go_backwards=True, dropout=0.5)(dropout_2)

    # present_node0 = concatenate([feature_node0, LSTM_backward], axis=-1)
    class_output = Dense(120)(fc)
    class_output = Activation('softmax', name='CLASS')(class_output)

    # Build model
    model = Model(inputs=[X_word_in, fltr_in], outputs=class_output)
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])
    return model


def Model_treeGCN_binaryC_1(node_count, wordvocabsize, tagvocabsize,
                            w2v_k, tag2v_k, word_W, tag_W,
                            l2_reg=5e-4):
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
    # word_embedding_x = Dropout(0.25)(word_embedding_x)

    input_tag = Input(shape=(1,), dtype='int32')
    tag_embedding_layer = Embedding(input_dim=tagvocabsize,
                                    output_dim=tag2v_k,
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=True,
                                    weights=[tag_W])
    tag_embedding = tag_embedding_layer(input_tag)

    embedding_x = Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([word_embedding_x, tag_embedding])
    embedding_x = Dropout(0.25)(embedding_x)

    graph_conv_1 = GraphConv(200,
                             activation='relu',
                             kernel_regularizer=l2(l2_reg),
                             use_bias=True)([embedding_x, fltr_in])
    dropout_1 = Dropout(0.5)(graph_conv_1)
    graph_conv_2 = GraphConv(200,
                             activation='relu',
                             kernel_regularizer=l2(l2_reg),
                             use_bias=True)([dropout_1, fltr_in])
    dropout_2 = Dropout(0.5)(graph_conv_2)

    feature_node0 = Lambda(lambda x: x[:, 0])(dropout_2)

    # pool = GlobalAttentionPool(200)(dropout_2)

    flatten = Flatten()(dropout_2)
    fc = Dense(512, activation='relu')(flatten)
    fc = Dropout(0.5)(fc)

    # LSTM_backward = LSTM(200, activation='tanh', return_sequences=False,
    #                      go_backwards=True, dropout=0.5)(dropout_2)

    # present_node0 = concatenate([feature_node0, LSTM_backward], axis=-1)
    class_output = Dense(2)(fc)
    class_output = Activation('softmax', name='CLASS')(class_output)

    # Build model
    model = Model(inputs=[X_word_in, input_tag, fltr_in], outputs=class_output)
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])
    return model


def Model_treeGAT_softmax_1(node_count, wordvocabsize, w2v_k, word_W,
                            l2_reg=5e-4):
    X_word_in = Input(shape=(node_count,), dtype='int32')
    fltr_in = Input(shape=(node_count, node_count), dtype='float32')
    # fltr_in1 = Input(tensor=sparse_tensor_to_dense(sp_matrix_to_sp_tensor(fltr)))

    word_embedding_layer = Embedding(input_dim=wordvocabsize + 1,
                                     output_dim=w2v_k,
                                     input_length=node_count,
                                     mask_zero=True,
                                     trainable=True,
                                     weights=[word_W])
    word_embedding_x = word_embedding_layer(X_word_in)
    word_embedding_x = Dropout(0.25)(word_embedding_x)

    graph_conv_1 = GraphAttention(200,
                                  attn_heads=3,
                                  activation='relu',
                                  kernel_regularizer=l2(l2_reg),
                                  dropout_rate=0.5,
                                  use_bias=True)([word_embedding_x, fltr_in])
    graph_conv_2 = GraphAttention(200,
                                  attn_heads=3,
                                  activation='relu',
                                  kernel_regularizer=l2(l2_reg),
                                  dropout_rate=0.5,
                                  use_bias=True)([graph_conv_1, fltr_in])

    feature_node0 = Lambda(lambda x: x[:, 0])(graph_conv_2)

    pool = GlobalAttentionPool(200)(graph_conv_2)

    flatten = Flatten()(graph_conv_2)
    flatten = Dense(512, activation='relu')(flatten)
    fc = Dropout(0.5)(flatten)

    # LSTM_backward = LSTM(200, activation='tanh', return_sequences=False,
    #                      go_backwards=True, dropout=0.5)(dropout_2)

    # present_node0 = concatenate([feature_node0, LSTM_backward], axis=-1)
    class_output = Dense(120)(fc)
    class_output = Activation('softmax', name='CLASS')(class_output)

    # Build model
    model = Model(inputs=[X_word_in, fltr_in], outputs=class_output)
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])
    return model