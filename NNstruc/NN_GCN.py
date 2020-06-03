# coding:utf-8

from tensorflow.keras.layers import Dropout, Embedding, Dense, Activation, Lambda, LSTM
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
from ProcessData import getGraph4Text

tf.compat.v1.disable_eager_execution()

graph_dict = getGraph4Text.GetGraph(max_context_l=35, max_e_1=6,
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
                             use_bias=False)([word_embedding_x, fltr_in])
    dropout_1 = Dropout(0.5)(graph_conv_1)
    graph_conv_2 = GraphConv(200,
                             activation='relu',
                             kernel_regularizer=l2(l2_reg),
                             use_bias=False)([dropout_1, fltr_in])
    dropout_2 = Dropout(0.5)(graph_conv_2)

    # feature_node0 = Lambda(lambda x: x[:, 0])(dropout_2)

    LSTM_forward = LSTM(200, activation='tanh', return_sequences=True,
                        go_backwards=False, dropout=0.5)(dropout_2)
    LSTM_backward = LSTM(200, activation='tanh', return_sequences=False,
                         go_backwards=True, dropout=0.5)(LSTM_forward)


    class_output = Dense(120)(LSTM_backward)
    class_output = Activation('softmax', name='CLASS')(class_output)

    # Build model
    model = Model(inputs=[X_word_in, fltr_in], outputs=class_output)
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  weighted_metrics=['acc'])
    return model

