import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import mnist
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor

tf.compat.v1.disable_eager_execution()

# Parameters
l2_reg = 5e-4         # Regularization rate for l2
learning_rate = 1e-3  # Learning rate for SGD
batch_size = 32       # Batch size
epochs = 1         # Number of training epochs 1000
es_patience = 10      # Patience fot early stopping

# Load data
X_train, y_train, X_val, y_val, X_test, y_test, A = mnist.load_data()

X_train, X_val, X_test = X_train[..., None], X_val[..., None], X_test[..., None]
N = X_train.shape[-2]      # Number of nodes in the graphs  784
print('N-----', N)
F = X_train.shape[-1]      # Node features dimensionality  1
print('f______', F)
n_out = 10                 # Dimension of the target

fltr = GraphConv.preprocess(A)

# Model definition
X_in = Input(shape=(N, F))
# Pass A as a fixed tensor, otherwise Keras will complain about inputs of
# different rank.
A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))

graph_conv = GraphConv(32,
                       activation='elu',
                       kernel_regularizer=l2(l2_reg))([X_in, A_in])
graph_conv = GraphConv(32,
                       activation='elu',
                       kernel_regularizer=l2(l2_reg))([graph_conv, A_in])
flatten = Flatten()(graph_conv)
fc = Dense(512, activation='relu')(flatten)
output = Dense(n_out, activation='softmax')(fc)

# Build model
model = Model(inputs=[X_in, A_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
model.summary()

# Train model
validation_data = (X_val, y_val)
model.fit(X_train,
          y_train,
          batch_size=batch_size,
          validation_data=validation_data,
          epochs=epochs,
          callbacks=[
              EarlyStopping(patience=es_patience, restore_best_weights=True)
          ])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate(X_test,
                              y_test,
                              batch_size=batch_size)
print('Done.\n'
      'Test loss: {}\n'
      'Test acc: {}'.format(*eval_results))
