""" LSTM AE bandymas """

import numpy as np
from src.lstm_ae_funkcijos import (
    lstm_ae,
    lstm_ae_hp
)
from keras_tuner import GridSearch
import tensorflow as tf
import keras
import pickle
from keras.callbacks import EarlyStopping

# DO NOT IMPORT THIS IN HPC
# import matplotlib.pyplot as plt

BATCH_SIZE = 128
keras.utils.set_random_seed(0)
tf.random.set_seed(0)

anom_x = np.load('lstm-anom-x.npy', allow_pickle=True)
anom_y = np.load('lstm-anom-y.npy', allow_pickle=True)

print(anom_x.shape)
print(anom_y.shape)

X_train = np.load('lstm-X_train_final.npy', allow_pickle=True)
y_train = np.load('lstm-y_train_final.npy', allow_pickle=True)
X_val = np.load('lstm-X_val_final.npy', allow_pickle=True)
y_val = np.load('lstm-y_val_final.npy', allow_pickle=True)

print( X_train.shape )
print( X_val.shape )

# https://www.tensorflow.org/tutorials/load_data/numpy:
train_dataset = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_train, y_train)
    )\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=10_000)

val_dataset = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_val, y_val)
    )\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=10_000)

gs_low = GridSearch(
    hypermodel = lambda hp: lstm_ae_hp(
        tau=0.025,
        hp=hp,
        n_out_features=y_train.shape[2],
        n_out_timesteps=y_train.shape[1]
    ),
    objective = 'val_loss',
    directory='lstm-ae-revision-8',
    project_name='lstm-ae-low',
    seed=0
)

gs_low.search(
    train_dataset,
    epochs=20,
    validation_data=val_dataset
)

hyper_low = gs_low.get_best_hyperparameters()[0]
model_low = lstm_ae(
    tau=0.025,
    n_out_features=2,
    n_out_timesteps=25,
    **hyper_low.values
)

print( hyper_low.values, sep='\n' )
print(
      'latent_dim', hyper_low.get('latent_dim'),
      'drop_frac', hyper_low.get('drop_frac'),
      'l1', hyper_low.get('l1'),
)
print(gs_low.get_best_hyperparameters())
history_low = model_low.fit(train_dataset, validation_data=val_dataset, epochs=100,
                            callbacks=EarlyStopping(patience=20))

print('Low model fit')


model_high = lstm_ae(
    tau=0.975,
    n_out_features=2,
    n_out_timesteps=25,
    **hyper_low.values
)
history_high = model_high.fit(train_dataset, validation_data=val_dataset, epochs=100,
                              callbacks=EarlyStopping(patience=20))

print('High model fit')


with open('output/history_low-final.pkl', 'wb+') as f:
    pickle.dump( history_low.history, f )

with open('output/history_high-final.pkl', 'wb+') as f:
    pickle.dump( history_high.history, f )

del X_train, y_train, train_dataset

X_test = np.load('lstm-X_test_final.npy', allow_pickle=True)
y_test = np.load('lstm-y_test_final.npy', allow_pickle=True)

test_dataset = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_test, y_test)
    )\
    .batch(BATCH_SIZE)


y_true = []
y_pred_high = []
y_pred_low = []

for X_batch, y_batch in test_dataset.as_numpy_iterator():
    y_true.append(y_batch)
    y_pred_low.append(
        model_low.predict(X_batch, verbose=0))

    y_pred_high.append(
        model_high.predict(X_batch, verbose=0))

y_true = np.concatenate( y_true )

y_pred_high = np.concatenate( y_pred_high )
y_pred_low = np.concatenate( y_pred_low )

y_true.dump('test-set-y_true.npy')
y_pred_low.dump('test-set-y_pred_low.npy')
y_pred_high.dump('test-set-y_pred_high.npy')

y_true = []
y_pred_high = []
y_pred_low = []

for X_batch, y_batch in val_dataset.as_numpy_iterator():
    y_true.append(y_batch)

    y_pred_low.append(
        model_low.predict(X_batch, verbose=0))

    y_pred_high.append(
        model_high.predict(X_batch, verbose=0))

y_true = np.concatenate( y_true )

y_pred_high = np.concatenate( y_pred_high )
y_pred_low = np.concatenate( y_pred_low )

y_true.dump('val-set-y_true.npy')
y_pred_low.dump('val-set-y_pred_low.npy')
y_pred_high.dump('val-set-y_pred_high.npy')

y_pred_low = model_low.predict(anom_x)
y_pred_high = model_high.predict(anom_x)

y_pred_low.dump('lstm-anom-y_pred_low.npy')
y_pred_high.dump('lstm-anom-y_pred_high.npy')
