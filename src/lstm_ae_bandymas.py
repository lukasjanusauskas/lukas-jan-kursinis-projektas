""" LSTM AE bandymas """

import numpy as np
from src.lstm_ae_funkcijos import (
    # lstm_ae,
    lstm_ae_hp
)
from keras_tuner import RandomSearch
import tensorflow as tf
# import pickle

# DO NOT IMPORT THIS IN HPC
# import matplotlib.pyplot as plt


BATCH_SIZE = 128

anom_x = np.load('anom-x.npy')
anom_y = np.load('anom-y.npy')

print(anom_x.shape)
print(anom_y.shape)

X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')
X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')

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

rs_low = RandomSearch(
    hypermodel = lambda hp: lstm_ae_hp(
        tau=0.025,
        hp=hp,
        n_out_features=y_train.shape[2],
        n_out_timesteps=y_train.shape[1]
    ),
    objective = 'val_loss',
    max_trials=9,
    directory='lstm-ae-05-16-1',
    project_name='lstm-ae-low'
)

rs_low.search(
    train_dataset,
    epochs=20,
    validation_data=val_dataset
)

model_low = rs_low.get_best_models(1)[0]

rs_high = RandomSearch(
    hypermodel = lambda hp: lstm_ae_hp(
        tau=0.975,
        hp=hp,
        n_out_features=y_train.shape[2],
        n_out_timesteps=y_train.shape[1]
    ),
    objective = 'val_loss',
    max_trials=9,
    directory='lstm-ae-05-16-1',
    project_name='lstm-ae-high'
)

rs_high.search(
    train_dataset,
    epochs=20,
    validation_data=val_dataset
)

model_high = rs_low.get_best_models(1)[0]



# history_high = model_high.fit(train_dataset, validation_data=val_dataset, epochs=20)
# print('High model fit')


# with open('output/history_low-final.pkl', 'wb+') as f:
#     pickle.dump( history_low, f )

# with open('output/history_high-final.pkl', 'wb+') as f:
#     pickle.dump( history_high, f )

del X_train, y_train, train_dataset
del X_val, y_val, val_dataset

X_test = np.load('X_test_final.npy')
y_test = np.load('y_test_final.npy')

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

print(
    'PICP:',
    np.mean( (y_pred_high >= y_true) & (y_pred_low <= y_true) )
)
print(
    'PINAW:',
    np.mean( np.abs(y_pred_high - y_pred_low) )
)

print(
    'PICP:',
    np.mean( (y_pred_high >= y_true) & (y_pred_low <= y_true), axis=0)
)
print(
    'PINAW:',
    np.mean( np.abs(y_pred_high - y_pred_low), axis=0)
)


y_pred_low = model_low.predict(anom_x)
y_pred_high = model_high.predict(anom_x)

y_pred_low.dump('anom-y_pred_low.npy')
y_pred_high.dump('anom-y_pred_high.npy')

