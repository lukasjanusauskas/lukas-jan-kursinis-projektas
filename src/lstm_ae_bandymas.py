""" LSTM AE bandymas """

import numpy as np
from src.lstm_ae_funkcijos import lstm_ae, lstm_ae_hp
import tensorflow as tf
import pickle

# DO NOT IMPORT THIS IN HPC
import matplotlib.pyplot as plt


BATCH_SIZE = 128

anom_x = np.load('anom-x.npy')
anom_y = np.load('anom-y.npy')

print(anom_x.shape)
print(anom_y.shape)

X_train = np.load(f'X_train_COGandDifoutput.npy')
y_train = np.load(f'y_train_COGandDifoutput.npy')
X_val = np.load(f'X_val_COGandDifoutput.npy')
y_val = np.load(f'y_val_COGandDifoutput.npy')

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

model_low = lstm_ae(
    tau=0.005,
    lstm_dim=20,
    latent_dim=16,
    n_out_features=2,
    n_out_timesteps=25,
    drop_frac=0.25
)

history_low = model_low.fit(train_dataset, validation_data=val_dataset, epochs=100)
print('Low model fit')

model_high = lstm_ae(
    tau=0.995,
    lstm_dim=8,
    latent_dim=20,
    n_out_features=2,
    n_out_timesteps=25,
    drop_frac=0.1
)

history_high = model_high.fit(train_dataset, validation_data=val_dataset, epochs=100)
print('High model fit')


with open('output/history_low.pkl', 'wb+') as f:
    pickle.dump( history_low, f )

with open('output/history_high.pkl', 'wb+') as f:
    pickle.dump( history_high, f )

del X_train, y_train, train_dataset
del X_val, y_val, val_dataset

X_test = np.load(f'X_test_COGandDifoutput.npy')
y_test = np.load(f'y_test_COGandDifoutput.npy')

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

