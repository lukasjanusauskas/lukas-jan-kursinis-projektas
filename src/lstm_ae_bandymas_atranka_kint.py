""" LSTM AE bandymas """

import pickle
import numpy as np
from keras.callbacks import EarlyStopping
from src.lstm_ae_funkcijos import (
    lstm_ae,
    # lstm_ae_hp
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

# Normaliu duomenu paruosimas

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



# Pozymiu atrankos

with open('output/best-feats-low.pkl', 'rb+') as f:
    feat_indices_low = pickle.load(f)
X_train_low = X_train[:, :, feat_indices_low]
X_val_low = X_val[:, :, feat_indices_low]

with open('output/best-feats-high.pkl', 'rb+') as f:
    feat_indices_high = pickle.load(f)
X_train_high = X_train[:, :, feat_indices_high]
X_val_high = X_val[:, :, feat_indices_high]

train_dataset_high = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_train_high, y_train)
    )\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=10_000)

val_dataset_high = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_val_high, y_val)
    )\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=10_000)

train_dataset_low = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_train_low, y_train)
    )\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=10_000)

val_dataset_low = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_val_low, y_val)
    )\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=10_000)

# Atrinkti kintamieji
model_high_selected = lstm_ae(
    tau=0.975,
    lstm_dim=10,
    latent_dim=5,
    n_out_features=2,
    n_out_timesteps=25
)
model_high_selected.fit(train_dataset_high, validation_data=val_dataset_high, epochs=100)

model_low_selected = lstm_ae(
    tau=0.025,
    lstm_dim=10,
    latent_dim=10,
    n_out_features=2,
    n_out_timesteps=25
)
model_low_selected.fit(train_dataset_low, validation_data=val_dataset_low, epochs=100)

# Normalus modeliai
model_high = lstm_ae(
    tau=0.975,
    lstm_dim=10,
    latent_dim=5,
    n_out_features=2,
    n_out_timesteps=25,
    drop_frac=0.05,
)
model_high.fit(train_dataset, validation_data=val_dataset, epochs=100)

model_low = lstm_ae(
    tau=0.025,
    lstm_dim=10,
    latent_dim=10,
    n_out_features=2,
    n_out_timesteps=25,
    drop_frac=0.25,
)
model_low.fit(train_dataset, validation_data=val_dataset, epochs=100)

anom_x = np.load('anom-x.npy')
anom_y = np.load('anom-y.npy')

y_pred_low = model_low.predict(anom_x)
y_pred_high = model_high.predict(anom_x)

y_pred_low.dump('anom-y_pred_low.npy')
y_pred_high.dump('anom-y_pred_high.npy')

print(y_pred_low)
print(y_pred_high)

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


test_dataset_low = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_test[:, :, feat_indices_low], y_test)
    )\
    .batch(BATCH_SIZE)

test_dataset_high = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_test[:, :, feat_indices_high], y_test)
    )\
    .batch(BATCH_SIZE)

y_true = []
y_pred_high = []
y_pred_low = []

for (X_batch_low, y_batch), (X_batch_high, _) in zip(
    test_dataset_low.as_numpy_iterator(),
    test_dataset_high.as_numpy_iterator()
):
    y_true.append(y_batch)
    y_pred_low.append(
        model_low_selected.predict(X_batch_low, verbose=0))
    y_pred_high.append(
        model_high_selected.predict(X_batch_high, verbose=0))

y_true = np.concatenate( y_true )

y_pred_high = np.concatenate( y_pred_high )
y_pred_low = np.concatenate( y_pred_low )

print(
    'Pilno modelio PICP',
    np.mean( ( y_true > y_pred_low ) & ( y_true < y_pred_high) )
)

y_true.dump('test-set-y_true-sel.npy')
y_pred_low.dump('test-set-y_pred_low-sel.npy')
y_pred_high.dump('test-set-y_pred_high-sel.npy')

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

print(
    'RFE modelio PICP',
    np.mean( ( y_true > y_pred_low ) & ( y_true < y_pred_high) )
)


y_true.dump('test-set-y_true-full.npy')
y_pred_low.dump('test-set-y_pred_low-full.npy')
y_pred_high.dump('test-set-y_pred_high-full.npy')

