""" LSTM AE bandymas """

import numpy as np
from src.lstm_ae_funkcijos import (
    lstm_ae,
    lstm_ae_hp
)
from keras_tuner import (
    # RandomSearch,
    GridSearch
)
import tensorflow as tf
import pickle
from keras.callbacks import EarlyStopping

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

gs_low = GridSearch(
    hypermodel = lambda hp: lstm_ae_hp(
        tau=0.025,
        hp=hp,
        n_out_features=y_train.shape[2],
        n_out_timesteps=y_train.shape[1]
    ),
    objective = 'val_loss',
    directory='lstm-ae-05-24-1',
    project_name='lstm-ae-low'
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
print(hyper_low)

print( hyper_low.values, sep='\n' )
print(
      'lstm_dim', hyper_low.get('lstm_dim'),
      'latent_dim', hyper_low.get('latent_dim'),
      'drop_frac', hyper_low.get('drop_frac'),
      'l1', hyper_low.get('l1'),
)
print(gs_low.get_best_hyperparameters())
history_low = model_low.fit(train_dataset, validation_data=val_dataset, epochs=100,
                            callbacks=EarlyStopping(patience=20))

print('Low model fit')


gs_high = GridSearch(
    hypermodel = lambda hp: lstm_ae_hp(
        tau=0.975,
        hp=hp,
        n_out_features=y_train.shape[2],
        n_out_timesteps=y_train.shape[1]
    ),
    objective = 'val_loss',
    directory='lstm-ae-05-24-1',
    project_name='lstm-ae-high'
)

gs_high.search(
    train_dataset,
    epochs=20,
    validation_data=val_dataset
)

hyper_high = gs_high.get_best_hyperparameters()[0]
model_high = lstm_ae(
    tau=0.975,
    n_out_features=2,
    n_out_timesteps=25,
    **hyper_high.values
)
print(
      'lstm_dim', hyper_high.get('lstm_dim'),
      'latent_dim', hyper_high.get('latent_dim'),
      'drop_frac', hyper_high.get('drop_frac'),
      'l1', hyper_high.get('l1'),
)
history_high = model_high.fit(train_dataset, validation_data=val_dataset, epochs=100,
                              callbacks=EarlyStopping(patience=20))

print('High model fit')


with open('output/history_low-final.pkl', 'wb+') as f:
    pickle.dump( history_low.history, f )

with open('output/history_high-final.pkl', 'wb+') as f:
    pickle.dump( history_high.history, f )

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

picp_through_time = np.mean( (y_pred_high >= y_true) & (y_pred_low <= y_true), axis=0)
with open('output/picp-through-time.npy', 'wb+') as f:
    np.save(f, picp_through_time)

pinaw_through_time = np.mean( np.abs(y_pred_high - y_pred_low), axis=0)
with open('output/pinaw-through-time.npy', 'wb+') as f:
    np.save(f, pinaw_through_time)

y_pred_low = model_low.predict(anom_x)
y_pred_high = model_high.predict(anom_x)

y_pred_low.dump('anom-y_pred_low.npy')
y_pred_high.dump('anom-y_pred_high.npy')
