""" LSTM AE bandymas """

import numpy as np
from src.lstm_ae_funkcijos import lstm_ae, lstm_ae_hp
import matplotlib.pyplot as plt
import keras_tuner
import tensorflow as tf

BATCH_SIZE = 128

anom_x = np.load('anom-x.npy')
anom_y = np.load('anom-y.npy')

print(anom_y.shape)

X_train = np.load(f'X_train_COGandDifoutput.npy')
y_train = np.load(f'y_train_COGandDifoutput.npy')
X_val = np.load(f'X_val_COGandDifoutput.npy')
y_val = np.load(f'y_val_COGandDifoutput.npy')
X_test = np.load(f'X_test_COGandDifoutput.npy')
y_test = np.load(f'y_test_COGandDifoutput.npy')

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

test_dataset = tf.data\
    .Dataset\
    .from_tensor_slices(
        (X_test, y_test)
    )\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=10_000)

model_low = lstm_ae(
    tau=0.025,
    lstm_dim=20,
    latent_dim=16,
    n_out_features=2,
    n_out_timesteps=25,
    drop_frac=0.25
)

model_low.fit(train_dataset, validation_data=val_dataset, epochs=100)
print('Low model fit')

model_high = lstm_ae(
    tau=0.975,
    lstm_dim=8,
    latent_dim=20,
    n_out_features=2,
    n_out_timesteps=25,
    drop_frac=0.1
)

model_high.fit(train_dataset, validation_data=val_dataset, epochs=100)
print('High model fit')

y_pred_low = model_low.predict(anom_x)
y_pred_high = model_high.predict(anom_x)

np.save('y_low_anom.npy', y_pred_low )
np.save('y_high_anom.npy', y_pred_high )

plt.plot(anom_y[:, 0].ravel(), label='true')
plt.plot(y_pred_low[:, 0].ravel(), label='low')
plt.plot(y_pred_high[:, 0].ravel(), label='high')
plt.savefig('lstm-ae-yipeng3-cog.png')
plt.show()

plt.plot(anom_y[:, 1].ravel(), label='true')
plt.plot(y_pred_low[:, 1].ravel(), label='low')
plt.plot(y_pred_high[:, 1].ravel(), label='high')
plt.savefig('lstm-ae-yipeng3-dif.png')
plt.show()

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
    'PICP:',
    np.mean( (y_pred_high >= y_true) & (y_pred_low <= y_true) )
)
print(
    'PINAW:',
    np.mean( np.abs(y_pred_high - y_pred_low) )
)


