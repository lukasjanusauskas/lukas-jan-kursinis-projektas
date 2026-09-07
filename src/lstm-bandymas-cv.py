import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import GridSearch
from keras.callbacks import EarlyStopping
from src.lstm_ae_funkcijos import lstm_ae
from sklearn.model_selection import KFold
from src.lstm_ae_funkcijos import lstm_ae_hp

BATCH_SIZE = 128

X_train = np.load('lstm-X_train_final.npy')
y_train = np.load('lstm-y_train_final.npy')

X_val = np.load('lstm-X_val_final.npy')
y_val = np.load('lstm-y_val_final.npy')

SLIDINGW_STEPS = 5
N_OUT_FEATS = y_train.shape[-1]
N_INP_FEATS = X_train.shape[-1]

#merge test and validation to do 5-fold cross-validation
X_train_val = np.concat([X_train, X_val])
y_train_val = np.concat([y_train, y_val])

n_row = X_train_val.shape[0]
n_time = y_train_val.shape[1]
n_inp = X_train_val.shape[2]
n_out = y_train_val.shape[2]


params = GridSearch(
    hypermodel = lambda hp: lstm_ae_hp(
        tau=0.025,
        hp=hp,
        n_out_features=y_train.shape[2],
        n_out_timesteps=y_train.shape[1]
    ),
    objective = 'val_loss', directory='lstm-ae-revision-8', project_name='lstm-ae-low', seed=0)\
    .get_best_hyperparameters()[0].values

results = {
    'picp_delta_cog': [],
    'pinaw_delta_cog': [],
    'picp_delta_dif': [],
    'pinaw_delta_dif': []
}

fold_obj =  KFold(n_splits=5, shuffle=True, random_state=0).split(X_train_val) 
for ix, (train_index, val_index) in enumerate(fold_obj):

    X_train_tmp = X_train_val\
        [train_index, :]
    y_train_tmp = y_train_val[train_index, :]
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
    ).batch(BATCH_SIZE).shuffle(buffer_size=10_000)

    X_val_tmp = X_train_val\
        [val_index, :]
    y_val_tmp = y_train_val[val_index, :]
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)
    ).batch(BATCH_SIZE).shuffle(buffer_size=10_000)

    model_low = lstm_ae(
        tau=0.025,
        n_out_features=n_out,
        n_out_timesteps=n_time,
        **params
    )
    _ = model_low.fit(
        x=X_train_tmp,
        y=y_train_tmp,
        validation_data=(X_val_tmp, y_val_tmp),
        epochs=1,
        callbacks=EarlyStopping(patience=20)
    )
    model_high = lstm_ae(
        tau=0.975,
        n_out_features=n_out,
        n_out_timesteps=n_time,
        **params
    )
    _ = model_high.fit(
        x=X_train_tmp,
        y=y_train_tmp,
        validation_data=(X_val_tmp, y_val_tmp),
        epochs=1,
        callbacks=EarlyStopping(patience=20)
    )

    y_pred_low = model_low.predict(X_val_tmp)
    y_pred_high = model_high.predict(X_val_tmp)

    in_pi =  (y_pred_low < y_val_tmp) & (y_pred_high > y_val_tmp) 
    in_pi = in_pi.reshape((-1, 25, 2))
    in_pi_delta_cog = in_pi[:, :, 0]
    in_pi_delta_dif = in_pi[:, :, 1]

    width_pi =  np.abs(y_pred_high - y_pred_low)
    width_pi = width_pi.reshape((-1, 25, 2))
    width_pi_delta_cog = width_pi[:, :, 0]
    width_pi_delta_dif = width_pi[:, :, 1]

    results['picp_delta_cog'].append(np.mean(in_pi_delta_cog))
    results['picp_delta_dif'].append(np.mean(in_pi_delta_dif))
    results['pinaw_delta_cog'].append(np.mean(width_pi_delta_cog))
    results['pinaw_delta_dif'].append(np.mean(width_pi_delta_dif))

    print( np.mean(in_pi_delta_cog) )
    print( np.mean(in_pi_delta_dif) )
    print( np.mean(width_pi_delta_cog) )
    print( np.mean(width_pi_delta_dif) )

print(results)
pd.DataFrame(results).to_csv('results-lstm.csv')

