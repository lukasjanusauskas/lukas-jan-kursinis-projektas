""" LSTM AE bandymas """

import numpy as np
from src.lstm_ae_funkcijos import lstm_ae
import tensorflow as tf
from sklearn.metrics import mean_pinball_loss
# import pickle

BATCH_SIZE = 128
EPOCHS = 1

BEST_LOWER = {
    'lstm_dim':10,
    'latent_dim':10,
    'n_out_features':2,
    'n_out_timesteps':25,
    'drop_frac':0.25,
}

BEST_UPPER = {
    'lstm_dim':10,
    'latent_dim':5,
    'n_out_features':2,
    'n_out_timesteps':25,
    'drop_frac':0.25,
}

# Normaliu duomenu paruosimas
X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')
X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')

print( X_train.shape )
print( X_val.shape )

def apmokyti_ivertinti(X_train, y_train, X_val, y_val, tmp_list, tau, best_params):
    # https://www.tensorflow.org/tutorials/load_data/numpy:
    train_dataset = tf.data\
        .Dataset\
        .from_tensor_slices(
            (X_train[:, :, tmp_list], y_train)
        )\
        .batch(BATCH_SIZE)\
        .shuffle(buffer_size=10_000)

    val_dataset = tf.data\
        .Dataset\
        .from_tensor_slices(
            (X_val[:, :, tmp_list], y_val)
        )\
        .batch(BATCH_SIZE)\
        .shuffle(buffer_size=10_000)

    model = lstm_ae(
        tau=tau,
        **best_params
    )
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

    return np.min( history.history['val_loss'] )


def forward_feat_select(X_train, y_train, X_val, y_val, tau: float, best_params: dict, max_feats: int=2):

    n_feat = X_train.shape[2]
    index_list = []

    while len(index_list) < max_feats:

        min_loss = np.inf
        best_index = None

        for i in range(n_feat):

            if i in index_list:
                continue

            tmp_list = index_list + [i]

            loss = apmokyti_ivertinti(
                X_train, y_train, X_val, y_val, tmp_list, tau, best_params
            )

            if loss < min_loss:
                min_loss = loss
                best_index = i

        index_list.append(best_index)

    return index_list

COLS = [
    'SOG',
    'COG',
    'currentSpeed',
    'gust',
    'swellHeight',
    'waveHeight',
    'windSpeed',
    'COG_vec_x',
    'COG_vec_y',
    'currentDirection_vec_x',
    'currentDirection_vec_y',
    'swellDirection_vec_x',
    'swellDirection_vec_y',
    'waveDirection_vec_x',
    'waveDirection_vec_y',
    'windDirection_vec_x',
    'windDirection_vec_y',
    'COG-dif',
    'delta_dif'
]
features = forward_feat_select(
    X_train, y_train, X_val, y_val, 0.025, BEST_LOWER
)

print( np.array(COLS)[features].tolist() )
