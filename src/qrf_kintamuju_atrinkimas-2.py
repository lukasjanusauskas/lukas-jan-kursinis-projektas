import numpy as np
import quantile_forest as qrf
import pickle

BEST_PARAMS = {
    'n_estimators': 128,
    'max_depth': 16
}

X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')
X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')

y_train_cog = y_train[:, 0, 0]
y_train_dif = y_train[:, 0, 1]
y_val_cog = y_val[:, 0, 0]
y_val_dif = y_val[:, 0, 1]

qrf_model_cog = qrf.RandomForestQuantileRegressor(
    n_jobs=-1,
    verbose=True,
    **BEST_PARAMS
)
qrf_model_dif = qrf.RandomForestQuantileRegressor(
    n_jobs=-1,
    verbose=True,
    **BEST_PARAMS
)

with open('output/best-feats-low.pkl', 'rb+') as f:
    best_cog = np.load(f, allow_pickle=True)

COLS = np.array([
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
])

print(COLS[best_cog])

reshape_params = (-1, X_train.shape[1]*len(best_cog))

X_train_tmp = X_train[:, :, best_cog]\
    .reshape(*reshape_params)
X_val_tmp = X_val[:, :, best_cog]\
    .reshape(*reshape_params)

qrf_model_cog.fit(
    X_train_tmp, y_train_cog
)

y_pred = qrf_model_cog.predict(X_val_tmp, quantiles=[0.025, 0.975])
y_pred_low, y_pred_high = y_pred[:, 0], y_pred[:, 1]

picp = np.mean( (y_val_cog > y_pred_low) & (y_val_cog < y_pred_high) )
pinaw = np.mean( y_pred_high - y_pred_low ) / 2

print(picp, pinaw)


with open('output/best-feats-high-qrf.pkl', 'rb+') as f:
    best_dif = pickle.load(f)

print(COLS[best_dif])

reshape_params = (-1, X_train.shape[1]*len(best_dif))


X_train_tmp = X_train[:, :, best_dif]\
    .reshape(*reshape_params)
X_val_tmp = X_val[:, :, best_dif]\
    .reshape(*reshape_params)

qrf_model_dif.fit(
    X_train_tmp, y_train_dif
)

y_pred = qrf_model_dif.predict(X_val_tmp, quantiles=[0.025, 0.975])
y_pred_low, y_pred_high = y_pred[:, 0], y_pred[:, 1]

picp = np.mean( (y_val_dif > y_pred_low) & (y_val_dif < y_pred_high) )
pinaw = np.mean( y_pred_high - y_pred_low ) / 2

print(picp, pinaw)

anom_x_cog = np.load('anom-x.npy')\
    [:, :, best_cog]\
    .reshape(reshape_params)

anom_x_dif = np.load('anom-x.npy')\
    [:, :, best_dif]\
    .reshape(reshape_params)

anom_y_cog = np.load('anom-y.npy')\
    [:, 0, 0]
anom_y_dif = np.load('anom-y.npy')\
    [:, 0, 1]

anom_pred_cog = qrf_model_cog.predict(anom_x_cog, quantiles=[0.025, 0.975])
anom_pred_cog.dump('anom-pred-qrf-cog.npy')

anom_pred_dif = qrf_model_dif.predict(anom_x_dif, quantiles=[0.025, 0.975])
anom_pred_dif.dump('anom-pred-qrf-dif.npy')
