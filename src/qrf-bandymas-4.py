import numpy as np
import quantile_forest as qrf
import pickle

N_EST = 128
MAX_D = 16

indices = np.load('output/best-feats-high-qrf.pkl', allow_pickle=True)
print(indices)

indices = np.load('output/best-feats-low-qrf.pkl', allow_pickle=True)
print(indices)

X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')

print(X_train.shape)
reshape_params = (-1, X_train.shape[1]*X_train.shape[2])
X_train = X_train.reshape(reshape_params)

def evaluate_metrics(
    qrf_model: qrf.RandomForestQuantileRegressor,
    X_val: np.ndarray, y_val: np.ndarray
) -> tuple:

    model_output = qrf_model.predict(X_val, quantiles=[0.025, 0.975])

    y_pred_low = model_output[:, 0]
    y_pred_high = model_output[:, 1]

    picp = np.mean(
        (y_pred_low <= y_val) &
        (y_val <= y_pred_high)
    )

    pinaw = .5 * np.mean(
        np.abs(y_pred_high - y_pred_low)
    )

    return {
        'picp': picp,
        'pinaw': pinaw
    }


y_train_cog = y_train[:, 0, 0].ravel()
y_train_dif = y_train[:, 0, 1].ravel()

qrf_model_cog = qrf.RandomForestQuantileRegressor(
    n_estimators=N_EST,
    max_depth=MAX_D,
    n_jobs=-1,
    verbose=True
)

qrf_model_cog.fit(X_train, y_train_cog)
with open('qrf-model-cog.pkl', 'wb+') as f:
    pickle.dump(qrf_model_cog, f)


qrf_model_dif = qrf.RandomForestQuantileRegressor(
    n_estimators=N_EST,
    max_depth=MAX_D,
    n_jobs=-1,
    verbose=True
)

qrf_model_dif.fit(X_train, y_train_dif)
with open('qrf-model-dif.pkl', 'wb+') as f:
    pickle.dump(qrf_model_dif, f)


X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')
X_val = X_val.reshape(reshape_params)

y_val_cog = y_val[:, 0, 0].ravel()
y_val_dif = y_val[:, 0, 1].ravel()


metrics = evaluate_metrics(qrf_model_cog, X_val, y_val_cog)
print('\n'*5)
print('COG metrics', metrics)
print('\n'*5)

metrics = evaluate_metrics(qrf_model_dif, X_val, y_val_dif)
print('\n'*5)
print('dif metrics', metrics)
print('\n'*5)

anom_x = np.load('anom-x.npy')\
    .reshape(reshape_params)

anom_y_cog = np.load('anom-y.npy')\
    [:, 0, 0]
anom_y_dif = np.load('anom-y.npy')\
    [:, 0, 1]

anom_pred_cog = qrf_model_cog.predict(anom_x, quantiles=[0.005, 0.995])
anom_pred_cog.dump('anom-pred-qrf-cog.npy')

anom_pred_dif = qrf_model_dif.predict(anom_x, quantiles=[0.005, 0.995])
anom_pred_dif.dump('anom-pred-qrf-dif.npy')
