import numpy as np
import quantile_forest as qrf
import pickle
from itertools import product

# DO NOT RUN IN HPC THIS:
import matplotlib.pyplot as plt

N_EST = 256
MAX_D = 16

X_train = np.load('X_train_COGandDifoutput.npy')
y_train = np.load('y_train_COGandDifoutput.npy')

print(X_train.shape)
reshape_params = (-1, X_train.shape[1]*X_train.shape[2])

X_train = X_train.reshape(reshape_params)

def prepare_output_data(i):

    y_train_cog = y_train[:, i, 0].ravel()
    y_val_cog = y_val[:, i, 0].ravel()

    y_train_dif = y_train[:, i, 1].ravel()
    y_val_dif = y_val[:, i, 1].ravel()

    return y_train_cog, y_val_cog, y_train_dif, y_val_dif


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

picps = []
pinaws = []

y_train_cog = y_train[:, 0, 0].ravel()
qrf_model = qrf.RandomForestQuantileRegressor(
    n_estimators=N_EST,
    max_depth=MAX_D,
    n_jobs=-1,
    verbose=True
)

qrf_model.fit(X_train, y_train_cog)

with open('qrf-model.pkl', 'wb+') as f:
    pickle.dump(qrf_model, f)

del X_train
del y_train_cog

X_val = np.load('X_val_COGandDifoutput.npy')
y_val = np.load('y_val_COGandDifoutput.npy')
X_val = X_val.reshape(reshape_params)

y_val_cog = y_val[:, 0, 0].ravel()
metrics = evaluate_metrics(qrf_model, X_val, y_val_cog)
print(metrics)

anom_x = np.load('anom-x.npy')\
    .reshape(reshape_params)

anom_y = np.load('anom-y.npy')\
    [:, 0, 0]

anom_pred = qrf_model.predict(anom_x, quantiles=[0.005, 0.995])
anom_pred.dump('anom-pred-qrf.npy')

plt.plot(anom_y)
plt.plot(anom_pred[:, 0])
plt.plot(anom_pred[:, 1])

plt.show()
