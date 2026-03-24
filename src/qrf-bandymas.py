import numpy as np
import quantile_forest as qrf
import pickle
from itertools import product
import matplotlib.pyplot as plt

X_train = np.load('X_train_COGandDifoutput.npy')
y_train = np.load('y_train_COGandDifoutput.npy')

X_val = np.load('X_val_COGandDifoutput.npy')
y_val = np.load('y_val_COGandDifoutput.npy')

reshape_params = (-1, X_train.shape[1]*X_train.shape[2])

X_train = X_train.reshape(reshape_params)
X_val = X_val.reshape(reshape_params)

y_train_cog = y_train[:, 0, 0].ravel()
y_val_cog = y_val[:, 0, 0].ravel()

y_train_dif = y_train[:, 0, 1].ravel()
y_val_dif = y_val[:, 0, 1].ravel()

print(X_train.shape, X_train.shape)
print(X_train.shape)

print(y_train.shape, y_val.shape)

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

params = {
    'n_estimators': [16, 64, 128],
    'max_depth': [None, 16, 64],
}

param_sets = list( product(*params.values()) )

grid_search_results = {
    'param_set': [],
    'picp': [],
    'pinaw': []
}

for param_set in param_sets:
    n_est, max_d = param_set

    qrf_model = qrf.RandomForestQuantileRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        n_jobs=-1,
        verbose=True
    )

    qrf_model.fit(X_train, y_train_cog)

    metrics = evaluate_metrics(qrf_model, X_val, y_val_cog)

    grid_search_results['param_set'].append(param_set)
    grid_search_results['picp'].append(
        metrics['picp']
    )
    grid_search_results['pinaw'].append(
        metrics['pinaw']
    )

    print( '='*10 )
    print( param_set )
    print( metrics )
    print( '='*10 )


x = np.load('anom-x.npy')
x = x.reshape( (-1, x.shape[1]*x.shape[2]) )
y = np.load('anom-y.npy')

y_pred = qrf_model.predict(x, quantiles=[0.025, 0.975])
y_pred_low = y_pred[:, 0]
y_pred_high = y_pred[:, 1]

print(y.shape, y[:, 0].ravel().shape, y_pred_low.shape, y_pred_high.shape)

plt.plot(y[:, 0].ravel(), label='true')
plt.plot(y_pred_low, label='low')
plt.plot(y_pred_high, label='high')
plt.legend()

plt.tight_layout()
plt.savefig('qrf-res-yipeng3.png', dpi=750)

print(y_pred_low)
