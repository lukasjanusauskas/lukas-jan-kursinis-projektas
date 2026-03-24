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

params = {
    'n_estimators': [12],
    'max_depth': [8],
}

picps = []
pinaws = []

for time_step in range(0, 25, 5):

    picp_step = []
    pinaw_step = []

    for _ in range(3):
        y_train_cog, y_val_cog, y_train_dif, y_val_dif = prepare_output_data(time_step)

        qrf_model = qrf.RandomForestQuantileRegressor(
            n_estimators=12,
            max_depth=8,
            n_jobs=-1,
            verbose=True
        )

        qrf_model.fit(X_train, y_train_cog)

        metrics = evaluate_metrics(qrf_model, X_val, y_val_cog)

        picp_step.append( metrics['picp'] )
        pinaw_step.append( metrics['pinaw'] )

        print(metrics)

    picp_mean = np.mean(picp_step)
    picps.append(picp_mean)
    print(picp_mean)

    pinaw_mean = np.mean(pinaw_step)
    pinaws.append(pinaw_mean)
    print(pinaw_mean)
    
print('\n'*10)
print(picps)

plt.plot(picps)
plt.show()

print(pinaws)

plt.plot(pinaws)
plt.show()

