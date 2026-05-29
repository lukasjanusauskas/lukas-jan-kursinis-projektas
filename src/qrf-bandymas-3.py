import numpy as np
import quantile_forest as qrf
import pickle
from itertools import product

X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')

X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')

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
    'n_estimators': [4, 12, 32, 128],
    'max_depth': [8, 16, 32],
}

param_sets = []
picps_cog = []
pinaws_cog = []

for n_est, max_depth in product( *params.values() ):

    y_train_cog, y_val_cog = y_train[:, 0, 0], y_val[:, 0, 0]

    qrf_model = qrf.RandomForestQuantileRegressor(
        n_estimators=n_est,
        max_depth=max_depth,
        n_jobs=-1,
        verbose=True
    )

    qrf_model.fit(X_train, y_train_cog)

    metrics = evaluate_metrics(qrf_model, X_val, y_val_cog)
    picp, pinaw = metrics['picp'], metrics['pinaw']

    picps_cog.append(picp)
    print(picps_cog)

    pinaws_cog.append(pinaw)
    print(pinaws_cog)

    param_sets.append( (n_est, max_depth) )
    print( (n_est, max_depth) )

print('\n'*10)
print(picps_cog)
print(pinaws_cog)
print( param_sets )

with open('output/picps_cog_qrf.pkl', 'wb+') as f:
    pickle.dump(picps_cog, f)
with open('output/picps_cog_qrf.pkl', 'wb+') as f:
    pickle.dump(pinaws_cog, f)
with open('output/picps_cog_params.pkl', 'wb+') as f:
    pickle.dump(param_sets, f)

"""

# Paleist pasitikrinimui:
picps_dif = []
pinaws_dif = []

for n_est, max_depth in product( *params.values() ):

    y_train_dif, y_val_dif = y_train[:, 0, 1], y_val[:, 0, 1]

    qrf_model = qrf.RandomForestQuantileRegressor(
        n_estimators=n_est,
        max_depth=max_depth,
        n_jobs=-1,
        verbose=True
    )

    qrf_model.fit(X_train, y_train_dif)

    metrics = evaluate_metrics(qrf_model, X_val, y_val_dif)
    picp, pinaw = metrics['picp'], metrics['pinaw']

    picps_dif.append(picp)
    print(picps_dif)

    pinaws_dif.append(pinaw)
    print(pinaws_dif)

    param_sets.append( (n_est, max_depth) )
    print( (n_est, max_depth) )
    
print(picps_dif)
print(pinaws_dif)
print( param_sets )

with open('output/picps_dif_qrf.pkl', 'wb+') as f:
    pickle.dump(picps_dif, f)
with open('output/picps_dif_qrf.pkl', 'wb+') as f:
    pickle.dump(pinaws_dif, f)
with open('output/picps_dif_params.pkl', 'wb+') as f:
    pickle.dump(param_sets, f)

"""