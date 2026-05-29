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

with open('output/best-feats-low.pkl', 'rb+') as f:
    best_cog = np.load(f, allow_pickle=True)
with open('output/best-feats-high-qrf.pkl', 'rb+') as f:
    best_dif = np.load(f, allow_pickle=True)

y_train_dif = y_train[:, 0, 1]
y_val_dif = y_val[:, 0, 1]


picps = []
pinaws = []

# COG modelis

for time_step in range(25):

    y_val_cog = y_val[:, time_step, 0]
    y_train_cog = y_train[:, time_step, 0]

    qrf_model_cog = qrf.RandomForestQuantileRegressor(
        n_jobs=-1,
        verbose=True,
        **BEST_PARAMS
    )

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

    picps.append(picp)
    pinaws.append(pinaw)


with open('output/picp-qrf-steps-cog.npy', 'wb+') as f:
    pickle.dump(picps, f)
with open('output/pinaw-qrf-steps-cog.npy', 'wb+') as f:
    pickle.dump(pinaws, f)


for time_step in range(25):
    y_val_dif = y_val[:, time_step, 0]
    y_train_dif = y_train[:, time_step, 0]

    qrf_model_dif = qrf.RandomForestQuantileRegressor(
        n_jobs=-1,
        verbose=True,
        **BEST_PARAMS
    )

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

    picps.append(picp)
    pinaws.append(pinaw)

    print(picp, pinaw)


with open('output/picp-qrf-steps-dif.npy', 'wb+') as f:
    pickle.dump(picps, f)
with open('output/pinaw-qrf-steps-dif.npy', 'wb+') as f:
    pickle.dump(pinaws, f)