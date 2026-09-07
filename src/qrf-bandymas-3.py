import numpy as np
import quantile_forest as qrf
import pickle
from sklearn.metrics import mean_pinball_loss
from itertools import product

X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')

X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')

reshape_params = (-1, X_train.shape[1]*X_train.shape[2])

X_train = X_train.reshape(reshape_params)
X_val = X_val.reshape(reshape_params)


params = {
    # 'n_estimators': [4, 12, 32, 128],
    'max_depth': [16, 32, None],
    'max_samples': [0.5, 0.75, 1.0],
    'max_samples_leaf': [1, 3, 5],
    'max_features': [0.5, 0.75, 'sqrt'],
}

param_sets = []
loss_cog = []

for param_set in product( *params.values() ):

    max_depth, max_samples, max_samples_leaf, max_features = param_set
    y_train_cog, y_val_cog = y_train[:, 0, 0], y_val[:, 0, 0]

    qrf_model = qrf.RandomForestQuantileRegressor(
        max_depth=max_depth,
        max_samples=max_samples,
        max_samples_leaf=max_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        verbose=True
    )

    qrf_model.fit(X_train, y_train_cog)

    y_pred_low = qrf_model.predict(X_val, quantiles=[0.025])
    loss = mean_pinball_loss(y_val_cog, y_pred_low, alpha=0.025)
    loss_cog.append(loss)

    param_sets.append(param_set)
    print( param_set, ' achieved val_loss: ', loss )

print('\n'*10)
print( param_sets )

with open('output/loss_cog_qrf.pkl', 'wb+') as f:
    pickle.dump(loss_cog, f)
with open('output/params_cog_qrf.pkl', 'wb+') as f:
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