import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_pinball_loss
from sklearn.feature_selection import RFECV
import pickle

X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')
X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')

X = np.concat([X_train, X_val])
y = np.concat([y_train, y_val])

print(X.shape)
print(y.shape)

X_flat = X.reshape( (X.shape[0], -1) )
y_cog = y[:, 0, 0]
y_diff = y[:, 0, 1]

print(X_flat.shape)
print(y_cog.shape)

def quantile_loss_for_gcv(alpha):
    def quantile_loss(estimator, X_test, y_test):
        y_pred = estimator.predict(X_test)
        return mean_pinball_loss(y_test, y_pred, alpha=alpha)
    return quantile_loss

xgb_low = XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.025
)

gcv_cog_low = GridSearchCV(
    param_grid={
        'n_estimators': [20, 50, 100],
        'learning_rate': [0.001, 0.1],
    },
    estimator=xgb_low,
    scoring=quantile_loss_for_gcv(0.025),
    cv=5,
    verbose=3
)
gcv_diff_low = GridSearchCV(
    param_grid={
        'n_estimators': [20, 50, 100],
        'learning_rate': [0.001, 0.1],
    },
    estimator=xgb_low,
    scoring=quantile_loss_for_gcv(0.025),
    cv=5,
    verbose=3
)

gcv_cog_low.fit(X_flat, y_cog)
gcv_diff_low.fit(X_flat, y_diff)

with open('output/low-gcv-cog.pkl', 'wb+') as f:
    pickle.dump(gcv_cog_low.best_params_, f)
with open('output/low-gcv-diff.pkl', 'wb+') as f:
    pickle.dump(gcv_diff_low.best_params_, f)

with open('output/low-gcv-cog-mdis.pkl', 'wb+') as f:
    pickle.dump(
        gcv_cog_low.best_estimator_.feature_importances_, f
    )
with open('output/low-gcv-diff-mdis.pkl', 'wb+') as f:
    pickle.dump(
        gcv_diff_low.best_estimator_.feature_importances_, f
    )


xgb_high = XGBRegressor(
    objective='reg:quantileerror',
    quantile_alpha=0.975
)

gcv_cog_high = GridSearchCV(
    param_grid={
        'n_estimators': [20, 50, 100],
        'learning_rate': [0.001, 0.1],
    },
    estimator=xgb_high,
    scoring=quantile_loss_for_gcv(0.975),
    cv=5,
    verbose=3
)
gcv_diff_high = GridSearchCV(
    param_grid={
        'n_estimators': [20, 50, 100],
        'learning_rate': [0.001, 0.1],
    },
    estimator=xgb_high,
    scoring=quantile_loss_for_gcv(0.975),
    cv=5,
    verbose=3
)

gcv_cog_high.fit(X_flat, y_cog)
gcv_diff_high.fit(X_flat, y_diff)

with open('output/high-gcv-cog.pkl', 'wb+') as f:
    pickle.dump(gcv_cog_high.best_params_, f)
with open('output/high-gcv-diff.pkl', 'wb+') as f:
    pickle.dump(gcv_diff_high.best_params_, f)

with open('output/high-gcv-cog-mdis.pkl', 'wb+') as f:
    pickle.dump(
        gcv_cog_high.best_estimator_.feature_importances_, f
    )
with open('output/high-gcv-diff-mdis.pkl', 'wb+') as f:
    pickle.dump(
        gcv_diff_high.best_estimator_.feature_importances_, f
    )
