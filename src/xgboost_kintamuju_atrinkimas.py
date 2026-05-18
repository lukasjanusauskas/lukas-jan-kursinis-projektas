import pickle
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_pinball_loss

def atlikti_rfe(best_params: dict, X_train, y_train, X_val, y_val, alpha, min_poz=2):
    # y_train, y_val - vienmaciai; alpha - kvantilis

    boolean_mask_features = [True]*X_train.shape[-1]
    list_features = list(range(X_train.shape[-1]))

    while sum(boolean_mask_features) >= min_poz:
        X_tmp = X_train[:, :, boolean_mask_features]
        X_tmp = X_tmp.reshape((X_tmp.shape[0], -1))
        X_tmp_val = X_val[:, :, boolean_mask_features]
        X_tmp_val = X_tmp_val.reshape((X_tmp_val.shape[0], -1))

        xgb_model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            **best_params
        )

        xgb_model.fit(X_tmp, y_train)
        y_pred = xgb_model.predict(X_tmp_val)
        loss = mean_pinball_loss(y_val, y_pred)
        print('\nLoss:\n', loss)

        mdi_scores = xgb_model.feature_importances_

        mdis = np.mean(
            mdi_scores.reshape((25, sum(boolean_mask_features))),
            axis=0
        )

        # Source - https://stackoverflow.com/a/58830654
        # Posted by shaik moeed, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-05-18, License - CC BY-SA 4.0

        sorted_indices = sorted(range(len(mdis)), key=lambda k: mdis[k])
        drop_indices = sorted_indices[:2]

        drop_indices_for_mask = [list_features[i] for i in drop_indices]

        boolean_mask_features = [
            belongs and (ix not in drop_indices_for_mask)
            for ix, belongs in enumerate(boolean_mask_features)
        ]

        print(boolean_mask_features)

        list_features = [
            list_el
            for ix, list_el in enumerate(list_features)
            if ix not in drop_indices
        ]

    return boolean_mask_features

with open('output/high-gcv-cog.pkl', 'rb+') as f:
    gcv_cog_high_params = pickle.load(f)

X_train = np.load('X_train_final.npy')
y_train = np.load('y_train_final.npy')
X_val = np.load('X_val_final.npy')
y_val = np.load('y_val_final.npy')

bool_mask_rfe = atlikti_rfe(
    best_params=gcv_cog_high_params,
    X_train=X_train,
    y_train=y_train[:, 0, 0],
    X_val=X_val,
    y_val=y_val[:, 0, 0],
    alpha=0.975,
    min_poz=4
)
best_feats = np.array(list(range(19)))[bool_mask_rfe]
with open('output/best-feats-low.pkl', 'wb+') as f:
    pickle.dump(best_feats, f)

with open('output/low-gcv-cog.pkl', 'rb+') as f:
    gcv_cog_low_params = pickle.load(f)

bool_mask_rfe = atlikti_rfe(
    best_params=gcv_cog_high_params,
    X_train=X_train,
    y_train=y_train[:, 0, 0],
    X_val=X_val,
    y_val=y_val[:, 0, 0],
    alpha=0.025,
    min_poz=4
)
best_feats = np.array(list(range(19)))[bool_mask_rfe]
with open('output/best-feats-high.pkl', 'wb+') as f:
    pickle.dump(best_feats, f)

# with open('output/high-gcv-diff.pkl', 'rb+') as f:
#     gcv_diff_high_params = pickle.load(f)

# with open('output/low-gcv-cog.pkl', 'rb+') as f:
#     gcv_cog_low_params = pickle.load(f)
# with open('output/low-gcv-diff.pkl', 'rb+') as f:
#     gcv_diff_low_params = pickle.load(f)

