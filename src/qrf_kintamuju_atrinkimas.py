import pickle
import numpy as np
import quantile_forest as qrf

BEST_PARAMS = {
    'n_estimators': 3,
    'max_depth': 4
}

def atlikti_rfe( X_train, y_train, X_val, y_val, min_poz=2, best_params: dict = BEST_PARAMS):
    # y_train, y_val - vienmaciai; alpha - kvantilis

    boolean_mask_features = [True]*X_train.shape[-1]
    list_features = list(range(X_train.shape[-1]))

    while sum(boolean_mask_features) >= min_poz:
        X_tmp = X_train[:, :, boolean_mask_features]
        X_tmp = X_tmp.reshape((X_tmp.shape[0], -1))
        X_tmp_val = X_val[:, :, boolean_mask_features]
        X_tmp_val = X_tmp_val.reshape((X_tmp_val.shape[0], -1))

        qrf_model = qrf.RandomForestQuantileRegressor(
            n_jobs=-1,
            verbose=True,
            **best_params
        )

        qrf_model.fit(X_tmp, y_train)
        y_pred = qrf_model.predict(X_tmp_val, quantiles=[0.025, 0.975])
        y_pred_low, y_pred_high = y_pred[:, 0], y_pred[:, 1]
        
        picp = np.mean( (y_val > y_pred_low) & (y_val < y_pred_high) )
        pinaw = np.mean( y_pred_high - y_pred_low ) / 2
        cwc = pinaw * (1 + np.exp(0.1 * (0.95 - picp) ))

        print('\nCWC:\n', cwc)

        mdi_scores = qrf_model.feature_importances_

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
    X_train=X_train,
    y_train=y_train[:, 0, 0],
    X_val=X_val,
    y_val=y_val[:, 0, 0],
    min_poz=4
)
best_feats = np.array(list(range(19)))[bool_mask_rfe]
with open('output/best-feats-low.pkl', 'wb+') as f:
    pickle.dump(best_feats, f)

bool_mask_rfe = atlikti_rfe(
    X_train=X_train,
    y_train=y_train[:, 0, 1],
    X_val=X_val,
    y_val=y_val[:, 0, 1],
    min_poz=4
)
best_feats = np.array(list(range(19)))[bool_mask_rfe]

with open('output/best-feats-high-qrf.pkl', 'wb+') as f:
    pickle.dump(best_feats, f)
