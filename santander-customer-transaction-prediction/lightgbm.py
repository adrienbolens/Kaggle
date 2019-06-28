import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
#  warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set(font_scale=1)

random_state = 0
np.random.seed(random_state)
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

test_val = test_df.drop(['ID_code'], axis=1).values
unique_samples = []

# For each feature, identify the unique values in test_df
unique_count = np.zeros_like(test_val)
for feature in range(test_val.shape[1]):
    _, index_, count_ = np.unique(test_val[:, feature], return_counts=True,
                                  return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real, the others are fake
# They are not taken into account for the evaluation
# Identifying them is a key process as they interfere with feature engineering
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = \
    np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

print(f'There are {len(real_samples_indexes)} real test samples.')
print(f'There are {len(synthetic_samples_indexes)} synthetic test samples.')
real_test_df = test_df.iloc[real_samples_indexes]

features = [c for c in train_df.columns if c not in ['target', 'ID_code']]

data_df = pd.concat([train_df, real_test_df], ignore_index=True)

# feature engineering:
# use the distribution of occurrences of distinct values for each feature
# A new feature is defined for each original feature:
# var -> occ_var = n. of occurrences of corresponding value in the whole set
for var in features:
    feature = data_df[var]
    n_occurences = feature.value_counts()[feature].values
    train_df['occ_' + var] = n_occurences[:len(train_df)]
    test_occ = np.zeros(len(test_df))
    test_occ[real_samples_indexes] = n_occurences[-len(real_test_df):]
    test_df['occ_' + var] = test_occ


original_features = features
features = [c for c in train_df.columns if c not in ['target', 'ID_code']]


lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting": 'gbdt',
    "max_depth": -1,
    "num_leaves": 13,
    "learning_rate": 0.01,
    "bagging_freq": 5,
    "bagging_fraction": 0.4,
    #  "feature_fraction": 0.05,
    "feature_fraction": 0.04,
    "min_data_in_leaf": 80,
    "min_sum_hessian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    # "lambda_l1": 5,
    # "lambda_l2": 5,
    "bagging_seed": random_state,
    "verbosity": 1,
    "seed": random_state,
    "num_threads": 16
}

n_splits = 7
n_loops = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                      random_state=random_state)
oof = train_df[['ID_code', 'target']]
oof['predict'] = 0
predictions = test_df[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()

X_test = test_df[features]

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df,
                                                    train_df['target'])):
    X_train = train_df.iloc[trn_idx][features]
    y_train = train_df.iloc[trn_idx]['target']
    X_valid = train_df.iloc[val_idx][features]
    y_valid = train_df.iloc[val_idx]['target']
    p_valid, p_test = 0, 0
    for i in range(n_loops):
        X_t, y_t = X_train, y_train

        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(
            lgb_params,
            trn_data,
            100000,
            valid_sets=[trn_data, val_data],
            early_stopping_rounds=3000,
            verbose_eval=1000,
            evals_result=evals_result
        )
        p_valid += lgb_clf.predict(X_valid)
        p_test += lgb_clf.predict(X_test)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df,
                                       fold_importance_df], axis=0)
    oof['predict'][val_idx] = p_valid/n_loops
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)

    predictions[f'fold{fold+1}'] = p_test/n_loops


mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print(f"Mean auc: {mean_auc:.5f}, std: {std_auc:.5f}. All auc: {all_auc:.5f}")

cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = \
    feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14, 40))
sns.barplot(x="importance", y="feature",
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig(f'{all_auc}_lgbm_importances.png')

# submission
predictions['target'] = \
    np.mean(predictions[[col for col in predictions.columns
                         if col not in ['ID_code', 'target']]].values, axis=1)
#  predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code": test_df["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv(f"lgb_submission.csv", index=False)
oof.to_csv(f'lgb_oof.csv', index=False)
