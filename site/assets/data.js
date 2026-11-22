/* Generated from preds_*.npz and best_params.json */
window.RESULTS = {
 "single": {
  "lightgbm_tuned": 0.385,
  "xgboost_tuned": 0.39142,
  "catboost_tuned": 0.39998,
  "rf_tuned": 0.40697,
  "lstm": 0.42304,
  "s1": 0.38571,
  "s2": 0.38595,
  "d3": 0.38721,
  "lightgbm": 0.38817,
  "xgboost": 0.39327,
  "catboost": 0.4101,
  "rf": 0.41365,
  "s0": 0.39001
 },
 "horizon": {
  "lightgbm_tuned": [
   0.3915,
   0.376,
   0.3813,
   0.3722,
   0.3884,
   0.3536,
   0.372,
   0.3823,
   0.3949,
   0.3659,
   0.3798,
   0.4026,
   0.3913,
   0.3952,
   0.4103,
   0.3985
  ],
  "xgboost_tuned": [
   0.3981,
   0.3872,
   0.3875,
   0.3705,
   0.394,
   0.3541,
   0.3728,
   0.3863,
   0.395,
   0.3687,
   0.38,
   0.3993,
   0.393,
   0.4049,
   0.4115,
   0.4509
  ],
  "catboost_tuned": [
   0.408,
   0.3892,
   0.3904,
   0.3823,
   0.4,
   0.3592,
   0.3836,
   0.3938,
   0.3984,
   0.3726,
   0.3827,
   0.4067,
   0.411,
   0.4062,
   0.4199,
   0.4818
  ],
  "rf_tuned": [
   0.42,
   0.4119,
   0.4087,
   0.3947,
   0.4116,
   0.3762,
   0.3954,
   0.4016,
   0.4047,
   0.3792,
   0.3976,
   0.4441,
   0.4198,
   0.4226,
   0.4159,
   0.4023
  ],
  "lstm": [
   0.3983,
   0.4174,
   0.4087,
   0.3881,
   0.4195,
   0.3915,
   0.4093,
   0.4329,
   0.4318,
   0.4059,
   0.4149,
   0.4121,
   0.4328,
   0.4113,
   0.4655,
   0.5122
  ],
  "d3": [
   0.385,
   0.3698,
   0.377,
   0.377,
   0.3976,
   0.3593,
   0.3783,
   0.3866,
   0.401,
   0.3736,
   0.3769,
   0.4035,
   0.3862,
   0.3941,
   0.4111,
   0.4136
  ],
  "s1": [
   0.3923,
   0.3751,
   0.3817,
   0.3694,
   0.3864,
   0.3531,
   0.3753,
   0.3825,
   0.3906,
   0.3689,
   0.3776,
   0.4021,
   0.3908,
   0.3998,
   0.407,
   0.4141
  ]
 },
 "blend": {
  "lightgbm_tuned+d3": 0.38206,
  "lightgbm_tuned+s1+s2+d3": 0.38208,
  "s1+s2+d3": 0.38225,
  "lightgbm_tuned+s2+d3": 0.38203,
  "lightgbm_tuned+d3+xgboost_tuned": 0.38294,
  "lightgbm_tuned+xgboost_tuned": 0.38585,
  "lightgbm_tuned+s1+s2+d3+xgboost_tuned": 0.38272,
  "lightgbm_tuned+d3+lstm": 0.38608,
  "lightgbm_tuned+d3+rf_tuned": 0.38474,
  "lightgbm_tuned+d3+catboost_tuned": 0.38393,
  "lightgbm_tuned+s1+d3": 0.38194
 },
 "trials": {
  "lightgbm": [
   {
    "rmsle": 0.38863,
    "cfg": "{}"
   },
   {
    "rmsle": 0.38821,
    "cfg": "{'num_leaves': 63}"
   },
   {
    "rmsle": 0.38753,
    "cfg": "{'num_leaves': 255}"
   },
   {
    "rmsle": 0.38866,
    "cfg": "{'min_data_in_leaf': 100}"
   },
   {
    "rmsle": 0.38906,
    "cfg": "{'min_data_in_leaf': 200}"
   },
   {
    "rmsle": 0.39092,
    "cfg": "{'feature_fraction': 0.6}"
   },
   {
    "rmsle": 0.38646,
    "cfg": "{'feature_fraction': 0.95}"
   },
   {
    "rmsle": 0.38797,
    "cfg": "{'lambda_l2': 10}"
   },
   {
    "rmsle": 0.38843,
    "cfg": "{'lambda_l2': 50}"
   },
   {
    "rmsle": 0.38734,
    "cfg": "{'num_leaves': 255, 'min_data_in_leaf': 100}"
   },
   {
    "rmsle": 0.38775,
    "cfg": "{'num_leaves': 255, 'min_data_in_leaf': 200, 'lambda_l2': 10}"
   },
   {
    "rmsle": 0.39448,
    "cfg": "{'num_leaves': 63, 'min_data_in_leaf': 100, 'feature_fraction': 0.6}"
   }
  ],
  "xgboost": [
   {
    "rmsle": 0.39274,
    "cfg": "{}"
   },
   {
    "rmsle": 0.39236,
    "cfg": "{'max_depth': 6}"
   },
   {
    "rmsle": 0.3912,
    "cfg": "{'max_depth': 10}"
   },
   {
    "rmsle": 0.38959,
    "cfg": "{'max_depth': 12}"
   },
   {
    "rmsle": 0.38988,
    "cfg": "{'min_child_weight': 10}"
   },
   {
    "rmsle": 0.39644,
    "cfg": "{'min_child_weight': 200}"
   },
   {
    "rmsle": 0.39302,
    "cfg": "{'colsample_bytree': 0.5}"
   },
   {
    "rmsle": 0.39718,
    "cfg": "{'colsample_bytree': 1.0}"
   },
   {
    "rmsle": 0.39555,
    "cfg": "{'subsample': 0.6}"
   },
   {
    "rmsle": 0.39178,
    "cfg": "{'reg_lambda': 10}"
   },
   {
    "rmsle": 0.39034,
    "cfg": "{'reg_lambda': 50}"
   },
   {
    "rmsle": 0.38957,
    "cfg": "{'max_depth': 10, 'min_child_weight': 100, 'reg_lambda': 10}"
   },
   {
    "rmsle": 0.39328,
    "cfg": "{'max_depth': 10, 'colsample_bytree': 0.6, 'reg_lambda': 10}"
   }
  ],
  "catboost": [
   {
    "rmsle": 0.40125,
    "cfg": "{}"
   },
   {
    "rmsle": 0.40671,
    "cfg": "{'depth': 6}"
   },
   {
    "rmsle": 0.40507,
    "cfg": "{'depth': 10}"
   },
   {
    "rmsle": 0.40106,
    "cfg": "{'depth': 10, 'l2_leaf_reg': 10}"
   },
   {
    "rmsle": 0.40363,
    "cfg": "{'l2_leaf_reg': 10}"
   },
   {
    "rmsle": 0.40681,
    "cfg": "{'l2_leaf_reg': 30}"
   },
   {
    "rmsle": 0.39941,
    "cfg": "{'learning_rate': 0.05}"
   },
   {
    "rmsle": 0.40106,
    "cfg": "{'border_count': 254}"
   },
   {
    "rmsle": 0.40595,
    "cfg": "{'depth': 10, 'learning_rate': 0.05, 'l2_leaf_reg': 10}"
   }
  ],
  "rf": [
   {
    "rmsle": 0.40996,
    "cfg": "{}"
   },
   {
    "rmsle": 0.40819,
    "cfg": "{'min_samples_leaf': 5}"
   },
   {
    "rmsle": 0.41062,
    "cfg": "{'min_samples_leaf': 50}"
   },
   {
    "rmsle": 0.41513,
    "cfg": "{'max_features': 0.33}"
   },
   {
    "rmsle": 0.41033,
    "cfg": "{'max_features': 0.8}"
   },
   {
    "rmsle": 0.40704,
    "cfg": "{'max_samples': 0.7}"
   },
   {
    "rmsle": 0.40705,
    "cfg": "{'min_samples_leaf': 5, 'max_features': 0.8, 'max_samples': 0.7}"
   }
  ]
 }
};
