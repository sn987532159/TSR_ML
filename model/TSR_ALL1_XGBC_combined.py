import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import TomekLinks
import joblib

# import data sets
csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_X_TRAIN.csv")
X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_y_TRAIN.csv")
y_train = pd.read_csv(csv_path)
y_train = np.ravel(y_train)

# class weight
y_train_1 = sum(y_train)
y_train_0 = len(y_train) - sum(y_train)

# base xgbc
xgbc = XGBClassifier(booster = "gbtree", random_state=19, use_label_encoder=False, eval_metric = "auc", tree_method = "hist", n_jobs=-1)

# tune xgbc
hyperparameters_xgbc = {"xgbclassifier__learning_rate": (0.01, 0.1, 1, 10),
                        "xgbclassifier__learning_rate": (0.01, 0.1, 1, 10),
                        "xgbclassifier__max_depth": (25, 75, 150),
                        "xgbclassifier__subsample": (0.2, 0.5, 0.8),
                        "xgbclassifier__colsample_bytree": (0.2, 0.5, 0.8),
                        "xgbclassifier__reg_lambda": (0.01, 0.1, 1, 10),
                        "xgbclassifier__reg_alpha": (0.01, 0.1, 1, 10),
                        "xgbclassifier__gamma": (0.01, 0.1, 1, 10),
                        "xgbclassifier__n_estimators": (25, 75, 150),
                        "xgbclassifier__scale_pos_weight": (y_train_1/y_train_0, y_train_0/y_train_1)}

pipeline = make_pipeline(TomekLinks(), XGBClassifier(booster = "gbtree", random_state=19, use_label_encoder=False, eval_metric="auc", tree_method = "hist"))

xgbc_rscv = RandomizedSearchCV(estimator=pipeline,
                                param_distributions=hyperparameters_xgbc,
                                n_jobs=-1,
                                scoring="roc_auc",
                                verbose=5,
                                cv=5,
                                n_iter=500,
                                random_state=19)

### BASED XGBC
xgbc = xgbc.fit(X_train, y_train)
joblib.dump(xgbc, "model_pickle/MICE1/TSR_ALL1_XGBC_BASED.pkl")

### TUNED XGBC
xgbc_rsCV = xgbc_rscv.fit(X_train, y_train)
joblib.dump(xgbc_rsCV, "model_pickle/MICE1/TSR_ALL1_XGBC_TUNED.pkl")

### CALIBRATED XGBC
xgbc_cccv = CalibratedClassifierCV(base_estimator=xgbc_rsCV.best_estimator_, cv=5)
xgbc_ccCV = xgbc_cccv.fit(X_train, y_train)
joblib.dump(xgbc_ccCV, "model_pickle/MICE1/TSR_ALL1_XGBC_CALIBRATED.pkl")