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

#GOOD when discharged
csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1G_X_TRAIN.csv")
G_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1G_y_TRAIN.csv")
G_y_train = pd.read_csv(csv_path)
G_y_train = np.ravel(G_y_train)

#BAD when discharged
csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1B_X_TRAIN.csv")
B_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1B_y_TRAIN.csv")
B_y_train = pd.read_csv(csv_path)
B_y_train = np.ravel(B_y_train)

# class weight
G_y_train_1 = sum(G_y_train)
G_y_train_0 = len(G_y_train) - sum(G_y_train)

B_y_train_1 = sum(B_y_train)
B_y_train_0 = len(B_y_train) - sum(B_y_train)

# base xgbc
xgbc = XGBClassifier(booster = "gbtree", random_state=19, use_label_encoder=False, eval_metric = "auc", tree_method = "hist", n_jobs=-1)

# tune xgbc
hyperparameters_xgbcG = {"xgbclassifier__learning_rate": (0.01, 0.1, 1, 10),
                        "xgbclassifier__learning_rate": (0.01, 0.1, 1, 10),
                        "xgbclassifier__max_depth": (25, 75, 150),
                        "xgbclassifier__subsample": (0.2, 0.5, 0.8),
                        "xgbclassifier__colsample_bytree": (0.2, 0.5, 0.8),
                        "xgbclassifier__reg_lambda": (0.01, 0.1, 1, 10),
                        "xgbclassifier__reg_alpha": (0.01, 0.1, 1, 10),
                        "xgbclassifier__gamma": (0.01, 0.1, 1, 10),
                        "xgbclassifier__n_estimators": (25, 75, 150),
                        "xgbclassifier__scale_pos_weight": (G_y_train_1/G_y_train_0, G_y_train_0/G_y_train_1)}

hyperparameters_xgbcB = {"xgbclassifier__learning_rate": (0.01, 0.1, 1, 10),
                        "xgbclassifier__learning_rate": (0.01, 0.1, 1, 10),
                        "xgbclassifier__max_depth": (25, 75, 150),
                        "xgbclassifier__subsample": (0.2, 0.5, 0.8),
                        "xgbclassifier__colsample_bytree": (0.2, 0.5, 0.8),
                        "xgbclassifier__reg_lambda": (0.01, 0.1, 1, 10),
                        "xgbclassifier__reg_alpha": (0.01, 0.1, 1, 10),
                        "xgbclassifier__gamma": (0.01, 0.1, 1, 10),
                        "xgbclassifier__n_estimators": (25, 75, 150),
                        "xgbclassifier__scale_pos_weight": (B_y_train_1/B_y_train_0, B_y_train_0/B_y_train_1)}

pipeline = make_pipeline(TomekLinks(), XGBClassifier(booster = "gbtree", random_state=19, use_label_encoder=False, eval_metric="auc", tree_method = "hist"))

xgbcG_rscv = RandomizedSearchCV(estimator=pipeline,
                                param_distributions=hyperparameters_xgbcG,
                                n_jobs=-1,
                                scoring="roc_auc",
                                verbose=5,
                                cv=5,
                                n_iter=500,
                                random_state=19)

xgbcB_rscv = RandomizedSearchCV(estimator=pipeline,
                                param_distributions=hyperparameters_xgbcB,
                                n_jobs=-1,
                                scoring="roc_auc",
                                verbose=5,
                                cv=5,
                                n_iter=500,
                                random_state=19)

#GOOD when discharged
### BASED XGBC
xgbcG = xgbc.fit(G_X_train, G_y_train)
joblib.dump(xgbcG, "model_pickle/MICE1/TSR_ALL1G_XGBC_BASED.pkl")

### TUNED XGBC
xgbcG_rsCV = xgbcG_rscv.fit(G_X_train, G_y_train)
joblib.dump(xgbcG_rsCV, "model_pickle/MICE1/TSR_ALL1G_XGBC_TUNED.pkl")

### CALIBRATED XGBC
xgbcG_cccv = CalibratedClassifierCV(base_estimator=xgbcG_rsCV.best_estimator_, cv=5)
xgbcG_ccCV = xgbcG_cccv.fit(G_X_train, G_y_train)
joblib.dump(xgbcG_ccCV, "model_pickle/MICE1/TSR_ALL1G_XGBC_CALIBRATED.pkl")

#BAD when discharged
### BASED XGBC
xgbcB = xgbc.fit(B_X_train, B_y_train)
joblib.dump(xgbcB, "model_pickle/MICE1/TSR_ALL1B_XGBC_BASED.pkl")

### TUNED XGBC
xgbcB_rsCV = xgbcB_rscv.fit(B_X_train, B_y_train)
joblib.dump(xgbcB_rsCV, "model_pickle/MICE1/TSR_ALL1B_XGBC_TUNED.pkl")

### CALIBRATED XGBC
xgbcB_cccv = CalibratedClassifierCV(base_estimator=xgbcB_rsCV.best_estimator_, cv=5)
xgbcB_ccCV = xgbcB_cccv.fit(B_X_train, B_y_train)
joblib.dump(xgbcB_ccCV, "model_pickle/MICE1/TSR_ALL1B_XGBC_CALIBRATED.pkl")