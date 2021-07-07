import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import TomekLinks
import joblib

# base et
et = ExtraTreesClassifier(random_state=19)

#tune et
hyperparameters_etG = {"extratreesclassifier__n_estimators": (25, 75, 150),
                      "extratreesclassifier__criterion": ("gini", "entropy"),
                      "extratreesclassifier__max_depth": (25, 75, 150),
                      "extratreesclassifier__min_samples_split": (25, 75, 150),
                      "extratreesclassifier__max_features": (0.2, 0.5, 0.8),
                      "extratreesclassifier__bootstrap": (True, False),
                      "extratreesclassifier__class_weight": ('balanced', {0: 1, 1: 32}, {0: 32, 1: 1}),
                      "extratreesclassifier__max_samples": (0.2, 0.5, 0.8)}

hyperparameters_etB = {"extratreesclassifier__n_estimators": (25, 75, 150),
                      "extratreesclassifier__criterion": ("gini", "entropy"),
                      "extratreesclassifier__max_depth": (25, 75, 150),
                      "extratreesclassifier__min_samples_split": (25, 75, 150),
                      "extratreesclassifier__max_features": (0.2, 0.5, 0.8),
                      "extratreesclassifier__bootstrap": (True, False),
                      "extratreesclassifier__class_weight": ('balanced', {0: 1, 1: 7}, {0: 7, 1: 1}),
                      "extratreesclassifier__max_samples": (0.2, 0.5, 0.8)}

pipeline = make_pipeline(TomekLinks(), ExtraTreesClassifier(random_state=19))

etG_rscv = RandomizedSearchCV(estimator=pipeline,
                              param_distributions=hyperparameters_etG,
                              n_jobs=-1,
                              scoring='roc_auc',
                              verbose=5,
                              cv=5,
                              n_iter=500,
                              random_state=19)

etB_rscv = RandomizedSearchCV(estimator=pipeline,
                              param_distributions=hyperparameters_etB,
                              n_jobs=-1,
                              scoring='roc_auc',
                              verbose=5,
                              cv=5,
                              n_iter=500,
                              random_state=19)

#GOOD when discharged
csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1G_X_TRAIN.csv")
G_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1G_y_TRAIN.csv")
G_y_train = pd.read_csv(csv_path)
G_y_train = np.ravel(G_y_train)

### BASED ET
etG = et.fit(G_X_train, G_y_train)
joblib.dump(etG, "model_pickle/TSR_ALL1G_ET_BASED.pkl")

### TUNED ET
etG_rsCV = etG_rscv.fit(G_X_train, G_y_train)
joblib.dump(etG_rsCV, "model_pickle/TSR_ALL1G_ET_TUNED.pkl")

### CALIBRATED ET
etG_cccv = CalibratedClassifierCV(base_estimator=etG_rsCV.best_estimator_, cv=5)
etG_ccCV = etG_cccv.fit(G_X_train, G_y_train)
joblib.dump(etG_ccCV, "model_pickle/TSR_ALL1G_ET_CALIBRATED.pkl")

#BAD when discharged
csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1B_X_TRAIN.csv")
B_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1B_y_TRAIN.csv")
B_y_train = pd.read_csv(csv_path)
B_y_train = np.ravel(B_y_train)

### BASED ET
etB = et.fit(B_X_train, B_y_train)
joblib.dump(etG, "model_pickle/TSR_ALL1B_ET_BASED.pkl")

### TUNED ET
etB_rsCV = etB_rscv.fit(B_X_train, B_y_train)
joblib.dump(etG_rsCV, "model_pickle/TSR_ALL1B_ET_TUNED.pkl")

### CALIBRATED ET
etB_cccv = CalibratedClassifierCV(base_estimator=etB_rsCV.best_estimator_, cv=5)
etB_ccCV = etB_cccv.fit(B_X_train, B_y_train)
joblib.dump(etG_ccCV, "model_pickle/TSR_ALL1B_ET_CALIBRATED.pkl")