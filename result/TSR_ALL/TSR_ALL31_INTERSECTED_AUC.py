import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.metrics import auc, roc_curve
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# Import datasets
csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31G_X_TRAIN.csv")
G_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31G_y_TRAIN.csv")
G_y_train = pd.read_csv(csv_path)
G_y_train = np.ravel(G_y_train)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31G_X_TEST.csv")
G_X_test = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31G_y_TEST.csv")
G_y_test = pd.read_csv(csv_path)
G_y_test = np.ravel(G_y_test)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31B_X_TRAIN.csv")
B_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31B_y_TRAIN.csv")
B_y_train = pd.read_csv(csv_path)
B_y_train = np.ravel(B_y_train)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31B_X_TEST.csv")
B_X_test = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31B_y_TEST.csv")
B_y_test = pd.read_csv(csv_path)
B_y_test = np.ravel(B_y_test)

csv_path = os.path.join("INFO", "TSR_ALL31", "Threshold_Intersection_31.csv")
T_I = pd.read_csv(csv_path)

# GOOD when Discharge
## Selected Columns
G_index = T_I["G31"].dropna().values
G_X_train_selected = G_X_train[G_index]
G_X_test_selected = G_X_test[G_index]

## Algorithms
### Extra trees
pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL31G_ET_TUNED.pkl")
G_ET_TUNED = joblib.load(pkl_path)

# base et_selected
et_selected = ExtraTreesClassifier(random_state=19)
G_ET_BASED_selected = et_selected.fit(G_X_train_selected, G_y_train)

G_y_test_pred = G_ET_BASED_selected.predict_proba(G_X_test_selected)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc_et_based = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc_et_based)

# tune et_selected
et_rscv_selected = G_ET_TUNED.best_estimator_
G_ET_TUNED_selected = et_rscv_selected.fit(G_X_train_selected, G_y_train)

G_y_test_pred = G_ET_TUNED_selected.predict_proba(G_X_test_selected)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc_et_tuned = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc_et_tuned)

# calibrate et_selected
et_cccv_selected = CalibratedClassifierCV(base_estimator=G_ET_TUNED_selected, cv=5)
G_ET_CALIBRATED_selected = et_cccv_selected.fit(G_X_train_selected, G_y_train)

G_y_test_pred = G_ET_CALIBRATED_selected.predict_proba(G_X_test_selected)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc_et_calibrated = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc_et_calibrated)

## XGBClassifier
pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL31G_XGBC_TUNED.pkl")
G_XGBC_TUNED = joblib.load(pkl_path)

# base xgbc_selected
xgbc_selected = XGBClassifier(booster="gbtree", random_state=19, use_label_encoder=False, eval_metric="auc",
                              tree_method="hist", n_jobs=-1)
G_XGBC_BASED_selected = xgbc_selected.fit(G_X_train_selected, G_y_train)

G_y_test_pred = G_XGBC_BASED_selected.predict_proba(G_X_test_selected)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc_xgbc_based = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc_xgbc_based)

# tune xgbc_selected
xgbcG_rscv_selected = G_XGBC_TUNED.best_estimator_
G_XGBC_TUNED_selected = xgbcG_rscv_selected.fit(G_X_train_selected, G_y_train)

G_y_test_pred = G_XGBC_TUNED_selected.predict_proba(G_X_test_selected)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc_xgbc_tuned = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc_xgbc_tuned)

# calibrate xgbc_selected
xgbcG_cccv_selected = CalibratedClassifierCV(base_estimator=G_XGBC_TUNED_selected, cv=5)
G_XGBC_CALIBRATED_selected = xgbcG_cccv_selected.fit(G_X_train_selected, G_y_train)

G_y_test_pred = G_XGBC_CALIBRATED_selected.predict_proba(G_X_test_selected)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc_xgbc_calibrated = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc_xgbc_calibrated)

G_et_list = [G_test_auroc_et_based, G_test_auroc_et_tuned, G_test_auroc_et_calibrated]
G_xgbc_list = [G_test_auroc_xgbc_based, G_test_auroc_xgbc_tuned, G_test_auroc_xgbc_calibrated]
x = "Based","Tuned","Calibrated"

plt.plot(x, G_et_list, label = "ET")
plt.plot(x, G_xgbc_list, label = "XGBC")
plt.legend()
for i in range(len(x)):
    plt.annotate(round(G_et_list[i],3), (x[i], G_et_list[i]))
    plt.annotate(round(G_xgbc_list[i],3), (x[i], G_xgbc_list[i]))
plt.title('TSR_ALL31_G_Intersection', fontsize=15)
plt.savefig('PLOT/TSR_ALL31/TSR_ALL31_G_Intersection.png')
plt.show()

# BAD when Discharge
## Selected Columns
B_index = T_I["B31"].dropna().values
B_X_train_selected = B_X_train[B_index]
B_X_test_selected = B_X_test[B_index]

## Algorithms
### Extra trees
pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL31B_ET_TUNED.pkl")
B_ET_TUNED = joblib.load(pkl_path)

# base et_selected
et_selected = ExtraTreesClassifier(random_state=19)
B_ET_BASED_selected = et_selected.fit(B_X_train_selected, B_y_train)

B_y_test_pred = B_ET_BASED_selected.predict_proba(B_X_test_selected)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc_et_based = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc_et_based)

# tune et_selected
et_rscv_selected = B_ET_TUNED.best_estimator_
B_ET_TUNED_selected = et_rscv_selected.fit(B_X_train_selected, B_y_train)

B_y_test_pred = B_ET_TUNED_selected.predict_proba(B_X_test_selected)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc_et_tuned = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc_et_tuned)

# calibrate et_selected
et_cccv_selected = CalibratedClassifierCV(base_estimator=B_ET_TUNED_selected, cv=5)
B_ET_CALIBRATED_selected = et_cccv_selected.fit(B_X_train_selected, B_y_train)

B_y_test_pred = B_ET_CALIBRATED_selected.predict_proba(B_X_test_selected)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc_et_calibrated = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc_et_calibrated)

## XGBClassifier
pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL31B_XGBC_TUNED.pkl")
B_XGBC_TUNED = joblib.load(pkl_path)

# base xgbc_selected
xgbc_selected = XGBClassifier(booster="gbtree", random_state=19, use_label_encoder=False, eval_metric="auc",
                              tree_method="hist", n_jobs=-1)
B_XGBC_BASED_selected = xgbc_selected.fit(B_X_train_selected, B_y_train)

B_y_test_pred = B_XGBC_BASED_selected.predict_proba(B_X_test_selected)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc_xgbc_based = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc_xgbc_based)

# tune xgbc_selected
xgbcB_rscv_selected = B_XGBC_TUNED.best_estimator_
B_XGBC_TUNED_selected = xgbcB_rscv_selected.fit(B_X_train_selected, B_y_train)

B_y_test_pred = B_XGBC_TUNED_selected.predict_proba(B_X_test_selected)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc_xgbc_tuned = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc_xgbc_tuned)

# calibrate xgbc_selected
xgbcB_cccv_selected = CalibratedClassifierCV(base_estimator=B_XGBC_TUNED_selected, cv=5)
B_XGBC_CALIBRATED_selected = xgbcB_cccv_selected.fit(B_X_train_selected, B_y_train)

B_y_test_pred = B_XGBC_CALIBRATED_selected.predict_proba(B_X_test_selected)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc_xgbc_calibrated = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc_xgbc_calibrated)

B_et_list = [B_test_auroc_et_based, B_test_auroc_et_tuned, B_test_auroc_et_calibrated]
B_xgbc_list = [B_test_auroc_xgbc_based, B_test_auroc_xgbc_tuned, B_test_auroc_xgbc_calibrated]
x = "Based","Tuned","Calibrated"

plt.plot(x, B_et_list, label = "ET")
plt.plot(x, B_xgbc_list, label = "XGBC")
plt.legend()
for i in range(len(x)):
    plt.annotate(round(B_et_list[i],3), (x[i], B_et_list[i]))
    plt.annotate(round(B_xgbc_list[i],3), (x[i], B_xgbc_list[i]))
plt.title('TSR_ALL31_B_Intersection', fontsize=15)
plt.savefig('PLOT/TSR_ALL31/TSR_ALL31_B_Intersection.png')
plt.show()
