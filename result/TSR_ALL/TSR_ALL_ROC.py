import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.metrics import roc_curve
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


def algorithms(X_train, X_test, y_train, y_test, tuned, calibrated):
    # CALIBRATED
    y_test_pred = calibrated.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])

    #### Selected Columns
    model_fi = calibrated.base_estimator._final_estimator.feature_importances_
    model_fi_df = pd.DataFrame(model_fi)
    model_fi_df.index = X_test.columns
    model_fi_df.columns = (["Value"])
    model_fi_df = model_fi_df.reset_index(drop=False)
    model_fi_df.columns = (["Feature", "Value"])
    model_fi_df = model_fi_df.sort_values(["Value"], ascending=False)

    ### sigma threshold
    model_fi_df_noZERO = model_fi_df[~model_fi_df.Value.isin([0])]
    model_fi_df_noZERO_mean = model_fi_df_noZERO.Value.mean()
    model_fi_df_noZERO_std = model_fi_df_noZERO.Value.std()
    sigma_n = len(model_fi_df_noZERO[model_fi_df_noZERO.Value > model_fi_df_noZERO_mean + model_fi_df_noZERO_std])
    # print(sigma_n)

    fpr_list_cc_list = []
    tpr_list_cc_list = []

    for i in sigma_n, 10, 20, 30:
        model_fi_index = model_fi_df[0:i].index

        X_train_selected = X_train.iloc[:, model_fi_index]
        X_test_selected = X_test.iloc[:, model_fi_index]

        # tune et_selected
        rscv_selected = tuned.best_estimator_
        TUNED_selected = rscv_selected.fit(X_train_selected, y_train)

        # calibrate et_selected
        cccv_selected = CalibratedClassifierCV(base_estimator=TUNED_selected, cv=5)
        CALIBRATED_selected = cccv_selected.fit(X_train_selected, y_train)

        y_test_pred = CALIBRATED_selected.predict_proba(X_test_selected)
        fpr_s, tpr_s, thresholds = roc_curve(y_test, y_test_pred[:, 1])
        fpr_list_cc_list.append(fpr_s)
        tpr_list_cc_list.append(tpr_s)

    fpr_list_cc_list.append(fpr)
    tpr_list_cc_list.append(tpr)

    return fpr_list_cc_list, tpr_list_cc_list
# import data
csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1G_X_TRAIN.csv")
G1_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1G_y_TRAIN.csv")
G1_y_train = pd.read_csv(csv_path)
G1_y_train = np.ravel(G1_y_train)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1G_X_TEST.csv")
G1_X_test = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1G_y_TEST.csv")
G1_y_test = pd.read_csv(csv_path)
G1_y_test = np.ravel(G1_y_test)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1B_X_TRAIN.csv")
B1_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1B_y_TRAIN.csv")
B1_y_train = pd.read_csv(csv_path)
B1_y_train = np.ravel(B1_y_train)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1B_X_TEST.csv")
B1_X_test = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1B_y_TEST.csv")
B1_y_test = pd.read_csv(csv_path)
B1_y_test = np.ravel(B1_y_test)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31G_X_TRAIN.csv")
G31_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31G_y_TRAIN.csv")
G31_y_train = pd.read_csv(csv_path)
G31_y_train = np.ravel(G31_y_train)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31G_X_TEST.csv")
G31_X_test = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31G_y_TEST.csv")
G31_y_test = pd.read_csv(csv_path)
G31_y_test = np.ravel(G31_y_test)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31B_X_TRAIN.csv")
B31_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31B_y_TRAIN.csv")
B31_y_train = pd.read_csv(csv_path)
B31_y_train = np.ravel(B31_y_train)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31B_X_TEST.csv")
B31_X_test = pd.read_csv(csv_path)

csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL31", "TSR_ALL31B_y_TEST.csv")
B31_y_test = pd.read_csv(csv_path)
B31_y_test = np.ravel(B31_y_test)

#G1
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL1G_XGBC_TUNED.pkl")
G1_XGBC_TUNED = joblib.load(pkl_path)

pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL1G_XGBC_CALIBRATED.pkl")
G1_XGBC_CALIBRATED = joblib.load(pkl_path)

G1_fpr_list_cc_list, G1_tpr_list_cc_list = algorithms(G1_X_train, G1_X_test, G1_y_train, G1_y_test, G1_XGBC_TUNED,
                                                      G1_XGBC_CALIBRATED)

### Logistic Regression
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL1G_LR_CALIBRATED.pkl")
G1_LR_CALIBRATED = joblib.load(pkl_path)

# CALIBRATED
y_test_pred = G1_LR_CALIBRATED.predict_proba(G1_X_test)
fpr_G1, tpr_G1, thresholds = roc_curve(G1_y_test, y_test_pred[:, 1])

#B1
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL1B_XGBC_TUNED.pkl")
B1_XGBC_TUNED = joblib.load(pkl_path)

pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL1B_XGBC_CALIBRATED.pkl")
B1_XGBC_CALIBRATED = joblib.load(pkl_path)

B1_fpr_list_cc_list, B1_tpr_list_cc_list = algorithms(B1_X_train, B1_X_test, B1_y_train, B1_y_test, B1_XGBC_TUNED,
                                                      B1_XGBC_CALIBRATED)

### Logistic Regression
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL1B_LR_CALIBRATED.pkl")
B1_LR_CALIBRATED = joblib.load(pkl_path)

# CALIBRATED
y_test_pred = B1_LR_CALIBRATED.predict_proba(B1_X_test)
fpr_B1, tpr_B1, thresholds = roc_curve(B1_y_test, y_test_pred[:, 1])

#G31
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE4", "TSR_ALL31G_XGBC_TUNED.pkl")
G31_XGBC_TUNED = joblib.load(pkl_path)

pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE4", "TSR_ALL31G_XGBC_CALIBRATED.pkl")
G31_XGBC_CALIBRATED = joblib.load(pkl_path)

G31_fpr_list_cc_list, G31_tpr_list_cc_list = algorithms(G31_X_train, G31_X_test, G31_y_train, G31_y_test,
                                                        G31_XGBC_TUNED, G31_XGBC_CALIBRATED)

### Logistic Regression
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE4", "TSR_ALL31G_LR_CALIBRATED.pkl")
G31_LR_CALIBRATED = joblib.load(pkl_path)

# CALIBRATED
y_test_pred = G31_LR_CALIBRATED.predict_proba(G31_X_test)
fpr_G31, tpr_G31, thresholds = roc_curve(G31_y_test, y_test_pred[:, 1])

#B31
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL31B_XGBC_TUNED.pkl")
B31_XGBC_TUNED = joblib.load(pkl_path)

pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL31B_XGBC_CALIBRATED.pkl")
B31_XGBC_CALIBRATED = joblib.load(pkl_path)

B31_fpr_list_cc_list, B31_tpr_list_cc_list = algorithms(B31_X_train, B31_X_test, B31_y_train, B31_y_test,
                                                        B31_XGBC_TUNED, B31_XGBC_CALIBRATED)

### Logistic Regression
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL31B_LR_CALIBRATED.pkl")
B31_LR_CALIBRATED = joblib.load(pkl_path)

# CALIBRATED
y_test_pred = B31_LR_CALIBRATED.predict_proba(B31_X_test)
fpr_B31, tpr_B31, thresholds = roc_curve(B31_y_test, y_test_pred[:, 1])

#plot roc curve
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
ax[0, 0].plot(G1_fpr_list_cc_list[0], G1_tpr_list_cc_list[0], lw=1, label='XGBC-threshold_min* (AUROC = %0.3f)' % 0.678)
ax[0, 0].plot(G1_fpr_list_cc_list[1], G1_tpr_list_cc_list[1], lw=1, label='XGBC-10 (AUROC = %0.3f)' % 0.665)
ax[0, 0].plot(G1_fpr_list_cc_list[2], G1_tpr_list_cc_list[2], lw=1, label='XGBC-20 (AUROC = %0.3f)' % 0.713)
ax[0, 0].plot(G1_fpr_list_cc_list[3], G1_tpr_list_cc_list[3], lw=1, label='XGBC-30 (AUROC = %0.3f)' % 0.719)
ax[0, 0].plot(G1_fpr_list_cc_list[4], G1_tpr_list_cc_list[4], lw=1, label='XGBC-ALL* (AUROC = %0.3f)' % 0.727)
ax[0, 0].plot(fpr_G1, tpr_G1, lw=1, label='LR-ALL* (AUROC = %0.3f)' % 0.708)
ax[0, 0].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
ax[0, 0].axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
ax[0, 0].set_xlabel('False Positive Rate')
ax[0, 0].set_ylabel('True Positive Rate')
ax[0, 0].set_title('ROC Curve of 1-month GOAD')
ax[0, 0].legend(loc="lower right")

ax[0, 1].plot(B1_fpr_list_cc_list[0], B1_tpr_list_cc_list[0], lw=1, label='XGBC-threshold_min* (AUROC = %0.3f)' % 0.816)
ax[0, 1].plot(B1_fpr_list_cc_list[1], B1_tpr_list_cc_list[1], lw=1, label='XGBC-10 (AUROC = %0.3f)' % 0.822)
ax[0, 1].plot(B1_fpr_list_cc_list[2], B1_tpr_list_cc_list[2], lw=1, label='XGBC-20 (AUROC = %0.3f)' % 0.830)
ax[0, 1].plot(B1_fpr_list_cc_list[3], B1_tpr_list_cc_list[3], lw=1, label='XGBC-30 (AUROC = %0.3f)' % 0.836)
ax[0, 1].plot(B1_fpr_list_cc_list[4], B1_tpr_list_cc_list[4], lw=1, label='XGBC-ALL* (AUROC = %0.3f)' % 0.841)
ax[0, 1].plot(fpr_B1, tpr_B1, lw=1, label='LR-ALL* (AUROC = %0.3f)' % 0.829)
ax[0, 1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
ax[0, 1].axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
ax[0, 1].set_xlabel('False Positive Rate')
ax[0, 1].set_ylabel('True Positive Rate')
ax[0, 1].set_title('ROC Curve of 1-month POAD')
ax[0, 1].legend(loc="lower right")

ax[1, 0].plot(G31_fpr_list_cc_list[0], G31_tpr_list_cc_list[0], lw=1, label='XGBC-threshold_min* (AUROC = %0.3f)' % 0.815)
ax[1, 0].plot(G31_fpr_list_cc_list[1], G31_tpr_list_cc_list[1], lw=1, label='XGBC-10 (AUROC = %0.3f)' % 0.817)
ax[1, 0].plot(G31_fpr_list_cc_list[2], G31_tpr_list_cc_list[2], lw=1, label='XGBC-20 (AUROC = %0.3f)' % 0.853)
ax[1, 0].plot(G31_fpr_list_cc_list[3], G31_tpr_list_cc_list[3], lw=1, label='XGBC-30 (AUROC = %0.3f)' % 0.859)
ax[1, 0].plot(G31_fpr_list_cc_list[4], G31_tpr_list_cc_list[4], lw=1, label='XGBC-ALL* (AUROC = %0.3f)' % 0.857)
ax[1, 0].plot(fpr_G31, tpr_G31, lw=1, label='LR-ALL* (AUROC = %0.3f)' % 0.825)
ax[1, 0].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
ax[1, 0].axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
ax[1, 0].set_xlabel('False Positive Rate')
ax[1, 0].set_ylabel('True Positive Rate')
ax[1, 0].set_title('ROC Curve of 3-month GOAD')
ax[1, 0].legend(loc="lower right")

ax[1, 1].plot(B31_fpr_list_cc_list[0], B31_tpr_list_cc_list[0], lw=1, label='XGBC-threshold_min* (AUROC = %0.3f)' % 0.919)
ax[1, 1].plot(B31_fpr_list_cc_list[1], B31_tpr_list_cc_list[1], lw=1, label='XGBC-10 (AUROC = %0.3f)' % 0.923)
ax[1, 1].plot(B31_fpr_list_cc_list[2], B31_tpr_list_cc_list[2], lw=1, label='XGBC-20 (AUROC = %0.3f)' % 0.929)
ax[1, 1].plot(B31_fpr_list_cc_list[3], B31_tpr_list_cc_list[3], lw=1, label='XGBC-30 (AUROC = %0.3f)' % 0.928)
ax[1, 1].plot(B31_fpr_list_cc_list[4], B31_tpr_list_cc_list[4], lw=1, label='XGBC-ALL* (AUROC = %0.3f)' % 0.929)
ax[1, 1].plot(fpr_B31, tpr_B31, lw=1, label='LR-ALL* (AUROC = %0.3f)' % 0.918)
ax[1, 1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
ax[1, 1].axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
ax[1, 1].set_xlabel('False Positive Rate')
ax[1, 1].set_ylabel('True Positive Rate')
ax[1, 1].set_title('ROC Curve of 3-month POAD')
ax[1, 1].legend(loc="lower right")
plt.savefig('PLOT/TSR_ALL_ROC.png', dpi=300, bbox_inches="tight")
plt.show()