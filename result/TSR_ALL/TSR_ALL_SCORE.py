import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

def algorithms(X_train, X_test, y_train, y_test, tuned, calibrated):
    # CALIBRATED
    y_test_pred = calibrated.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
    auc_list = []
    for i in thresholds:
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred[:, 1] > i).ravel()
        auc1 = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
        auc_list.append(auc1)
    number = auc_list.index(max(auc_list))
    threshold_number = thresholds[number]
    print('confusion_matrix of test set:', confusion_matrix(y_test, y_test_pred[:, 1] > threshold_number))
    print('roc_auc_score of test set:', round(roc_auc_score(y_test, y_test_pred[:, 1] > threshold_number), 3))
    print('accuracy_score of test set:', round(accuracy_score(y_test, y_test_pred[:, 1] > threshold_number), 3))
    print('precision_score of test set:', round(precision_score(y_test, y_test_pred[:, 1] > threshold_number), 3))
    print('recall_score of test set:', round(recall_score(y_test, y_test_pred[:, 1] > threshold_number), 3))

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
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
        auc_list = []
        for i in thresholds:
            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred[:, 1] > i).ravel()
            auc1 = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
            auc_list.append(auc1)
        number = auc_list.index(max(auc_list))
        threshold_number = thresholds[number]
        print('confusion_matrix of test set:', confusion_matrix(y_test, y_test_pred[:, 1] > threshold_number))
        print('roc_auc_score of test set:', round(roc_auc_score(y_test, y_test_pred[:, 1] > threshold_number), 3))
        print('accuracy_score of test set:', round(accuracy_score(y_test, y_test_pred[:, 1] > threshold_number), 3))
        print('precision_score of test set:', round(precision_score(y_test, y_test_pred[:, 1] > threshold_number), 3))
        print('recall_score of test set:', round(recall_score(y_test, y_test_pred[:, 1] > threshold_number), 3))

    #return fpr_list_cc_list, tpr_list_cc_list

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

print("G1")
algorithms(G1_X_train, G1_X_test, G1_y_train, G1_y_test, G1_XGBC_TUNED, G1_XGBC_CALIBRATED)

### Logistic Regression
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL1G_LR_CALIBRATED.pkl")
G1_LR_CALIBRATED = joblib.load(pkl_path)

# CALIBRATED
y_test_pred = G1_LR_CALIBRATED.predict_proba(G1_X_test)
fpr, tpr, thresholds = roc_curve(G1_y_test, y_test_pred[:, 1])
auc_list = []
for i in thresholds:
    tn, fp, fn, tp = confusion_matrix(G1_y_test, y_test_pred[:, 1] > i).ravel()
    auc1 = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
    auc_list.append(auc1)
number = auc_list.index(max(auc_list))
threshold_number = thresholds[number]
print('confusion_matrix of test set:', confusion_matrix(G1_y_test, y_test_pred[:, 1] > threshold_number))
print('roc_auc_score of test set:', round(roc_auc_score(G1_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('accuracy_score of test set:', round(accuracy_score(G1_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('precision_score of test set:', round(precision_score(G1_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('recall_score of test set:', round(recall_score(G1_y_test, y_test_pred[:, 1] > threshold_number), 3))

#B1
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL1B_XGBC_TUNED.pkl")
B1_XGBC_TUNED = joblib.load(pkl_path)

pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL1B_XGBC_CALIBRATED.pkl")
B1_XGBC_CALIBRATED = joblib.load(pkl_path)

print("B1")
algorithms(B1_X_train, B1_X_test, B1_y_train, B1_y_test, B1_XGBC_TUNED, B1_XGBC_CALIBRATED)

### Logistic Regression
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL1B_LR_CALIBRATED.pkl")
B1_LR_CALIBRATED = joblib.load(pkl_path)

# CALIBRATED
y_test_pred = B1_LR_CALIBRATED.predict_proba(B1_X_test)
fpr, tpr, thresholds = roc_curve(B1_y_test, y_test_pred[:, 1])
auc_list = []
for i in thresholds:
    tn, fp, fn, tp = confusion_matrix(B1_y_test, y_test_pred[:, 1] > i).ravel()
    auc1 = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
    auc_list.append(auc1)
number = auc_list.index(max(auc_list))
threshold_number = thresholds[number]
print('confusion_matrix of test set:', confusion_matrix(B1_y_test, y_test_pred[:, 1] > threshold_number))
print('roc_auc_score of test set:', round(roc_auc_score(B1_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('accuracy_score of test set:', round(accuracy_score(B1_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('precision_score of test set:', round(precision_score(B1_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('recall_score of test set:', round(recall_score(B1_y_test, y_test_pred[:, 1] > threshold_number), 3))

#G31
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE4", "TSR_ALL31G_XGBC_TUNED.pkl")
G31_XGBC_TUNED = joblib.load(pkl_path)

pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE4", "TSR_ALL31G_XGBC_CALIBRATED.pkl")
G31_XGBC_CALIBRATED = joblib.load(pkl_path)

print("G31")
algorithms(G31_X_train, G31_X_test, G31_y_train, G31_y_test, G31_XGBC_TUNED, G31_XGBC_CALIBRATED)

### Logistic Regression
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE4", "TSR_ALL31G_LR_CALIBRATED.pkl")
G31_LR_CALIBRATED = joblib.load(pkl_path)

# CALIBRATED
y_test_pred = G31_LR_CALIBRATED.predict_proba(G31_X_test)
fpr, tpr, thresholds = roc_curve(G31_y_test, y_test_pred[:, 1])
auc_list = []
for i in thresholds:
    tn, fp, fn, tp = confusion_matrix(G31_y_test, y_test_pred[:, 1] > i).ravel()
    auc1 = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
    auc_list.append(auc1)
number = auc_list.index(max(auc_list))
threshold_number = thresholds[number]
print('confusion_matrix of test set:', confusion_matrix(G31_y_test, y_test_pred[:, 1] > threshold_number))
print('roc_auc_score of test set:', round(roc_auc_score(G31_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('accuracy_score of test set:', round(accuracy_score(G31_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('precision_score of test set:', round(precision_score(G31_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('recall_score of test set:', round(recall_score(G31_y_test, y_test_pred[:, 1] > threshold_number), 3))

#B31
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL31B_XGBC_TUNED.pkl")
B31_XGBC_TUNED = joblib.load(pkl_path)

pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL31B_XGBC_CALIBRATED.pkl")
B31_XGBC_CALIBRATED = joblib.load(pkl_path)

print("B31")
algorithms(B31_X_train, B31_X_test, B31_y_train, B31_y_test, B31_XGBC_TUNED, B31_XGBC_CALIBRATED)

### Logistic Regression
pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE1", "TSR_ALL31B_LR_CALIBRATED.pkl")
B31_LR_CALIBRATED = joblib.load(pkl_path)

# CALIBRATED
y_test_pred = B31_LR_CALIBRATED.predict_proba(B31_X_test)
fpr, tpr, thresholds = roc_curve(B31_y_test, y_test_pred[:, 1])
auc_list = []
for i in thresholds:
    tn, fp, fn, tp = confusion_matrix(B31_y_test, y_test_pred[:, 1] > i).ravel()
    auc1 = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
    auc_list.append(auc1)
number = auc_list.index(max(auc_list))
threshold_number = thresholds[number]
print('confusion_matrix of test set:', confusion_matrix(B31_y_test, y_test_pred[:, 1] > threshold_number))
print('roc_auc_score of test set:', round(roc_auc_score(B31_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('accuracy_score of test set:', round(accuracy_score(B31_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('precision_score of test set:', round(precision_score(B31_y_test, y_test_pred[:, 1] > threshold_number), 3))
print('recall_score of test set:', round(recall_score(B31_y_test, y_test_pred[:, 1] > threshold_number), 3))