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

def intersected_auc(X_train, X_test, y_train, y_test, intersection_df, selected_columns, based_model, tuned_model):
    # Selected Columns
    selected_index = intersection_df[selected_columns].dropna().values
    X_train_selected = X_train[selected_index]
    X_test_selected = X_test[selected_index]

    # based model_selected
    BASED_selected = based_model.fit(X_train_selected, y_train)

    y_test_pred = BASED_selected.predict_proba(X_test_selected)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
    test_auroc_based = auc(fpr, tpr)
    print('AUC of testing set:', test_auroc_based)

    # tuned model_selected
    rscv_selected = tuned_model.best_estimator_
    TUNED_selected = rscv_selected.fit(X_train_selected, y_train)

    y_test_pred = TUNED_selected.predict_proba(X_test_selected)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
    test_auroc_tuned = auc(fpr, tpr)
    print('AUC of testing set:', test_auroc_tuned)

    # calibrated model_selected
    cccv_selected = CalibratedClassifierCV(base_estimator=TUNED_selected, cv=5)
    CALIBRATED_selected = cccv_selected.fit(X_train_selected, y_train)

    y_test_pred = CALIBRATED_selected.predict_proba(X_test_selected)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
    test_auroc_calibrated = auc(fpr, tpr)
    print('AUC of testing set:', test_auroc_calibrated)

    auc_list = [test_auroc_based, test_auroc_tuned, test_auroc_calibrated]

    return auc_list


if __name__ == '__main__':
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

    ## Based models
    based_et = ExtraTreesClassifier(random_state=19)
    based_xgbc = XGBClassifier(booster="gbtree", random_state=19, use_label_encoder=False, eval_metric="auc",
                               tree_method="hist", n_jobs=-1)

    # GOOD when discharge
    ### ET
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL31G_ET_TUNED.pkl")
    G_ET_TUNED = joblib.load(pkl_path)
    G_et_list = intersected_auc(G_X_train, G_X_test, G_y_train, G_y_test, T_I, "G31", based_et, G_ET_TUNED)

    ### XGBC
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL31G_XGBC_TUNED.pkl")
    G_XGBC_TUNED = joblib.load(pkl_path)
    G_xgbc_list = intersected_auc(G_X_train, G_X_test, G_y_train, G_y_test, T_I, "G31", based_xgbc, G_XGBC_TUNED)

    ### plot
    x = "Based", "Tuned", "Calibrated"
    plt.plot(x, G_et_list, label="ET")
    plt.plot(x, G_xgbc_list, label="XGBC")
    plt.legend()
    for i in range(len(x)):
        plt.annotate(round(G_et_list[i], 3), (x[i], G_et_list[i]))
        plt.annotate(round(G_xgbc_list[i], 3), (x[i], G_xgbc_list[i]))
    plt.title('TSR_ALL31_G_Intersection', fontsize=15)
    # plt.savefig('PLOT/TSR_ALL31/TSR_ALL31_G_Intersection.png')
    plt.show()

    # BAD when discharge
    ### ET
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL31B_ET_TUNED.pkl")
    B_ET_TUNED = joblib.load(pkl_path)
    B_et_list = intersected_auc(B_X_train, B_X_test, B_y_train, B_y_test, T_I, "B31", based_et, B_ET_TUNED)

    ### XGBC
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL31B_XGBC_TUNED.pkl")
    B_XGBC_TUNED = joblib.load(pkl_path)
    B_xgbc_list = intersected_auc(B_X_train, B_X_test, B_y_train, B_y_test, T_I, "B31", based_xgbc, B_XGBC_TUNED)

    ### plot
    x = "Based", "Tuned", "Calibrated"
    plt.plot(x, B_et_list, label="ET")
    plt.plot(x, B_xgbc_list, label="XGBC")
    plt.legend()
    for i in range(len(x)):
        plt.annotate(round(B_et_list[i], 3), (x[i], B_et_list[i]))
        plt.annotate(round(B_xgbc_list[i], 3), (x[i], B_xgbc_list[i]))
    plt.title('TSR_ALL31_B_Intersection', fontsize=15)
    # plt.savefig('PLOT/TSR_ALL31/TSR_ALL31_B_Intersection.png')
    plt.show()