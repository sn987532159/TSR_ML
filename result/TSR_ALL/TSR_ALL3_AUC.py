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

def algorithms(X_train, X_test, y_train, y_test, based, tuned, calibrated, model_selected):
    # BASED
    y_test_pred = based.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
    test_auroc = auc(fpr, tpr)
    print('AUC of testing set:', test_auroc)

    # TUNED
    print('--> Tuned Parameters Best Score: ', tuned.best_score_)
    print('--> Best Parameters: \n', tuned.best_params_)

    y_test_pred = tuned.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
    test_auroc_tuned = auc(fpr, tpr)
    print('AUC of testing set:', test_auroc_tuned)

    # CALIBRATED
    y_test_pred = calibrated.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
    test_auroc_cc = auc(fpr, tpr)
    print('AUC of testing set:', test_auroc_cc)

    #### Selected Columns

    model_fi = calibrated.base_estimator._final_estimator.feature_importances_
    model_fi_df = pd.DataFrame(model_fi)
    model_fi_df.index = X_test.columns
    model_fi_df.columns = (["Value"])
    model_fi_df = model_fi_df.reset_index(drop=False)
    model_fi_df.columns = (["Feature", "Value"])
    model_fi_df = model_fi_df.sort_values(["Value"], ascending=False)

    test_auroc_list = []
    test_auroc_tuned_list = []
    test_auroc_cc_list = []

    for i in 10, 20, 30:
        model_fi_index = model_fi_df[0:i].index

        X_train_selected = X_train.iloc[:, model_fi_index]
        X_test_selected = X_test.iloc[:, model_fi_index]

        # base et_selected
        BASED_selected = model_selected.fit(X_train_selected, y_train)

        y_test_pred = BASED_selected.predict_proba(X_test_selected)
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
        test_auroc_selected = auc(fpr, tpr)
        test_auroc_list.append(test_auroc_selected)
        print('AUC of testing set:', test_auroc_selected)

        # tune et_selected
        rscv_selected = tuned.best_estimator_
        TUNED_selected = rscv_selected.fit(X_train_selected, y_train)

        y_test_pred = TUNED_selected.predict_proba(X_test_selected)
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
        test_auroc_selected_tuned = auc(fpr, tpr)
        test_auroc_tuned_list.append(test_auroc_selected_tuned)
        print('AUC of testing set:', test_auroc_selected_tuned)

        # calibrate et_selected
        cccv_selected = CalibratedClassifierCV(base_estimator=TUNED_selected, cv=5)
        CALIBRATED_selected = cccv_selected.fit(X_train_selected, y_train)

        y_test_pred = CALIBRATED_selected.predict_proba(X_test_selected)
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred[:, 1])
        test_auroc_selected_cc = auc(fpr, tpr)
        test_auroc_cc_list.append(test_auroc_selected_cc)
        print('AUC of testing set:', test_auroc_selected_cc)

    test_auroc_list.append(test_auroc)
    test_auroc_tuned_list.append(test_auroc_tuned)
    test_auroc_cc_list.append(test_auroc_cc)

    return test_auroc_list, test_auroc_tuned_list, test_auroc_cc_list, CALIBRATED_selected

if __name__ == '__main__':
    # Import datasets
    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_X_TRAIN.csv")
    G_X_train = pd.read_csv(csv_path)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_y_TRAIN.csv")
    G_y_train = pd.read_csv(csv_path)
    G_y_train = np.ravel(G_y_train)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_X_TEST.csv")
    G_X_test = pd.read_csv(csv_path)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_y_TEST.csv")
    G_y_test = pd.read_csv(csv_path)
    G_y_test = np.ravel(G_y_test)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_X_TRAIN.csv")
    B_X_train = pd.read_csv(csv_path)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_y_TRAIN.csv")
    B_y_train = pd.read_csv(csv_path)
    B_y_train = np.ravel(B_y_train)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_X_TEST.csv")
    B_X_test = pd.read_csv(csv_path)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_y_TEST.csv")
    B_y_test = pd.read_csv(csv_path)
    B_y_test = np.ravel(B_y_test)

    ## Based models
    et_selected = ExtraTreesClassifier(random_state=19)
    xgbc_selected = XGBClassifier(booster="gbtree", random_state=19, use_label_encoder=False, eval_metric="auc",
                                  tree_method="hist", n_jobs=-1)

    # GOOD when Discharge
    ### Extra trees
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3G_ET_BASED.pkl")
    G_ET_BASED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3G_ET_TUNED.pkl")
    G_ET_TUNED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3G_ET_CALIBRATED.pkl")
    G_ET_CALIBRATED = joblib.load(pkl_path)

    G_test_auroc_list, G_test_auroc_tuned_list, G_test_auroc_cc_list, G_ET_CALIBRATED_selected = algorithms(G_X_train,
                                                                                                            G_X_test,
                                                                                                            G_y_train,
                                                                                                            G_y_test,
                                                                                                            G_ET_BASED,
                                                                                                            G_ET_TUNED,
                                                                                                            G_ET_CALIBRATED,
                                                                                                            et_selected)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3G_ET_CALIBRATED_selected.pkl")
    joblib.dump(G_ET_CALIBRATED_selected, pkl_path)

    x = "10", "20", "30", "310"
    plt.plot(x, G_test_auroc_list, label="based")
    plt.plot(x, G_test_auroc_tuned_list, label="tuned")
    plt.plot(x, G_test_auroc_cc_list, label="calibrated")
    plt.legend()
    for i in range(len(x)):
        plt.annotate(round(G_test_auroc_list[i], 3), (x[i], G_test_auroc_list[i]))
        plt.annotate(round(G_test_auroc_tuned_list[i], 3), (x[i], G_test_auroc_tuned_list[i]))
        plt.annotate(round(G_test_auroc_cc_list[i], 3), (x[i], G_test_auroc_cc_list[i]))
    plt.title('TSR_ALL3_ET_G', fontsize=15)
    # plt.savefig('PLOT/TSR_ALL3/TSR_ALL3_ET_G.png')
    plt.show()

    ### XGBClassifier
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3G_XGBC_BASED.pkl")
    G_XGBC_BASED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3G_XGBC_TUNED.pkl")
    G_XGBC_TUNED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3G_XGBC_CALIBRATED.pkl")
    G_XGBC_CALIBRATED = joblib.load(pkl_path)

    G_test_auroc_list, G_test_auroc_tuned_list, G_test_auroc_cc_list, G_XGBC_CALIBRATED_selected = algorithms(G_X_train,
                                                                                                              G_X_test,
                                                                                                              G_y_train,
                                                                                                              G_y_test,
                                                                                                              G_XGBC_BASED,
                                                                                                              G_XGBC_TUNED,
                                                                                                              G_XGBC_CALIBRATED,
                                                                                                              xgbc_selected)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3G_XGBC_CALIBRATED_selected.pkl")
    joblib.dump(G_XGBC_CALIBRATED_selected, pkl_path)

    x = "10", "20", "30", "310"
    plt.plot(x, G_test_auroc_list, label="based")
    plt.plot(x, G_test_auroc_tuned_list, label="tuned")
    plt.plot(x, G_test_auroc_cc_list, label="calibrated")
    plt.legend()
    for i in range(len(x)):
        plt.annotate(round(G_test_auroc_list[i], 3), (x[i], G_test_auroc_list[i]))
        plt.annotate(round(G_test_auroc_tuned_list[i], 3), (x[i], G_test_auroc_tuned_list[i]))
        plt.annotate(round(G_test_auroc_cc_list[i], 3), (x[i], G_test_auroc_cc_list[i]))
    plt.title('TSR_ALL3_XGBC_G', fontsize=15)
    # plt.savefig('PLOT/TSR_ALL3/TSR_ALL3_XGBC_G.png')
    plt.show()

    # BAD when Discharge
    ### Extra trees
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3B_ET_BASED.pkl")
    B_ET_BASED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3B_ET_TUNED.pkl")
    B_ET_TUNED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3B_ET_CALIBRATED.pkl")
    B_ET_CALIBRATED = joblib.load(pkl_path)

    B_test_auroc_list, B_test_auroc_tuned_list, B_test_auroc_cc_list, B_ET_CALIBRATED_selected = algorithms(B_X_train,
                                                                                                            B_X_test,
                                                                                                            B_y_train,
                                                                                                            B_y_test,
                                                                                                            B_ET_BASED,
                                                                                                            B_ET_TUNED,
                                                                                                            B_ET_CALIBRATED,
                                                                                                            et_selected)
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3B_ET_CALIBRATED_selected.pkl")
    joblib.dump(B_ET_CALIBRATED_selected, pkl_path)

    x = "10", "20", "30", "310"
    plt.plot(x, B_test_auroc_list, label="based")
    plt.plot(x, B_test_auroc_tuned_list, label="tuned")
    plt.plot(x, B_test_auroc_cc_list, label="calibrated")
    plt.legend()
    for i in range(len(x)):
        plt.annotate(round(B_test_auroc_list[i], 3), (x[i], B_test_auroc_list[i]))
        plt.annotate(round(B_test_auroc_tuned_list[i], 3), (x[i], B_test_auroc_tuned_list[i]))
        plt.annotate(round(B_test_auroc_cc_list[i], 3), (x[i], B_test_auroc_cc_list[i]))
    plt.title('TSR_ALL3_ET_B', fontsize=15)
    # plt.savefig('PLOT/TSR_ALL3/TSR_ALL3_ET_B.png')
    plt.show()

    ### XGBClassifier
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3B_XGBC_BASED.pkl")
    B_XGBC_BASED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3B_XGBC_TUNED.pkl")
    B_XGBC_TUNED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3B_XGBC_CALIBRATED.pkl")
    B_XGBC_CALIBRATED = joblib.load(pkl_path)

    B_test_auroc_list, B_test_auroc_tuned_list, B_test_auroc_cc_list, B_XGBC_CALIBRATED_selected = algorithms(B_X_train,
                                                                                                              B_X_test,
                                                                                                              B_y_train,
                                                                                                              B_y_test,
                                                                                                              B_XGBC_BASED,
                                                                                                              B_XGBC_TUNED,
                                                                                                              B_XGBC_CALIBRATED,
                                                                                                              xgbc_selected)
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "TSR_ALL3B_XGBC_CALIBRATED_selected.pkl")
    joblib.dump(B_XGBC_CALIBRATED_selected, pkl_path)

    x = "10", "20", "30", "310"
    plt.plot(x, B_test_auroc_list, label="based")
    plt.plot(x, B_test_auroc_tuned_list, label="tuned")
    plt.plot(x, B_test_auroc_cc_list, label="calibrated")
    plt.legend()
    for i in range(len(x)):
        plt.annotate(round(B_test_auroc_list[i], 3), (x[i], B_test_auroc_list[i]))
        plt.annotate(round(B_test_auroc_tuned_list[i], 3), (x[i], B_test_auroc_tuned_list[i]))
        plt.annotate(round(B_test_auroc_cc_list[i], 3), (x[i], B_test_auroc_cc_list[i]))
    plt.title('TSR_ALL3_XGBC_B', fontsize=15)
    # plt.savefig('PLOT/TSR_ALL3/TSR_ALL3_XGBC_B.png')
    plt.show()