import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.metrics import auc, roc_curve
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

def algorithms(X_train, X_validation, y_train, y_validation, based, tuned, calibrated, model_selected):
    # BASED
    y_validation_pred = based.predict_proba(X_validation)
    fpr, tpr, thresholds = roc_curve(y_validation, y_validation_pred[:, 1])
    validation_auroc = auc(fpr, tpr)
    print('AUC of validation set:', round(validation_auroc, 3))

    # TUNED
    print('--> Tuned Parameters Best Score: ', tuned.best_score_)
    print('--> Best Parameters: \n', tuned.best_params_)

    y_validation_pred = tuned.predict_proba(X_validation)
    fpr, tpr, thresholds = roc_curve(y_validation, y_validation_pred[:, 1])
    validation_auroc_tuned = auc(fpr, tpr)
    print('AUC of validation set:', round(validation_auroc_tuned, 3))

    # CALIBRATED
    y_validation_pred = calibrated.predict_proba(X_validation)
    fpr, tpr, thresholds = roc_curve(y_validation, y_validation_pred[:, 1])
    validation_auroc_cc = auc(fpr, tpr)
    print('AUC of validation set:', round(validation_auroc_cc, 3))

    #### Selected Columns
    model_fi = calibrated.base_estimator._final_estimator.feature_importances_
    model_fi_df = pd.DataFrame(model_fi)
    model_fi_df.index = X_validation.columns
    model_fi_df.columns = (["Value"])
    model_fi_df = model_fi_df.reset_index(drop=False)
    model_fi_df.columns = (["Feature", "Value"])
    model_fi_df = model_fi_df.sort_values(["Value"], ascending=False)

    ### sigma threshold
    model_fi_df_noZERO = model_fi_df[~model_fi_df.Value.isin([0])]
    model_fi_df_noZERO_mean = model_fi_df_noZERO.Value.mean()
    model_fi_df_noZERO_std = model_fi_df_noZERO.Value.std()
    sigma_n = len(model_fi_df_noZERO[model_fi_df_noZERO.Value > model_fi_df_noZERO_mean + model_fi_df_noZERO_std])
    print(sigma_n)

    validation_auroc_list = []
    validation_auroc_tuned_list = []
    validation_auroc_cc_list = []

    for i in sigma_n, 10, 20, 30:
        model_fi_index = model_fi_df[0:i].index

        X_train_selected = X_train.iloc[:, model_fi_index]
        X_validation_selected = X_validation.iloc[:, model_fi_index]

        # base et_selected
        BASED_selected = model_selected.fit(X_train_selected, y_train)

        y_validation_pred = BASED_selected.predict_proba(X_validation_selected)
        fpr, tpr, thresholds = roc_curve(y_validation, y_validation_pred[:, 1])
        validation_auroc_selected = auc(fpr, tpr)
        validation_auroc_list.append(validation_auroc_selected)
        print('AUC of validation set:', round(validation_auroc_selected, 3))

        # tune et_selected
        rscv_selected = tuned.best_estimator_
        TUNED_selected = rscv_selected.fit(X_train_selected, y_train)

        y_validation_pred = TUNED_selected.predict_proba(X_validation_selected)
        fpr, tpr, thresholds = roc_curve(y_validation, y_validation_pred[:, 1])
        validation_auroc_selected_tuned = auc(fpr, tpr)
        validation_auroc_tuned_list.append(validation_auroc_selected_tuned)
        print('AUC of validation set:', round(validation_auroc_selected_tuned, 3))

        # calibrate et_selected
        cccv_selected = CalibratedClassifierCV(base_estimator=TUNED_selected, cv=5)
        CALIBRATED_selected = cccv_selected.fit(X_train_selected, y_train)

        y_validation_pred = CALIBRATED_selected.predict_proba(X_validation_selected)
        fpr, tpr, thresholds = roc_curve(y_validation, y_validation_pred[:, 1])
        validation_auroc_selected_cc = auc(fpr, tpr)
        validation_auroc_cc_list.append(validation_auroc_selected_cc)
        print('AUC of validation set:', round(validation_auroc_selected_cc, 3))

    validation_auroc_list.append(validation_auroc)
    validation_auroc_tuned_list.append(validation_auroc_tuned)
    validation_auroc_cc_list.append(validation_auroc_cc)

    return validation_auroc_list, validation_auroc_tuned_list, validation_auroc_cc_list, CALIBRATED_selected

if __name__ == '__main__':
    # Import datasets
    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_X_TRAIN.csv")
    X_train = pd.read_csv(csv_path)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_y_TRAIN.csv")
    y_train = pd.read_csv(csv_path)
    y_train = np.ravel(y_train)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_X_VALIDATION.csv")
    X_validation = pd.read_csv(csv_path)

    csv_path = os.path.join("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_y_VALIDATION.csv")
    y_validation = pd.read_csv(csv_path)
    y_validation = np.ravel(y_validation)

    ## Based models
    xgbc_selected = XGBClassifier(booster="gbtree", random_state=19, use_label_encoder=False, eval_metric="auc",
                                  tree_method="hist", n_jobs=-1)

    # GOOD when discharged

    ### XGBClassifier
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL3_XGBC_BASED.pkl")
    XGBC_BASED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL3_XGBC_TUNED.pkl")
    XGBC_TUNED = joblib.load(pkl_path)

    pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL3_XGBC_CALIBRATED.pkl")
    XGBC_CALIBRATED = joblib.load(pkl_path)

    validation_auroc_list, validation_auroc_tuned_list, validation_auroc_cc_list, XGBC_CALIBRATED_selected = algorithms(X_train,
                                                                                                      X_validation,
                                                                                                      y_train,
                                                                                                      y_validation,
                                                                                                      XGBC_BASED,
                                                                                                      XGBC_TUNED,
                                                                                                      XGBC_CALIBRATED,
                                                                                                      xgbc_selected)
    pkl_path = os.path.join("..", "..", "model", "model_pickle", "MICE5", "TSR_ALL3_XGBC_CALIBRATED_selected.pkl")
    #joblib.dump(XGBC_CALIBRATED_selected, pkl_path)

    x = "sigma", "10", "20", "30", "310"
    plt.plot(x, validation_auroc_list, label="based")
    plt.plot(x, validation_auroc_tuned_list, label="tuned")
    plt.plot(x, validation_auroc_cc_list, label="calibrated")
    plt.legend()
    for i in range(len(x)):
        plt.annotate(round(validation_auroc_list[i], 3), (x[i], validation_auroc_list[i]))
        plt.annotate(round(validation_auroc_tuned_list[i], 3), (x[i], validation_auroc_tuned_list[i]))
        plt.annotate(round(validation_auroc_cc_list[i], 3), (x[i], validation_auroc_cc_list[i]))
    plt.title('TSR_ALL3_XGBC', fontsize=15)
    # plt.savefig('PLOT/TSR_ALL3/TSR_ALL3_XGBC.png')
    plt.show()