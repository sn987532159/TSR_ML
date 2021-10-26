import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

def SPAN(df, df1, outcome):
    # GOOD
    df_G = df[df["discharged_mrs"].isin([0, 1, 2])]

    df_1 = df_G.loc[:, "nihs_1a_in" : "nihs_11_in"]
    df_1["age"] = df_G["age"]
    df_1_span = df_1.sum(axis = 1)

    df_1_span[df_1_span<100] = 0 #not changed
    df_1_span[df_1_span>=100] = 1 # changed

    df_G[outcome][df_G[outcome].isin([0,1,2])] = 0 # not changed
    df_G[outcome][~df_G[outcome].isin([0,1,2])] = 1 # changed

    ### confusion matrix
    print('confusion_matrix of test set:', confusion_matrix(df_G[outcome], df_1_span))

    ### auc
    span_G_auc = round(roc_auc_score(df_G[outcome], df_1_span), 3)
    print('roc_auc_score of test set:', span_G_auc)

    ### accuracy
    span_G_acc = round(accuracy_score(df_G[outcome], df_1_span), 3)
    print('accuracy_score of test set:', span_G_acc)

    ### specificity
    span_G_spe = round(precision_score(df_G[outcome], df_1_span), 3)
    print('precision_score of test set:', span_G_spe)

    ### sensitivity
    span_G_sen = round(recall_score(df_G[outcome], df_1_span), 3)
    print('recall_score of test set:', span_G_sen)

    # POOR
    df_B = df1[~df1["discharged_mrs"].isin([0, 1, 2])]
    df_2 = df_B.loc[:, "nihs_1a_in": "nihs_11_in"]
    df_2["age"] = df_B["age"]
    df_2_span = df_2.sum(axis=1)

    df_2_span[df_2_span < 100] = 1  # changed
    df_2_span[df_2_span >= 100] = 0  # not changed

    df_B[outcome][df_B[outcome].isin([0, 1, 2])] = 1  # changed
    df_B[outcome][~df_B[outcome].isin([0, 1, 2])] = 0  # not changed

    ### confusion matrix
    print('confusion_matrix of test set:', confusion_matrix(df_B[outcome], df_2_span))

    ### auc
    span_B_auc = round(roc_auc_score(df_B[outcome], df_2_span), 3)
    print('roc_auc_score of test set:', span_B_auc)

    ### accuracy
    span_B_acc = round(accuracy_score(df_B[outcome], df_2_span), 3)
    print('accuracy_score of test set:', span_B_acc)

    ### specificity
    span_B_spe = round(precision_score(df_B[outcome], df_2_span), 3)
    print('precision_score of test set:', span_B_spe)

    ### sensitivity
    span_B_sen = round(recall_score(df_B[outcome], df_2_span), 3)
    print('recall_score of test set:', span_B_sen)
    #return span_G_acc_list, span_G_auc_list, span_G_spe_list, span_G_sen_list, span_B_acc_list, span_B_auc_list, span_B_spe_list, span_B_sen_list

def THRIVE(df, df1, outcome, num_thrive):
    # GOOD
    df_G = df[df["discharged_mrs"].isin([0, 1, 2])]
    df_G = df_G[df_G["ht_id"].isin([0, 1])]
    df_G = df_G[df_G["dm_id"].isin([0, 1])]

    df_1 = df_G.loc[:, "nihs_1a_in": "nihs_11_in"]
    df_1 = pd.DataFrame(df_1.sum(axis=1))
    df_1.columns = ["nihss"]
    df_1["nihss"][df_1["nihss"] <= 10] = 0
    df_1["nihss"][(df_1["nihss"] >= 11) & (df_1["nihss"] <= 20)] = 2
    df_1["nihss"][df_1["nihss"] >= 21] = 4

    df_1["age"] = df_G["age"]
    df_1["age"][df_1["age"] <= 59] = 0
    df_1["age"][(df_1["age"] >= 60) & (df_1["nihss"] <= 79)] = 1
    df_1["age"][df_1["age"] >= 80] = 2

    df_1["thda_fl"] = df_G["thda_fl"]
    df_1["ht_id"] = df_G["ht_id"]
    df_1["dm_id"] = df_G["dm_id"]
    df_1_thrive = df_1.sum(axis=1)

    df_1_thrive[df_1_thrive < num_thrive] = 0  # not changed
    df_1_thrive[df_1_thrive >= num_thrive] = 1  # changed

    df_G[outcome][df_G[outcome].isin([0, 1, 2])] = 0  # not changed
    df_G[outcome][~df_G[outcome].isin([0, 1, 2])] = 1  # changed

    ### confusion matrix
    print('confusion_matrix of test set:', confusion_matrix(df_G[outcome], df_1_thrive))

    ### auc
    thrive_G_auc = round(roc_auc_score(df_G[outcome], df_1_thrive), 3)
    print('roc_auc_score of test set:', thrive_G_auc)

    ### accuracy
    thrive_G_acc = round(accuracy_score(df_G[outcome], df_1_thrive), 3)
    print('accuracy_score of test set:', thrive_G_acc)

    ### specificity
    thrive_G_spe = round(precision_score(df_G[outcome], df_1_thrive), 3)
    print('precision_score of test set:', thrive_G_spe)

    ### sensitivity
    thrive_G_sen = round(recall_score(df_G[outcome], df_1_thrive), 3)
    print('recall_score of test set:', thrive_G_sen)

    #POOR
    df_B = df1[~df1["discharged_mrs"].isin([0, 1, 2])]
    df_B = df_B[df_B["ht_id"].isin([0, 1])]
    df_B = df_B[df_B["dm_id"].isin([0, 1])]

    df_2 = df_B.loc[:, "nihs_1a_in": "nihs_11_in"]
    df_2 = pd.DataFrame(df_2.sum(axis=1))
    df_2.columns = ["nihss"]
    df_2["nihss"][df_2["nihss"] <= 10] = 0
    df_2["nihss"][(df_2["nihss"] >= 11) & (df_2["nihss"] <= 20)] = 2
    df_2["nihss"][df_2["nihss"] >= 21] = 4

    df_2["age"] = df_B["age"]
    df_2["age"][df_2["age"] <= 59] = 0
    df_2["age"][(df_2["age"] >= 60) & (df_2["nihss"] <= 79)] = 1
    df_2["age"][df_2["age"] >= 80] = 2

    df_2["thda_fl"] = df_B["thda_fl"]
    df_2["ht_id"] = df_B["ht_id"]
    df_2["dm_id"] = df_B["dm_id"]
    df_2_thrive = df_2.sum(axis=1)

    df_2_thrive[df_2_thrive < num_thrive] = 1  # changed
    df_2_thrive[df_2_thrive >= num_thrive] = 0  # not changed

    df_B[outcome][df_B[outcome].isin([0, 1, 2])] = 1  # changed
    df_B[outcome][~df_B[outcome].isin([0, 1, 2])] = 0  # not changed

    ### confusion matrix
    print('confusion_matrix of test set:', confusion_matrix(df_B[outcome], df_2_thrive))

    ### auc
    thrive_B_auc = round(roc_auc_score(df_B[outcome], df_2_thrive), 3)
    print('roc_auc_score of test set:', thrive_B_auc)

    ### accuracy
    thrive_B_acc = round(accuracy_score(df_B[outcome], df_2_thrive), 3)
    print('accuracy_score of test set:', thrive_B_acc)

    ### specificity
    thrive_B_spe = round(precision_score(df_B[outcome], df_2_thrive), 3)
    print('precision_score of test set:', thrive_B_spe)

    ### sensitivity
    thrive_B_sen = round(recall_score(df_B[outcome], df_2_thrive), 3)
    print('recall_score of test set:', thrive_B_sen)
    #return thrive_G_acc_list, thrive_G_auc_list, thrive_G_spe_list, thrive_G_sen_list, thrive_B_acc_list, thrive_B_auc_list, thrive_B_spe_list, thrive_B_sen_list

if __name__ == '__main__':
    # 1-month
    csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TEST_MICE1.csv")
    G1 = pd.read_csv(csv_path)
    csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TEST_MICE5.csv")
    B1 = pd.read_csv(csv_path)

    SPAN(G1, B1, "mrs_tx_1")
    THRIVE(G1, B1, "mrs_tx_1", 3)
    THRIVE(G1, B1, "mrs_tx_1", 6)

    # 3-month
    csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_TEST_MICE4.csv")
    G31 = pd.read_csv(csv_path)
    csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_TEST_MICE1.csv")
    B31 = pd.read_csv(csv_path)

    SPAN(G31, B31, "mrs_tx_3")
    THRIVE(G31, B31, "mrs_tx_3", 3)
    THRIVE(G31, B31, "mrs_tx_3", 6)