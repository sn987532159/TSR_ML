import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, f1_score

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
    tn, fp, fn, tp =confusion_matrix(df_G[outcome], df_1_span).ravel()
    print(confusion_matrix(df_G[outcome], df_1_span))

    ### auc
    print('auc of test set:', round(0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))), 3))

    ### accuracy
    print('accuracy score of test set:', (tn+tp)/(tn+tp+fn+fp))

    ### specificity
    print('specificity score of test set:', tn/(tn+fp))

    ### sensitivity (recall)
    print('sensitivity score of test set:', tp/(tp+fn))

    ### PPV (precision)
    print('PPV score of test set:', tp / (tp + fp))

    prev = (tp + fn) / (tp + fp + fn + tn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print('prevalance of test set:', round(prev, 3))
    print('PPV of test set:', round(sens * prev / (sens * prev + (1 - spec) * (1 - prev)), 3))

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
    tn, fp, fn, tp = confusion_matrix(df_B[outcome], df_2_span).ravel()
    print(confusion_matrix(df_B[outcome], df_2_span))

    ### auc
    print('auc of test set:', round(0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))), 3))

    ### accuracy
    print('accuracy score of test set:', (tn + tp) / (tn + tp + fn + fp))

    ### specificity
    print('specificity score of test set:', tn / (tn + fp))

    ### sensitivity (recall)
    print('sensitivity score of test set:', tp / (tp + fn))

    ### PPV (precision)
    print('PPV score of test set:', tp / (tp + fp))

    prev = (tp + fn) / (tp + fp + fn + tn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print('prevalance of test set:', round(prev, 3))
    print('PPV of test set:', round(sens * prev / (sens * prev + (1 - spec) * (1 - prev)), 3))
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
    tn, fp, fn, tp = confusion_matrix(df_G[outcome], df_1_thrive).ravel()
    print(confusion_matrix(df_G[outcome], df_1_thrive))

    ### auc
    print('auc of test set:', round(0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))), 3))

    ### accuracy
    print('accuracy score of test set:', (tn + tp) / (tn + tp + fn + fp))

    ### specificity
    print('specificity score of test set:', tn / (tn + fp))

    ### sensitivity (recall)
    print('sensitivity score of test set:', tp / (tp + fn))

    ### PPV (precision)
    print('PPV score of test set:', tp / (tp + fp))

    prev = (tp + fn) / (tp + fp + fn + tn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print('prevalance of test set:', round(prev, 3))
    print('PPV of test set:', round(sens * prev / (sens * prev + (1 - spec) * (1 - prev)), 3))

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
    tn, fp, fn, tp = confusion_matrix(df_B[outcome], df_2_thrive).ravel()
    print(confusion_matrix(df_B[outcome], df_2_thrive))

    ### auc
    print('auc of test set:', round(0.5 * ((tp / (tp + fn)) + (tn / (tn + fp))), 3))

    ### accuracy
    print('accuracy score of test set:', (tn + tp) / (tn + tp + fn + fp))

    ### specificity
    print('specificity score of test set:', tn / (tn + fp))

    ### sensitivity (recall)
    print('sensitivity score of test set:', tp / (tp + fn))

    ### PPV (precision)
    print('PPV score of test set:', tp / (tp + fp))

    prev = (tp + fn) / (tp + fp + fn + tn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print('prevalance of test set:', round(prev, 3))
    print('PPV of test set:', round(sens * prev / (sens * prev + (1 - spec) * (1 - prev)), 3))

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