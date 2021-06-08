import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, roc_curve
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# Import datasets

csv_path = os.path.join("..", "..", "data","LINKED_DATA", "TSR_ALL", "TSR_ALL3_MICE5.csv")
tsr_3_imp_mean = pd.read_csv(csv_path)
tsr_3_imp_mean.shape

# Convert the multiple feature and outcome into binary ones

mRS3 = tsr_3_imp_mean.mrs_tx_3
mRS3[(mRS3 == 0) | (mRS3 == 1) | (mRS3 == 2)] = 1 #GOOD
mRS3[(mRS3 == 3) | (mRS3 == 4) | (mRS3 == 5) | (mRS3 == 6) | (mRS3 == 9)] = 0 #BAD
# mRS1 = tsr_3_imp_mean.mrs_tx_1
# mRS1[(mRS1 == 0) | (mRS1 == 1) | (mRS1 == 2)] = 1 #GOOD
# mRS1[(mRS1 == 3) | (mRS1 == 4) | (mRS1 == 5) | (mRS1 == 6) | (mRS1 == 9)] = 0 #BAD
# discharged = tsr_3_imp_mean.discharged_mrs
# discharged[(discharged == 0) | (discharged == 1) | (discharged == 2)] = 1 #GOOD
# discharged[(discharged == 3) | (discharged == 4) | (discharged == 5) | (discharged == 6) | (discharged == 9)] = 0 #BAD

# Group all features and the outcome

nominal_features = ["edu_id", "pro_id", "opc_id", "toast_id", "offdt_id", "gender_tx", "hd_id", "pcva_id",
                    "pcvaci_id", "pcvach_id", "po_id", "ur_id", "sm_id", "ptia_id", "hc_id", "hcht_id",
                    "hchc_id", "ht_id", "dm_id", "pad_id", "al_id", "ca_id", "fahiid_parents_1",
                    "fahiid_parents_2", "fahiid_parents_3", "fahiid_parents_4", "fahiid_brsi_1",
                    "fahiid_brsi_2", "fahiid_brsi_3", "fahiid_brsi_4"]
ordinal_features = ["gcse_nm", "gcsv_nm", "gcsm_nm", "discharged_mrs", "feeding", "transfers",
                    "bathing", "toilet_use", "grooming", "mobility", "stairs", "dressing", "bowel_control",
                    "bladder_control", "nihs_1a_in", "nihs_1b_in", "nihs_1c_in", "nihs_2_in", "nihs_3_in", "nihs_4_in",
                    "nihs_5al_in", "nihs_5br_in", "nihs_6al_in", "nihs_6br_in", "nihs_7_in", "nihs_8_in", "nihs_9_in",
                    "nihs_10_in", "nihs_11_in", "nihs_1a_out", "nihs_1b_out", "nihs_1c_out", "nihs_2_out", "nihs_3_out",
                    "nihs_4_out", "nihs_5al_out", "nihs_5br_out", "nihs_6al_out", "nihs_6br_out", "nihs_7_out",
                    "nihs_8_out", "nihs_9_out", "nihs_10_out", "nihs_11_out"]
boolean = ["toastle_fl", "toastli_fl", "toastsce_fl", "toastsmo_fl", "toastsra_fl", "toastsdi_fl",
           "toastsmi_fl", "toastsantip_fl", "toastsau_fl", "toastshy_fl", "toastspr_fl", "toastsantit_fl",
           "toastsho_fl", "toastshys_fl", "toastsca_fl", "thda_fl", "thdh_fl", "thdi_fl", "thdam_fl", "thdv_fl",
           "thde_fl", "thdm_fl", "thdr_fl", "thdp_fl", "trman_fl", "trmas_fl", "trmti_fl", "trmhe_fl",
           "trmwa_fl", "trmia_fl", "trmfo_fl", "trmta_fl", "trmsd_fl", "trmre_fl", "trmen_fl", "trmag_fl",
           "trmcl_fl", "trmpl_fl", "trmlm_fl", "trmiv_fl", "trmve_fl", "trmng_fl", "trmdy_fl", "trmicu_fl",
           "trmsm_fl", "trmed_fl", "trmop_fl", "om_fl", "omas_fl", "omag_fl", "omti_fl", "omcl_fl", "omwa_fl",
           "ompl_fl", "omanh_fl", "omand_fl", "omli_fl", "am_fl", "amas_fl", "amag_fl", "amti_fl", "amcl_fl",
           "amwa_fl", "ampl_fl", "amanh_fl", "amand_fl", "amli_fl", "compn_fl", "comut_fl", "comug_fl",
           "compr_fl", "compu_fl", "comac_fl", "comse_fl", "comde_fl", "detst_fl", "dethe_fl", "detho_fl",
           "detha_fl", "detva_fl", "detre_fl", "detme_fl", "ct_fl", "mri_fl", "ecgl_fl", "ecga_fl", "ecgq_fl",
           "cortical_aca_ctr", "cortical_mca_ctr", "subcortical_aca_ctr", "subcortical_mca_ctr", "pca_cortex_ctr",
           "thalamus_ctr", "brainstem_ctr", "cerebellum_ctr", "watershed_ctr", "hemorrhagic_infarct_ctr",
           "old_stroke_ctci", "cortical_aca_ctl", "cortical_mca_ctl", "subcortical_aca_ctl", "subcortical_mca_ctl",
           "pca_cortex_ctl", "thalamus_ctl", "brainstem_ctl", "cerebellum_ctl", "watershed_ctl",
           "hemorrhagic_infarct_ctl", "old_stroke_ctch", "cortical_aca_mrir", "cortical_mca_mrir",
           "subcortical_aca_mrir", "subcortical_mca_mrir", "pca_cortex_mrir", "thalamus_mrir", "brainstem_mrir",
           "cerebellum_mrir", "watershed_mrir", "hemorrhagic_infarct_mrir", "old_stroke_mrici", "cortical_aca_mril",
           "cortical_mca_mril", "subcortical_aca_mril", "subcortical_mca_mril", "pca_cortex_mril",
           "thalamus_mril", "brainstem_mril", "cerebellum_mril", "watershed_mril", "hemorrhagic_infarct_mril",
           "old_stroke_mrich"]
continuous = ["height_nm", "weight_nm", "sbp_nm", "dbp_nm", "bt_nm", "hr_nm", "rr_nm", "hb_nm",
              "hct_nm", "platelet_nm", "wbc_nm", "ptt1_nm", "ptt2_nm", "ptinr_nm", "er_nm", "bun_nm",
              "cre_nm", "alb_nm", "crp_nm", "hbac_nm", "ac_nm", "ua_nm", "tcho_nm", "tg_nm", "hdl_nm",
              "ldl_nm", "got_nm", "gpt_nm", "age", "hospitalised_time"]
labels = ["mrs_tx_3", "mrs_tx_1"]

# Machine Learning

## Preprocess input data (GOOD when Discharge)

## discharged mRS = GOOD (tsr_3_imp_mean.discharged_mrs == 1)
mrs_dis1 = tsr_3_imp_mean[(tsr_3_imp_mean.discharged_mrs == 1) | (tsr_3_imp_mean.discharged_mrs == 0) | (tsr_3_imp_mean.discharged_mrs == 2)]
##########mrs_dis1["mrs_tx_3"][mrs_dis1["mrs_tx_3"] == 9] = 7

## input dataset
tsr_3G_input = mrs_dis1.drop(["mrs_tx_1", "mrs_tx_3"], axis=1)
print(tsr_3G_input.shape)
tsr_3G_input = tsr_3G_input.astype("float64")
tsr_3G_input = np.array(tsr_3G_input.values)

## output dataset
tsr_3G_output = mrs_dis1.mrs_tx_3
print(tsr_3G_output.shape)
tsr_3G_output = tsr_3G_output.astype("float64")
tsr_3G_output = np.array(tsr_3G_output.values)

## train_test_split
G_X_train, G_X_test, G_y_train, G_y_test = train_test_split(tsr_3G_input, tsr_3G_output, test_size=0.3, random_state=19)
print("The shape of GOOD's X_train:", G_X_train.shape)
print("The shape of GOOD's y_train:", G_y_train.shape)
print("The shape of GOOD's X_test:", G_X_test.shape)
print("The shape of GOOD's y_test:", G_y_test.shape)

## scale G_X_train
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
G_X_train = pd.DataFrame(G_X_train)
G_X_train.columns = tsr_3_imp_mean.drop(["mrs_tx_1", "mrs_tx_3"], axis=1).columns

scaler = MinMaxScaler()
G_X_train[continuous] = scaler.fit_transform(G_X_train[continuous])

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=6)
G_X_train[ordinal_features] = encoder.fit_transform(G_X_train[ordinal_features])

ohe = OneHotEncoder(sparse=False)
nominal_train = ohe.fit_transform(G_X_train[nominal_features])
G_X_train = pd.concat([G_X_train, pd.DataFrame(nominal_train)], axis=1)
G_X_train = G_X_train.drop(nominal_features, axis=1)

G_X_train = np.array(G_X_train.values)

## scale G_X_test
G_X_test = pd.DataFrame(G_X_test)
G_X_test.columns = tsr_3_imp_mean.drop(["mrs_tx_1", "mrs_tx_3"], axis=1).columns

G_X_test[continuous] = scaler.transform(G_X_test[continuous])

G_X_test[ordinal_features] = encoder.transform(G_X_test[ordinal_features])

nominal_test = ohe.transform(G_X_test[nominal_features])
G_X_test = pd.concat([G_X_test, pd.DataFrame(nominal_test)], axis=1)
G_X_test = G_X_test.drop(nominal_features, axis=1)

G_X_test = np.array(G_X_test.values)

## Oversampling
from collections import Counter
from imblearn.over_sampling import SMOTE
print('Original dataset shape %s' % Counter(G_y_train))
smote = SMOTE(sampling_strategy='minority', random_state=19)  # oversampling
G_X_train_smote, G_y_train_smote = smote.fit_resample(G_X_train, G_y_train)
print('Resampled dataset shape %s' % Counter(G_y_train_smote))

## Undersampling
# Method 1 - RandomUnderSampler
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
print('Original dataset shape %s' % Counter(G_y_train))
rus = RandomUnderSampler(sampling_strategy='majority', random_state=19)  # oversampling
G_X_train_rus, G_y_train_rus = rus.fit_resample(G_X_train, G_y_train)
print('Resampled dataset shape %s' % Counter(G_y_train_rus))

# Method 2 - TomekLinks
from collections import Counter
from imblearn.under_sampling import TomekLinks
print('Original dataset shape %s' % Counter(G_y_train))
tl = TomekLinks()
G_X_train_tl, G_y_train_tl = tl.fit_resample(G_X_train, G_y_train)
print('Resampled dataset shape %s' % Counter(G_y_train_tl))

## Algorithms
#____________________________________________________________
### try decision tree pruning
from sklearn.tree import DecisionTreeClassifier
path=DecisionTreeClassifier(random_state=1).cost_complexity_pruning_path(G_X_train_tl, G_y_train_tl)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.show()

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=ccp_alpha)
    clf.fit(G_X_train_tl, G_y_train_tl)
    clfs.append(clf)

print("Number of nodes in the last tree is: {} with ccp_alpha: {} and a depth of: {}".format(clfs[-1].tree_.node_count, ccp_alphas[-1],clfs[-1].tree_.max_depth))

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
#fig.tight_layout()
plt.show()

train_scores = [clf.score(G_X_train_tl, G_y_train_tl) for clf in clfs]
test_scores = [clf.score(G_X_test, G_y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

index_best_model = np.argmax(test_scores)
best_model = clfs[index_best_model]
print('Training accuracy of best model: ',best_model.score(G_X_train_tl, G_y_train_tl))
print('Test accuracy of best model: ',best_model.score(G_X_test, G_y_test))

best_model.fit(G_X_train_tl, G_y_train_tl)
print("AUC of training set:", roc_auc_score(G_y_train_tl, best_model.predict(G_X_train_tl)))
G_y_predicted = best_model.predict(G_X_test)
print("AUC of testing set:", roc_auc_score(G_y_test, G_y_predicted))
confusion_matrix(G_y_test, G_y_predicted)
#____________________________________________________________

### base rf
rf = RandomForestClassifier(random_state=19, class_weight={0: 1, 1: 25}) # when using tl
# rf = RandomForestClassifier(random_state=19, class_weight={0: 1, 1: 25}, ccp_alpha = 0.000009) # when using tl
# rf = RandomForestClassifier(random_state=19)
rf.fit(G_X_train_tl, G_y_train_tl)
G_y_train_pred = rf.predict_proba(G_X_train_tl)
fpr, tpr, thresholds = roc_curve(G_y_train_tl, G_y_train_pred[:, 1])
G_train_auroc = auc(fpr, tpr)
print('AUC of training set:', G_train_auroc)
G_y_test_pred = rf.predict_proba(G_X_test)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc)

# confusion_matrix(G_y_test, G_y_predicted)

### base et
et = ExtraTreesClassifier(random_state=19, class_weight={0: 1, 1: 25}) # when using tl
# et = ExtraTreesClassifier(random_state=19, class_weight={0: 1, 1: 25}, ccp_alpha = 0.000009) # when using tl
# et = ExtraTreesClassifier(random_state=19)
et.fit(G_X_train_tl, G_y_train_tl)
G_y_train_pred = et.predict_proba(G_X_train_tl)
fpr, tpr, thresholds = roc_curve(G_y_train_tl, G_y_train_pred[:, 1])
G_train_auroc = auc(fpr, tpr)
print('AUC of training set:', G_train_auroc)
G_y_test_pred = et.predict_proba(G_X_test)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc)

# confusion_matrix(G_y_test, G_y_predicted)

### base svc
svc = SVC(random_state=19, probability = True, class_weight={0: 1, 1: 25})  # when using tl
# svc = SVC(random_state=19, probability = True)
svc.fit(G_X_train_tl, G_y_train_tl)
G_y_train_pred = svc.predict_proba(G_X_train_tl)
fpr, tpr, thresholds = roc_curve(G_y_train_tl, G_y_train_pred[:, 1])
G_train_auroc = auc(fpr, tpr)
print('AUC of training set:', G_train_auroc)
G_y_test_pred = svc.predict_proba(G_X_test)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc)

# confusion_matrix(G_y_test, G_y_predicted)

### lsvc
lsvc = LinearSVC(random_state=19, dual=False, class_weight={0: 1, 1: 25})  # when using tl
# lsvc = LinearSVC(dual=False, random_state=19)
lsvc.fit(G_X_train_tl, G_y_train_tl)
G_y_train_pred = lsvc._predict_proba_lr(G_X_train_tl)
fpr, tpr, thresholds = roc_curve(G_y_train_tl, G_y_train_pred[:, 1])
G_train_auroc = auc(fpr, tpr)
print('AUC of training set:', G_train_auroc)
G_y_test_pred = lsvc._predict_proba_lr(G_X_test)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc)

# confusion_matrix(G_y_test, G_y_predicted)

### xgbc
xgbc = XGBClassifier(random_state=19, use_label_encoder=False, eval_metric="auc",
                     scale_pos_weight=(1 / 25))  # when using tl
# xgbc = XGBClassifier(random_state=19, use_label_encoder=False, eval_metric = "auc")
xgbc.fit(G_X_train_tl, G_y_train_tl)
G_y_train_pred = xgbc.predict_proba(G_X_train_tl)
fpr, tpr, thresholds = roc_curve(G_y_train_tl, G_y_train_pred[:, 1])
G_train_auroc = auc(fpr, tpr)
print('AUC of training set:', G_train_auroc)
G_y_test_pred = xgbc.predict_proba(G_X_test)
fpr, tpr, thresholds = roc_curve(G_y_test, G_y_test_pred[:, 1])
G_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', G_test_auroc)

# confusion_matrix(G_y_test, G_y_predicted)

### neural network
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, input_dataset):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dataset, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim = 1) # dim = 1 -> compute by rows
        return x

model = Model(G_X_train_tl.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
model

import tqdm
EPOCHS = 100
G_X_train_tl = Variable(torch.from_numpy(G_X_train_tl)).float()
G_y_train_tl = Variable(torch.from_numpy(G_y_train_tl)).long()
G_X_test = Variable(torch.from_numpy(G_X_test)).float()
G_y_test = Variable(torch.from_numpy(G_y_test)).long()

loss_list = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))

for epoch in tqdm.trange(EPOCHS):
    y_pred = model(G_X_train_tl)
    loss = loss_fn(y_pred, G_y_train_tl)
    loss_list[epoch] = loss.item()

    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        y_pred = model(G_X_test)
        correct = (torch.argmax(y_pred, dim=1) == G_y_test).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)
ax1.plot(accuracy_list)
ax1.set_ylabel("validation accuracy")
ax2.plot(loss_list)
ax2.set_ylabel("validation loss")
ax2.set_xlabel("epochs");
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')

# One hot encoding
enc = OneHotEncoder()
Y_onehot = enc.fit_transform(G_y_test[:, np.newaxis]).toarray()

with torch.no_grad():
    y_pred = model(G_X_test).numpy()
    fpr, tpr, threshold = roc_curve(Y_onehot.ravel(), y_pred.ravel())

plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()

# Anomaly detection
G_X_train, G_X_test, G_y_train, G_y_test = train_test_split(tsr_3G_input, tsr_3G_output, test_size=0.3, random_state=19)
print("The shape of GOOD's X_train:", G_X_train.shape)
print("The shape of GOOD's y_train:", G_y_train.shape)
print("The shape of GOOD's X_test:", G_X_test.shape)
print("The shape of GOOD's y_test:", G_y_test.shape)

G_X_train.shape, G_y_train.shape

G_X_train = pd.DataFrame(G_X_train)
# G_X_train.columns = tsr_3_imp_mean.drop(["mrs_tx_1", "mrs_tx_3"], axis=1).columns
G_y_train = pd.DataFrame(G_y_train)
G_y_train.columns = ["mrs_tx_3"]

train = pd.concat([G_X_train, G_y_train], axis=1)
train

from pyod.models.loci import LOCI

clf = LOCI(contamination=0.1)
clf.fit(G_X_train, G_y_train)

# get outlier scores
y_train_scores = clf.decision_scores_  # raw outlier scores
y_test_scores = clf.decision_function(G_X_test)  # outlier scores

labels = pd.DataFrame(clf.labels_)

from pyod.models.loci import LOCI

clf = LOCI(contamination=0.1)
clf.fit(G_X_train, G_y_train)

# get outlier scores
y_train_scores = clf.decision_scores_  # raw outlier scores
y_test_scores = clf.decision_function(G_X_test)  # outlier scores

labels = pd.DataFrame(clf.labels_)

train.reset_index(inplace=False)
train["labels"] = labels
train = train[train.labels == 0]
train.shape

train

G_y_train = train["mrs_tx_3"]
print(np.array(G_y_train.values))
G_y_train = np.array(G_y_train)

G_X_train = train.drop(['labels', 'mrs_tx_3'], axis=1)
G_X_train = np.array(G_X_train)

from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score

auc = round(roc_auc_score(G_y_test, y_test_scores), ndigits=4)
prn = round(precision_n_scores(G_y_test, y_test_scores), ndigits=4)
print(f'{clf} AUC:{auc}, precision @ rank n:{prn}')

# Machine Learning

## Preprocess input data (BAD when Discharge)

# discharged mRS = BAD (tsr_3_imp_mean.discharged_mrs == 0)
mrs_dis0 = tsr_3_imp_mean[(tsr_3_imp_mean.discharged_mrs != 1) & (tsr_3_imp_mean.discharged_mrs != 0) & (tsr_3_imp_mean.discharged_mrs != 2)]
##########mrs_dis0["mrs_tx_3"][mrs_dis0["mrs_tx_3"] == 9] = 7

## input dataset
tsr_3B_input = mrs_dis0.drop(["mrs_tx_1", "mrs_tx_3"], axis=1)
print(tsr_3B_input.shape)
tsr_3B_input = tsr_3B_input.astype("float64")
tsr_3B_input = np.array(tsr_3B_input.values)

## output dataset
tsr_3B_output = mrs_dis0.mrs_tx_3
print(tsr_3B_output.shape)
tsr_3B_output = tsr_3B_output.astype("float64")
tsr_3B_output = np.array(tsr_3B_output.values)

## train_test_split
B_X_train, B_X_test, B_y_train, B_y_test = train_test_split(tsr_3B_input, tsr_3B_output, test_size=0.3, random_state=19)
print("The shape of X_train:", B_X_train.shape)
print("The shape of y_train:", B_y_train.shape)
print("The shape of X_test:", B_X_test.shape)
print("The shape of y_test:", B_y_test.shape)

## scale B_X_train
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

B_X_train = pd.DataFrame(B_X_train)
B_X_train.columns = tsr_3_imp_mean.drop(["mrs_tx_1", "mrs_tx_3"], axis=1).columns

scaler = MinMaxScaler()
B_X_train[continuous] = scaler.fit_transform(B_X_train[continuous])

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=9)
B_X_train[ordinal_features] = encoder.fit_transform(B_X_train[ordinal_features])

ohe = OneHotEncoder(sparse=False)
nominal_train = ohe.fit_transform(B_X_train[nominal_features])
B_X_train = pd.concat([B_X_train, pd.DataFrame(nominal_train)], axis=1)
B_X_train = B_X_train.drop(nominal_features, axis=1)

B_X_train = np.array(B_X_train.values)

## scale B_X_test
B_X_test = pd.DataFrame(B_X_test)

B_X_test.columns = tsr_3_imp_mean.drop(["mrs_tx_1", "mrs_tx_3"], axis=1).columns

B_X_test[continuous] = scaler.transform(B_X_test[continuous])

B_X_test[ordinal_features] = encoder.transform(B_X_test[ordinal_features])

nominal_test = ohe.transform(B_X_test[nominal_features])
B_X_test = pd.concat([B_X_test, pd.DataFrame(nominal_test)], axis=1)
B_X_test = B_X_test.drop(nominal_features, axis=1)

B_X_test = np.array(B_X_test.values)

## Oversampling

print('Original dataset shape %s' % Counter(B_y_train))
smote = SMOTE(sampling_strategy='minority', random_state=19)  # oversampling
B_X_train_smote, B_y_train_smote = smote.fit_resample(B_X_train, B_y_train)
print('Resampled dataset shape %s' % Counter(B_y_train_smote))

## Undersampling

# Method 1 - RandomUnderSampler
print('Original dataset shape %s' % Counter(B_y_train))
rus = RandomUnderSampler(sampling_strategy='majority', random_state=19)  # oversampling
B_X_train_rus, B_y_train_rus = rus.fit_resample(B_X_train, B_y_train)
print('Resampled dataset shape %s' % Counter(B_y_train_rus))

# Method 2 - TomekLinks
from collections import Counter
from imblearn.under_sampling import TomekLinks
print('Original dataset shape %s' % Counter(B_y_train))
tl = TomekLinks()
B_X_train_tl, B_y_train_tl = tl.fit_resample(B_X_train, B_y_train)
print('Resampled dataset shape %s' % Counter(B_y_train_tl))

## Algorithms

### base rf
rf = RandomForestClassifier(random_state=19, class_weight={0: 1, 1: 25}) # when using tl
# rf = RandomForestClassifier(random_state=19, class_weight={0: 1, 1: 25}, ccp_alpha = 0.000009) # when using tl
# rf = RandomForestClassifier(random_state=19)
rf.fit(B_X_train_tl, B_y_train_tl)
B_y_train_pred = rf.predict_proba(B_X_train_tl)
fpr, tpr, thresholds = roc_curve(B_y_train_tl, B_y_train_pred[:, 1])
B_train_auroc = auc(fpr, tpr)
print('AUC of training set:', B_train_auroc)
B_y_test_pred = rf.predict_proba(B_X_test)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc)

# confusion_matrix(B_y_test, B_y_predicted)

### tune rf
from sklearn.model_selection import GridSearchCV
hyperparameters_rf = {"n_estimators": [50, 100],
                      "criterion": ["gini", "entropy"],
                      "max_depth": [None, 10],
                      "min_samples_split": [2, 100],
                      "max_features": ["auto", "sqrt", "log2"],
                      "bootstrap": [True, False],
                      "class_weight": ['balanced', {0: 680, 1: 13445}, {0: 13445, 1: 680}],
                      "max_samples": [100, 0.7],
                      "random_state": [19]}
rf_gscv = GridSearchCV(estimator=RandomForestClassifier(),
                       param_grid=hyperparameters_rf,
                       n_jobs=3,
                       scoring='roc_auc',
                       verbose=2,
                       cv=10)

rf_gsCV = rf_gscv.fit(B_X_train_tl, B_y_train_tl)
print('--> Tuned Parameters Best Score: ', rf_gsCV.best_score_)
print('--> Best Parameters: \n', rf_gsCV.best_params_)

### best rf
best_rf = RandomForestClassifier(bootstrap=False,
                                 class_weight="balanced",
                                 criterion="entropy",
                                 max_depth=None,
                                 max_features="auto",
                                 max_samples=100,
                                 min_samples_split=2,
                                 n_estimators=100,
                                 random_state=19)
best_rf.fit(B_X_train_tl, B_y_train_tl)
print("AUC of training set:", roc_auc_score(B_y_train_tl, best_rf.predict(B_X_train_tl)))
B_y_predicted = best_rf.predict(B_X_test)
print("AUC of testing set:", roc_auc_score(B_y_test, B_y_predicted))
confusion_matrix(B_y_test, B_y_predicted)

### base et
et = ExtraTreesClassifier(random_state=19, class_weight={0: 1, 1: 25}) # when using tl
# et = ExtraTreesClassifier(random_state=19, class_weight={0: 1, 1: 25}, ccp_alpha = 0.000009) # when using tl
# et = ExtraTreesClassifier(random_state=19)
et.fit(B_X_train_tl, B_y_train_tl)
B_y_train_pred = et.predict_proba(B_X_train_tl)
fpr, tpr, thresholds = roc_curve(B_y_train_tl, B_y_train_pred[:, 1])
B_train_auroc = auc(fpr, tpr)
print('AUC of training set:', B_train_auroc)
B_y_test_pred = et.predict_proba(B_X_test)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc)

# confusion_matrix(B_y_test, B_y_predicted)

### base svc
svc = SVC(random_state=19, probability = True, class_weight={0: 1, 1: 25})  # when using tl
# svc = SVC(random_state=19, probability = True)
svc.fit(B_X_train_tl, B_y_train_tl)
B_y_train_pred = svc.predict_proba(B_X_train_tl)
fpr, tpr, thresholds = roc_curve(B_y_train_tl, B_y_train_pred[:, 1])
B_train_auroc = auc(fpr, tpr)
print('AUC of training set:', B_train_auroc)
B_y_test_pred = svc.predict_proba(B_X_test)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc)

# confusion_matrix(B_y_test, B_y_predicted)

### tune svc
param_range = [0.01, 0.1, 1, 10, 100]
hyperparameters_svc = [
    {'kernel': ['rbf', 'linear'], 'gamma': ['auto', 'scale', param_range], 'C': param_range, 'shrinking': [True, False],
     'decision_function_shape': ['ovo', 'ovr'], 'random_state': [19]},
    {'kernel': ['sigmoid'], 'gamma': ['auto', 'scale', param_range], 'C': param_range, 'shrinking': [True, False],
     'coef0': param_range, 'decision_function_shape': ['ovo', 'ovr'], 'random_state': [19]}]
svc_gscv = GridSearchCV(estimator=SVC(),
                        param_grid=hyperparameters_svc,
                        n_jobs=3,
                        scoring='roc_auc',
                        verbose=2,
                        cv=10)

svc_gsCV = svc_gscv.fit(B_X_train_tl, B_y_train_tl)
print('--> Tuned Parameters Best Score: ', svc_gsCV.best_score_)
print('--> Best Parameters: \n', svc_gsCV.best_params_)

### best svc
best_svc = SVC(C=10,
               decision_function_shape="ovo",
               gamma="auto",
               kernel="rbf",
               random_state=19,
               shrinking=True,
               probability=True)
best_svc.fit(B_X_train_tl, B_y_train_tl)
print("AUC of training set:", roc_auc_score(B_y_train_tl, best_svc.predict(B_X_train_tl)))
B_y_predicted = best_svc.predict(B_X_test)
print("AUC of testing set:", roc_auc_score(B_y_test, B_y_predicted))
confusion_matrix(B_y_test, B_y_predicted)

### base lsvc
lsvc = LinearSVC(random_state=19, dual=False, class_weight={0: 1, 1: 25})  # when using tl
# lsvc = LinearSVC(random_state=19)
lsvc.fit(B_X_train_tl, B_y_train_tl)
B_y_train_pred = lsvc._predict_proba_lr(B_X_train_tl)
fpr, tpr, thresholds = roc_curve(B_y_train_tl, B_y_train_pred[:, 1])
B_train_auroc = auc(fpr, tpr)
print('AUC of training set:', B_train_auroc)
B_y_test_pred = lsvc._predict_proba_lr(B_X_test)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc)

# confusion_matrix(B_y_test, B_y_predicted)

### base xgbc
xgbc = XGBClassifier(random_state=19, use_label_encoder=False, eval_metric="auc",
                     scale_pos_weight=(1 / 25))  # when using tl
# xgbc = XGBClassifier(random_state=19, use_label_encoder=False, eval_metric = "auc")
xgbc.fit(B_X_train_tl, B_y_train_tl)
B_y_train_pred = xgbc.predict_proba(B_X_train_tl)
fpr, tpr, thresholds = roc_curve(B_y_train_tl, B_y_train_pred[:, 1])
B_train_auroc = auc(fpr, tpr)
print('AUC of training set:', B_train_auroc)
B_y_test_pred = xgbc.predict_proba(B_X_test)
fpr, tpr, thresholds = roc_curve(B_y_test, B_y_test_pred[:, 1])
B_test_auroc = auc(fpr, tpr)
print('AUC of testing set:', B_test_auroc)

# confusion_matrix(B_y_test, B_y_predicted)