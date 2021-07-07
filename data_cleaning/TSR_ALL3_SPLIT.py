import pandas as pd
pd.options.mode.chained_assignment = None
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# Import datasets
csv_path = os.path.join("..", "data","LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_MICE5.csv")
tsr_all3_df = pd.read_csv(csv_path)
tsr_all3_df.shape

# Convert the multiple feature and outcome into binary ones
mRS3 = tsr_all3_df.mrs_tx_3
mRS3[(mRS3 == 0) | (mRS3 == 1) | (mRS3 == 2)] = 1 #GOOD
mRS3[(mRS3 == 3) | (mRS3 == 4) | (mRS3 == 5) | (mRS3 == 6) | (mRS3 == 9)] = 0 #BAD

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
              "cre_nm", "ua_nm", "tcho_nm", "tg_nm", "hdl_nm",
              "ldl_nm", "gpt_nm", "age", "hospitalised_time"]
labels = ["mrs_tx_3", "mrs_tx_1"]
column_names_B = ["Height", "Weight", "GCS-E", "GCS-V", "GCS-M", "Systolic BP", "Diastolic BP", "Bleeding time",
                "Heart rate", "Respiratory rate", "Large artery atherosclerosis-extra",
                "Large artery atherosclerosis-intra", "Cerebral venous thrombosis", "Moyamoya Syndrome", "Radiation",
                "Dissection", "Migraine", "Antiphospholipid Ab Syndrome", "Autoimmune disease", "Hyperfibrinogenemia",
                "Prot C/Prot S deficiency", "Antithrombin III deficiency", "Homocystinuria", "Hypercoagulable state",
                "Cancer", "Atrial fibrillation", "Heart failure", "Ischemic heart (CAD, old MI)", "AMI<4W",
                "Valvular replacement", "Endocarditis", "Myxoma", "Rheumatic heart disease", "Patent foramen ovale",
                "Hemoglobin", "Hematocrit", "Platelet", "White blood cell", "PTT1", "PTT2", "PT(INR)", "Glucose (ER)",
                "Blood urea nitrogen", "Creatinine", "Uric acid", "T-CHO", "TG", "HDL", "LDL", "GPT",
                "Hospitalised treatment-Antithrombotic drugs start within 48h", "Hospitalised treatment-Aspirin",
                "Hospitalised treatment-Ticlopidine", "Hospitalised treatment-Heparin",
                "Hospitalised treatment-Warfarin", "Hospitalised treatment-IA thrombolysis",
                "Hospitalised treatment-Foley", "Hospitalised treatment-Transarterial embolization",
                "Hospitalised treatment-Sign DNR", "Hospitalised treatment-Rehab",
                "Hospitalised treatment-Endovascular treatment", "Hospitalised treatment-Aggrenox",
                "Hospitalised treatment-Clopidogrel", "Hospitalised treatment-Pletaal",
                "Hospitalised treatment-LMW heparin", "Hospitalised treatment-IV t-PA",
                "Hospitalised treatment-Ventilator", "Hospitalised treatment-Nasogastric tube",
                "Hospitalised treatment-Dysphagia Screen", "Hospitalised treatment-admission to ICU",
                "Hospitalised treatment-Smoking cessation counseling", "Hospitalised treatment-Education about stroke",
                "Hospitalised treatment-Operation", "None medications after discharged", "Aspirin after discharged",
                "Aggrenox after discharged", "Ticlopidine after discharged", "Clopidogrel after discharged",
                "Warfarin after discharged", "Pletaal after discharged", "Anti H/T drug after discharged",
                "Anti DM drug after discharged", "Lipid lowering drug after discharged",
                "None medications before admitted", "Aspirin before admitted", "Aggrenox before admitted",
                "Ticlopidine before admitted", "Clopidogrel before admitted", "Warfarin before admitted",
                "Pletaal before admitted", "Anti H/T drug before admitted", "Anti DM drug before admitted",
                "Lipid lowering drug before admitted", "Pneumonia", "Urinary tract infection", "UGI bleeding",
                "Pressure sore", "Pulmonary edema", "Acute coronary syndrome", "Seizure", "Deep vein thrombosis",
                "Stroke-in-evolution (changed NIHSS more than 2)", "Herniation", "Hemorrhagic infarct",
                "Hematoma enlargement(ICH)", "Vasospasm(SAH)", "Re-bleeding(SAH)", "Medical problems", "CT no findings",
                "MRI no findings", "LVH", "Af", "Q wave", "BI_feeding", "BI_transfers", "BI_bathing", "BI_toilet_use",
                "BI_grooming", "BI_mobility", "BI_stairs", "BI_dressing", "BI_bowel control", "BI_bladder control",
                "Discharged mRS", "CT_cortical ACA_right", "CT_cortical MCA_right", "CT_subcortical ACA_right",
                "CT_subcortical MCA_right", "CT_PCA cortex_right", "CT_thalamus_right", "CT_brainstem_right",
                "CT_cerebellum_right", "CT_watershed_right", "CT_hemorrhagic infract_right", "CT_old stroke_CI",
                "CT_cortical ACA_left", "CT_cortical MCA_left", "CT_subcortical ACA_left", "CT_subcortical MCA_left",
                "CT_PCA cortex_left", "CT_thalamus_left", "CT_brainstem_left", "CT_cerebellum_left",
                "CT_watershed_left", "CT_hemorrhagic infract_left", "CT_old stroke_CH", "MRI_cortical ACA_right",
                "MRI_cortical MCA_right", "MRI_subcortical ACA_right", "MRI_subcortical MCA_right",
                "MRI_PCA cortex_right", "MRI_thalamus_right", "MRI_brainstem_right", "MRI_cerebellum_right",
                "MRI_watershed_right", "MRI_hemorrhagic infract_right", "MRI_old stroke_CI", "MRI_cortical ACA_left",
                "MRI_cortical MCA_left", "MRI_subcortical ACA_left", "MRI_subcortical MCA_left", "MRI_PCA cortex_left",
                "MRI_thalamus_left", "MRI_brainstem_left", "MRI_cerebellum_left", "MRI_watershed_left",
                "MRI_hemorrhagic infract_left", "MRI_old stroke_CH", "Admitted NIHSS_1a", "Admitted NIHSS_1b",
                "Admitted NIHSS_1c", "Admitted NIHSS_2", "Admitted NIHSS_3", "Admitted NIHSS_4", "Admitted NIHSS_5al",
                "Admitted NIHSS_5br", "Admitted NIHSS_6al", "Admitted NIHSS_6br", "Admitted NIHSS_7",
                "Admitted NIHSS_8", "Admitted NIHSS_9", "Admitted NIHSS_10", "Admitted NIHSS_11", "Discharged NIHSS_1a",
                "Discharged NIHSS_1b", "Discharged NIHSS_1c", "Discharged NIHSS_2", "Discharged NIHSS_3",
                "Discharged NIHSS_4", "Discharged NIHSS_5al", "Discharged NIHSS_5br", "Discharged NIHSS_6al",
                "Discharged NIHSS_6br", "Discharged NIHSS_7", "Discharged NIHSS_8", "Discharged NIHSS_9",
                "Discharged NIHSS_10", "Discharged NIHSS_11", "Age", "Hospitalised duration", "Education_1",
                "Education_2", "Education_3", "Education_4", "Education_5", "Education_6", "Education_98",
                "Profession_1", "Profession_2", "Profession_3", "Profession_4", "Profession_5", "Profession_6",
                "Profession_7", "Profession_8", "Profession_9", "Profession_10", "Profession_98", "Profession_99",
                "Ways of admission_1", "Ways of admission_2", "Ways of admission_3",
                "Ischemic subtype-Large artery atherosclerosis", "Ischemic subtype-Small vessel occlusion",
                "Ischemic subtype-Cardioembolism", "Ischemic subtype-Specific etiology",
                "Ischemic subtype-Undetermined etiology", "Destination after discharged_1",
                "Destination after discharged_2", "Destination after discharged_3", "Destination after discharged_4",
                "Destination after discharged_5", "Female", "Male", "Heart disease_0", "Heart disease_1",
                "Heart disease_2", "Previous CVA_0", "Previous CVA_1", "Previous CVA_2",
                "Previous cerebral infraction_0", "Previous cerebral infraction_1", "Previous cerebral infraction_2",
                "Previous cerebral hemorrhage_0", "Previous cerebral hemorrhage_1", "Previous cerebral hemorrhage_2",
                "Polycythemia_0", "Polycythemia_1", "Polycythemia_2", "Uremia_0", "Uremia_1", "Uremia_2", "Smoking_0",
                "Smoking_1", "Smoking_2", "Previous TIA_0", "Previous TIA_1", "Previous TIA_2", "Dyslipidemia_0",
                "Dyslipidemia_1", "Dyslipidemia_2", "Hypertriglyceridemia_0", "Hypertriglyceridemia_1",
                "Hypertriglyceridemia_2", "Hypercholesterolemia_0", "Hypercholesterolemia_1", "Hypercholesterolemia_2",
                "Hypertension_0", "Hypertension_1", "Hypertension_2", "Diabetes Mellitus_0", "Diabetes Mellitus_1",
                "Diabetes Mellitus_2", "Peripheral artery disease_0", "Peripheral artery disease_1",
                "Peripheral artery disease_2", "Alcohol_0", "Alcohol_1", "Alcohol_2", "Cancer_0", "Cancer_1",
                "Cancer_2", "Parents having hypertension_0", "Parents having hypertension_1",
                "Parents having hypertension_2", "Parents having Diabetes Mellitus_0",
                "Parents having Diabetes Mellitus_1", "Parents having Diabetes Mellitus_2",
                "Parents having ischemic heart disease_0", "Parents having ischemic heart disease_1",
                "Parents having ischemic heart disease_2", "Parents having stroke or TIA_0",
                "Parents having stroke or TIA_1", "Parents having stroke or TIA_2",
                "Siblings having hypertension_0", "Siblings having hypertension_1",
                "Siblings having hypertension_2", "Siblings having hypertension_9",
                "Siblings having Diabetes Mellitus_0", "Siblings having Diabetes Mellitus_1",
                "Siblings having Diabetes Mellitus_2", "Siblings having Diabetes Mellitus_9",
                "Siblings having ischemic heart disease_0", "Siblings having ischemic heart disease_1",
                "Siblings having ischemic heart disease_2", "Siblings having ischemic heart disease_9",
                "Siblings having stroke or TIA_0", "Siblings having stroke or TIA_1", "Siblings having stroke or TIA_2",
                "Siblings having stroke or TIA_9"]
column_names_G = column_names_B[:]
column_names_G.remove("Destination after discharged_4")

# Machine Learning
## Preprocess input data (GOOD when Discharge)
## discharged mRS = GOOD (tsr_all3_df.discharged_mrs == 1)
mrs_dis1 = tsr_all3_df[(tsr_all3_df.discharged_mrs == 1) | (tsr_all3_df.discharged_mrs == 0) | (tsr_all3_df.discharged_mrs == 2)]

## input dataset
tsr_3G_input = mrs_dis1.drop(["icase_id", "idcase_id", "mrs_tx_1", "mrs_tx_3"], axis=1)
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

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_y_TRAIN.csv")
G_y_train = pd.DataFrame(G_y_train)
G_y_train.to_csv(csv_save, index=False)

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_y_TEST.csv")
G_y_test = pd.DataFrame(G_y_test)
G_y_test.to_csv(csv_save, index=False)

## scale G_X_train
G_X_train = pd.DataFrame(G_X_train)
G_X_train.columns = tsr_all3_df.drop(["icase_id", "idcase_id", "mrs_tx_1", "mrs_tx_3"], axis=1).columns

scaler = MinMaxScaler()
G_X_train[continuous] = scaler.fit_transform(G_X_train[continuous])

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=6)
G_X_train[ordinal_features] = encoder.fit_transform(G_X_train[ordinal_features])

ohe = OneHotEncoder(sparse=False, handle_unknown = "ignore")
nominal_train = ohe.fit_transform(G_X_train[nominal_features])
G_X_train = pd.concat([G_X_train, pd.DataFrame(nominal_train)], axis=1)
G_X_train = G_X_train.drop(nominal_features, axis=1)
G_X_train.columns = column_names_G

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_X_TRAIN.csv")
G_X_train.to_csv(csv_save, index=False)

## scale G_X_test
G_X_test = pd.DataFrame(G_X_test)
G_X_test.columns = tsr_all3_df.drop(["icase_id", "idcase_id", "mrs_tx_1", "mrs_tx_3"], axis=1).columns

G_X_test[continuous] = scaler.transform(G_X_test[continuous])

G_X_test[ordinal_features] = encoder.transform(G_X_test[ordinal_features])

nominal_test = ohe.transform(G_X_test[nominal_features])
G_X_test = pd.concat([G_X_test, pd.DataFrame(nominal_test)], axis=1)
G_X_test = G_X_test.drop(nominal_features, axis=1)
G_X_test.columns = column_names_G

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_X_TEST.csv")
G_X_test.to_csv(csv_save, index=False)

## Preprocess input data (BAD when Discharge)
# discharged mRS = BAD (tsr_all3_df.discharged_mrs == 0)
mrs_dis0 = tsr_all3_df[(tsr_all3_df.discharged_mrs != 1) & (tsr_all3_df.discharged_mrs != 0) & (tsr_all3_df.discharged_mrs != 2)]

## input dataset
tsr_3B_input = mrs_dis0.drop(["icase_id", "idcase_id", "mrs_tx_1", "mrs_tx_3"], axis=1)
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

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_y_TRAIN.csv")
B_y_train = pd.DataFrame(B_y_train)
B_y_train.to_csv(csv_save, index=False)

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_y_TEST.csv")
B_y_test = pd.DataFrame(B_y_test)
B_y_test.to_csv(csv_save, index=False)

## scale B_X_train
B_X_train = pd.DataFrame(B_X_train)
B_X_train.columns = tsr_all3_df.drop(["icase_id", "idcase_id", "mrs_tx_1", "mrs_tx_3"], axis=1).columns

scaler = MinMaxScaler()
B_X_train[continuous] = scaler.fit_transform(B_X_train[continuous])

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=9)
B_X_train[ordinal_features] = encoder.fit_transform(B_X_train[ordinal_features])

ohe = OneHotEncoder(sparse=False)
nominal_train = ohe.fit_transform(B_X_train[nominal_features])
B_X_train = pd.concat([B_X_train, pd.DataFrame(nominal_train)], axis=1)
B_X_train = B_X_train.drop(nominal_features, axis=1)
B_X_train.columns = column_names_B

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_X_TRAIN.csv")
B_X_train.to_csv(csv_save, index=False)

## scale B_X_test
B_X_test = pd.DataFrame(B_X_test)
B_X_test.columns = tsr_all3_df.drop(["icase_id", "idcase_id", "mrs_tx_1", "mrs_tx_3"], axis=1).columns

B_X_test[continuous] = scaler.transform(B_X_test[continuous])

B_X_test[ordinal_features] = encoder.transform(B_X_test[ordinal_features])

nominal_test = ohe.transform(B_X_test[nominal_features])
B_X_test = pd.concat([B_X_test, pd.DataFrame(nominal_test)], axis=1)
B_X_test = B_X_test.drop(nominal_features, axis=1)
B_X_test.columns = column_names_B

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_X_TEST.csv")
B_X_test.to_csv(csv_save, index=False)
