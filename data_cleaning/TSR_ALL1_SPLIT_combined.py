import pandas as pd
pd.options.mode.chained_assignment = None
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# Import datasets
csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TRAIN_MICE5.csv")
tsr_all1_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_VALIDATION_MICE5.csv")
tsr_all1_validation = pd.read_csv(csv_path)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TEST_MICE5.csv")
tsr_all1_test = pd.read_csv(csv_path)

# Group all features and the outcome
nominal_features = ["opc_id", "toast_id", "offdt_id", "gender_tx", "hd_id", "pcva_id",
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
labels = ["mrs_tx_1"]
column_names = ["Height", "Weight", "GCS-E", "GCS-V", "GCS-M", "Systolic BP", "Diastolic BP", "Bleeding time",
                  "Heart rate", "Respiratory rate", "Large artery atherosclerosis-extra",
                  "Large artery atherosclerosis-intra", "Cerebral venous thrombosis", "Moyamoya Syndrome", "Radiation",
                  "Dissection", "Migraine", "Antiphospholipid Ab Syndrome", "Autoimmune disease", "Hyperfibrinogenemia",
                  "Prot C/Prot S deficiency", "Antithrombin III deficiency", "Homocystinuria", "Hypercoagulable state",
                  "Cancer", "Atrial fibrillation", "Heart failure", "Ischemic heart (CAD or old MI)", "AMI smaller than 4W",
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
                  "Discharged NIHSS_10", "Discharged NIHSS_11", "Age", "Hospitalised duration",
                  "Ways of admission_inpatient", "Ways of admission_outpatient", "Ways of admission_emergency",
                  "Ischemic subtype-Large artery atherosclerosis", "Ischemic subtype-Small vessel occlusion",
                  "Ischemic subtype-Cardioembolism", "Ischemic subtype-Specific etiology",
                  "Ischemic subtype-Undetermined etiology", "Destination after discharged_home",
                  "Destination after discharged_nursing home", "Destination after discharged_transfer",
                  "Destination after discharged_respiratory care ward", "Destination after discharged_Rehabilitation",
                  "Female", "Male", "Without heart disease history", "With heart disease history",
                  "Heart disease history unknown", "Without previous CVA history", "With Previous CVA history",
                  "Previous CVA history unknown", "Without previous cerebral infraction history",
                  "With previous cerebral infraction history", "Previous cerebral infraction history unknown",
                  "Without previous cerebral hemorrhage history", "With previous cerebral hemorrhage history",
                  "Previous cerebral hemorrhage history unknown", "Without polycythemia history",
                  "With polycythemia history", "Polycythemia history unknown", "Without uremia history",
                  "with uremia history", "Uremia history unknown", "Without smoking history", "With smoking history",
                  "Smoking history unknown", "Without previous TIA history", "With previous TIA history",
                  "Previous TIA history unknown", "Without dyslipidemia history", "With dyslipidemia history",
                  "Dyslipidemia history unknown", "Without hypertriglyceridemia history", "With hypertriglyceridemia history",
                  "Hypertriglyceridemia history unknown", "Without hypercholesterolemia history",
                  "With hypercholesterolemia history", "Hypercholesterolemia history unknown", "Without hypertension history",
                  "With hypertension history", "Hypertension history unknown", "Without diabetes mellitus history",
                  "With diabetes mellitus history", "Diabetes mellitus history unknown",
                  "Without peripheral artery disease history", "With peripheral artery disease history",
                  "Peripheral artery disease history unknown", "Without alcohol history", "With alcohol history",
                  "Alcohol history unknown", "Without cancer history", "With cancer history", "Cancer history unknown",
                  "Parents without hypertension history", "Parents with hypertension history",
                  "Parents hypertension history unknown", "Parents without diabetes mellitus history",
                  "Parents with diabetes mellitus history", "Parents diabetes mellitus history unknown",
                  "Parents without ischemic heart disease history", "Parents with ischemic heart disease history",
                  "Parents ischemic heart disease history unknown", "Parents without stroke or TIA history",
                  "Parents with stroke or TIA history", "Parents stroke or TIA history unknown",
                  "Siblings without hypertension history", "Siblings with hypertension history",
                  "Siblings hypertension history unknown", "Hypertension_no siblings",
                  "Siblings without diabetes mellitus history", "Siblings with diabetes mellitus history",
                  "Siblings diabetes mellitus history unknown", "Diabetes mellitus_no siblings",
                  "Siblings without ischemic heart disease history", "Siblings with ischemic heart disease history",
                  "Siblings ischemic heart disease history unknown", "Ischemic heart disease_no siblings",
                  "Siblings without stroke or TIA history", "Siblings with stroke or TIA history",
                  "Siblings stroke or TIA history unknown", "Stroke or TIA_no siblings"]

# Machine Learning
## Preprocess input data (GOOD when Discharge, discharged_mrs == 0)
G_train = tsr_all1_train[tsr_all1_train["discharged_mrs"].isin([0, 1, 2])]
G_validation = tsr_all1_validation[tsr_all1_validation["discharged_mrs"].isin([0, 1, 2])]
G_test = tsr_all1_test[tsr_all1_test["discharged_mrs"].isin([0, 1, 2])]

## CHANGE or NOT CHANGE
for i in [G_train, G_validation, G_test]:
    i.reset_index(drop=True, inplace=True)
    i["CHANGE"] = 1  # changes
    i["CHANGE"][i["mrs_tx_1"].isin([0, 1, 2])] = 0  # no changes

## Preprocess input data (BAD when Discharge, discharged_mrs == 1)
B_train = tsr_all1_train[~tsr_all1_train["discharged_mrs"].isin([0,1,2])]
B_validation = tsr_all1_validation[~tsr_all1_validation["discharged_mrs"].isin([0,1,2])]
B_test = tsr_all1_test[~tsr_all1_test["discharged_mrs"].isin([0,1,2])]

## CHANGE or NOT CHANGE
for i in [B_train, B_validation, B_test]:
    i.reset_index(drop=True, inplace=True)
    i["CHANGE"] = 1 #changes
    i["CHANGE"][~i["mrs_tx_1"].isin([0,1,2])] = 0 #no changes

tsr_train = pd.concat([G_train, B_train], axis=0)
tsr_validation = pd.concat([G_validation, B_validation], axis=0)
tsr_test = pd.concat([G_test, B_test], axis=0)

## input dataset
tsr_train_x = tsr_train.drop(["icase_id", "idcase_id", "mrs_tx_1", "CHANGE"], axis=1).reset_index(drop=True)
print(tsr_train_x.shape)

tsr_validation_x = tsr_validation.drop(["icase_id", "idcase_id", "mrs_tx_1", "CHANGE"], axis=1).reset_index(drop=True)
print(tsr_validation_x.shape)

tsr_test_x = tsr_test.drop(["icase_id", "idcase_id", "mrs_tx_1", "CHANGE"], axis=1).reset_index(drop=True)
print(tsr_test_x.shape)

## output dataset
tsr_train_y = tsr_train.CHANGE
print(tsr_train_y.shape)

tsr_validation_y = tsr_validation.CHANGE
print(tsr_validation_y.shape)

tsr_test_y = tsr_test.CHANGE
print(tsr_test_y.shape)

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_y_TRAIN.csv")
pd.DataFrame(tsr_train_y).to_csv(csv_save, index=False)

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_y_VALIDATION.csv")
pd.DataFrame(tsr_validation_y).to_csv(csv_save, index=False)

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_y_TEST.csv")
pd.DataFrame(tsr_test_y).to_csv(csv_save, index=False)

## scale G_X_train
scaler = MinMaxScaler()
tsr_train_x[continuous] = scaler.fit_transform(tsr_train_x[continuous])

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=99)
tsr_train_x[ordinal_features] = encoder.fit_transform(tsr_train_x[ordinal_features])

ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
nominal_train = ohe.fit_transform(tsr_train_x[nominal_features])
tsr_train_x = pd.concat([tsr_train_x, pd.DataFrame(nominal_train)], axis=1)
tsr_train_x = tsr_train_x.drop(nominal_features, axis=1)
tsr_train_x.columns = column_names

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_X_TRAIN.csv")
tsr_train_x.to_csv(csv_save, index=False)

## scale G_X_validation
tsr_validation_x[continuous] = scaler.transform(tsr_validation_x[continuous])

tsr_validation_x[ordinal_features] = encoder.transform(tsr_validation_x[ordinal_features])

nominal_validation = ohe.transform(tsr_validation_x[nominal_features])
tsr_validation_x = pd.concat([tsr_validation_x, pd.DataFrame(nominal_validation)], axis=1)
tsr_validation_x = tsr_validation_x.drop(nominal_features, axis=1)
tsr_validation_x.columns = column_names

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_X_VALIDATION.csv")
tsr_validation_x.to_csv(csv_save, index=False)

## scale G_X_test
tsr_test_x[continuous] = scaler.transform(tsr_test_x[continuous])

tsr_test_x[ordinal_features] = encoder.transform(tsr_test_x[ordinal_features])

nominal_test = ohe.transform(tsr_test_x[nominal_features])
tsr_test_x = pd.concat([tsr_test_x, pd.DataFrame(nominal_test)], axis=1)
tsr_test_x = tsr_test_x.drop(nominal_features, axis=1)
tsr_test_x.columns = column_names

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_X_TEST.csv")
tsr_test_x.to_csv(csv_save, index=False)