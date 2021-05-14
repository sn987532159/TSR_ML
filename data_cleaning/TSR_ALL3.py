import pandas as pd
import os
import numpy as np

csv_path = os.path.join("..","data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3.csv")
tsr_all3 = pd.read_csv(csv_path, low_memory=False, encoding = "windows-1252")
tsr_all3.head()

tsr_all3.describe()

#drop unrelated features
tsr_all3 = tsr_all3.drop(["fstatus_id_1", "location_id_1", "torg_id_1", "flu_id_1", "fluorg_id_1", "fluorg_tx_1",
                          "fluresult_tx_1", "death_dt_1", "death_id_1", "deathsk_id_1", "deatho_tx_1", 
                          "veihdorg_id_1", "versorg_id_1", "torg_tx_1", "versorg_tx_1", "veihdorg_tx_1", 
                          "fstatus_id_3", "location_id_3", "torg_id_3", "flu_id_3", "fluorg_id_3", "fluorg_tx_3",
                          "fluresult_tx_3", "deatho_tx_3", "versorg_id_3", "veihdorg_id_3", "torg_tx_3", 
                          "versorg_tx_3", "veihdorg_tx_3", "index", "iprotocol_id", "icase_id.1", "idcase_id.1",
                          "cstatus_id", "org_id", "dctype24_id", "patient_id", "input_nm", "age_nm", "proot_tx",
                          "itown_id", "addr_tx", "telh_tx", "telp_tx", "telf_tx", "ftitle_tx", "casememo_tx", 
                          "ivtpath_fl", "ivtpaah_fl", "nivtpa99_tx", "icd_tx", "icdtia_id", "icdo_tx", 
                          "toastscat_tx", "toastso_tx", "cich_id", "csah_id", "csaho_tx", "thdo_fl", "thdoo_tx",
                          "trmot_tx", "om_id", "omwa_tx", "omand_id", "omli_id", "omliot_tx", "omliot2_tx", 
                          "amliot_tx", "amliot2_tx", "como_tx", "deto_tx", "offd_tx", "offdtorg_id", "offdtorg_tx",
                          "nihsinti_tx", "nihsotti_tx", "brs_dt", "ctti_tx", "cto_tx", "mriti_tx", "mrio_tx", 
                          "ecgo_tx", "create_dt", "createstaff_id", "sysupd_dt", "sysupdstaff_id", "modify_nm",
                          "iguid_ft", "icase_id.2", "idcase_id.2", "icase_id.3", "idcase_id.3", "index.1", 
                          "iprotocol_id.1", "icase_id.4", "idcase_id.4", "hdmt_id", "pcvamt_id", "pomt_id",
                          "ua_id", "uamt_id", "urmt_id", "ptiamt_id", "hcy_nm", "hcmt_id", "hty_nm", "htmt_id",
                          "dmy_nm", "dmmt_id", "padmt_id", "ca_tx", "ot_tx", "thishc_id", "iguid_ft.1", 
                          "icase_id.5", "idcase_id.5", "icase_id.6", "idcase_id.6", "index.2", "iprotocol_id.2",
                          "icase_id.7", "cname_tx", "cid_id", "birth_dt", "ve_id_1", "ve_id_3", "offd_id", 
                          "nivtpa99_fl", "toastso_fl", "thdoo_fl", "trmot_fl", "omliot_fl", "omliot2_fl", 
                          "amliot_fl", "amliot2_fl", "como_fl", "deto_fl", "ot_id", "smc_nm", "smy_nm", "ecgo_fl", 
                          "omora_fl", "omins_fl", "omst_fl", "omns_fl", "cd_id", "tccs_id", "mcd_id", "vers_fl_1", 
                          "veihd_fl_1", "vers_fl_3", "veihd_fl_3", "ih_fl", "onset_fl", "ot_fl", "flook_fl",
                          "fctoh_fl", "nivtpa_id", "nivtpa1_fl", "nivtpa2_fl", "nivtpa3_fl", "nivtpa4_fl", 
                          "nivtpa5_fl", "nivtpa6_fl", "nivtpa7_fl", "nivtpa8_fl", "nivtpa9_fl", "nivtpa10_fl", 
                          "nivtpa11_fl", "omad_fl", "dethoh_fl", "ecg_id", "mra_fl","cta_fl", "dsa_fl", "cdr_id", 
                          "cdl_id", "tccsr_id", "tccsl_id", "tccsba_id", "mcdr_id", "mcdl_id", "mcdba_id", 
                          "mcdri_id", "mcdli_id"],
                          axis = 1)

tsr_all3 = tsr_all3.drop(["verscich_id_1", "vers_dt_1", "veihd_id_1", "veihd_dt_1", "death_dt_3", "deathsk_id_3",
                          "verscich_id_3", "vers_dt_3", "veihd_id_3", "veihd_dt_3", "ivtpath_id", "ivtpa_dt", 
                          "ivtpah_nm", "ivtpam_nm", "ivtpamg_nm", "toastu_id", "thd_id", "trm_id", "trmen_id", 
                          "trmop_id", "com_id", "offd_dt", "offre_dt", "omad_id", "smcp_id", "thishy_id", 
                          "thisdi_id", "det_id", "death_id_3"],
                          axis = 1)

#icase_id, idcase_id, icd_id, off_id
nominal_features = ["edu_id", "pro_id", "opc_id", "toast_id", "offdt_id", "gender_tx", "hd_id", "pcva_id", 
                    "pcvaci_id", "pcvach_id", "po_id", "ur_id", "sm_id", "ptia_id", "hc_id", "hcht_id", 
                    "hchc_id", "ht_id", "dm_id", "pad_id", "al_id", "ca_id", "fahiid_parents_1", 
                    "fahiid_parents_2", "fahiid_parents_3", "fahiid_parents_4", "fahiid_brsi_1", 
                    "fahiid_brsi_2", "fahiid_brsi_3", "fahiid_brsi_4"]

ordinal_features = ["mrs_tx_1", "mrs_tx_3", "gcse_nm", "gcsv_nm", "gcsm_nm", "discharged_mrs"]

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

barthel = ["feeding", "transfers", "bathing", "toilet_use", "grooming", "mobility", "stairs", "dressing", 
           "bowel_control", "bladder_control"]

nihss_in = ["nihs_1a_in", "nihs_1b_in", "nihs_1c_in", "nihs_2_in", "nihs_3_in", "nihs_4_in", "nihs_5al_in", 
            "nihs_5br_in", "nihs_6al_in", "nihs_6br_in", "nihs_7_in", "nihs_8_in", "nihs_9_in", "nihs_10_in", 
            "nihs_11_in"]

nihss_out = ["nihs_1a_out", "nihs_1b_out", "nihs_1c_out", "nihs_2_out", "nihs_3_out", 
            "nihs_4_out", "nihs_5al_out", "nihs_5br_out", "nihs_6al_out", "nihs_6br_out", "nihs_7_out", 
            "nihs_8_out", "nihs_9_out", "nihs_10_out", "nihs_11_out"]

hour = ["onseth_nm", "ottih_nm", "flookh_nm",  "fcth_nm", "nihsinh_nm", "nihsoth_nm", "cth_nm",  "mrih_nm"]

minute = ["onsetm_nm", "ottim_nm", "flookm_nm", "fctm_nm", "nihsinm_nm", "nihsotm_nm", "ctm_nm", "mrim_nm"]

date = ["rfur_dt_1", "rfur_dt_3", "ih_dt", "oh_dt", "onset_dt", "ot_dt", "flook_dt", "fct_dt", "nihsin_dt",
        "nihsot_dt", "ct_dt", "mri_dt"]

#nominal_features
for i in tsr_all3[nominal_features]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")

tsr_all3["edu_id"][(tsr_all3["edu_id"] != 98) & (tsr_all3["edu_id"] != 1) & (tsr_all3["edu_id"] != 2) & (tsr_all3["edu_id"] != 3) & (tsr_all3["edu_id"] != 4) & (tsr_all3["edu_id"] != 5) & (tsr_all3["edu_id"] != 6)]  = np.nan
tsr_all3["pro_id"][(tsr_all3["pro_id"] != 1) & (tsr_all3["pro_id"] != 2) & (tsr_all3["pro_id"] != 3) & (tsr_all3["pro_id"] != 4) & (tsr_all3["pro_id"] != 5) & (tsr_all3["pro_id"] != 6) & (tsr_all3["pro_id"] != 7) & (tsr_all3["pro_id"] != 8) & (tsr_all3["pro_id"] != 9) & (tsr_all3["pro_id"] != 10) & (tsr_all3["pro_id"] != 98) & (tsr_all3["pro_id"] != 99)]  = np.nan
tsr_all3["opc_id"][(tsr_all3["opc_id"] != 1) & (tsr_all3["opc_id"] != 2) & (tsr_all3["opc_id"] != 3)]  = np.nan
tsr_all3["toast_id"][(tsr_all3["toast_id"] != 1) & (tsr_all3["toast_id"] != 2) & (tsr_all3["toast_id"] != 3) & (tsr_all3["toast_id"] != 4) & (tsr_all3["toast_id"] != 5)]  = np.nan
tsr_all3["offdt_id"][(tsr_all3["offdt_id"] != 1) & (tsr_all3["offdt_id"] != 2) & (tsr_all3["offdt_id"] != 3) & (tsr_all3["offdt_id"] != 4) & (tsr_all3["offdt_id"] != 5)]  = np.nan
tsr_all3["gender_tx"][(tsr_all3["gender_tx"] != 1) & (tsr_all3["gender_tx"] != 0)]  = np.nan

for i in tsr_all3.loc[ : , "hd_id":"ca_id"]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")
    tsr_all3[i][(tsr_all3[i] != 0) & (tsr_all3[i] != 1) & (tsr_all3[i] != 2)] = np.nan
    
for i in tsr_all3.loc[ : , "fahiid_parents_1":"fahiid_parents_4"]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")
    tsr_all3[i][(tsr_all3[i] != 0) & (tsr_all3[i] != 1) & (tsr_all3[i] != 2)] = np.nan
    
for i in tsr_all3.loc[ : , "fahiid_brsi_1":"fahiid_brsi_4"]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")
    tsr_all3[i][(tsr_all3[i] != 0) & (tsr_all3[i] != 1) & (tsr_all3[i] != 2) & (tsr_all3[i] != 9)] = np.nan

#ordinal_features
for i in tsr_all3[ordinal_features]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")

tsr_all3["mrs_tx_1"][(tsr_all3["mrs_tx_1"] != 0) & (tsr_all3["mrs_tx_1"] != 1) & (tsr_all3["mrs_tx_1"] != 2) & (tsr_all3["mrs_tx_1"] != 3) & (tsr_all3["mrs_tx_1"] != 4) & (tsr_all3["mrs_tx_1"] != 5) & (tsr_all3["mrs_tx_1"] != 6)]  = np.nan
tsr_all3["mrs_tx_3"][(tsr_all3["mrs_tx_3"] != 0) & (tsr_all3["mrs_tx_3"] != 1) & (tsr_all3["mrs_tx_3"] != 2) & (tsr_all3["mrs_tx_3"] != 3) & (tsr_all3["mrs_tx_3"] != 4) & (tsr_all3["mrs_tx_3"] != 5) & (tsr_all3["mrs_tx_3"] != 6) & (tsr_all3["mrs_tx_3"] != 9)]  = np.nan
tsr_all3["gcse_nm"][(tsr_all3["gcse_nm"] != 1) & (tsr_all3["gcse_nm"] != 2) & (tsr_all3["gcse_nm"] != 3) & (tsr_all3["gcse_nm"] != 4)]  = np.nan
tsr_all3["gcsv_nm"][(tsr_all3["gcsv_nm"] != 1) & (tsr_all3["gcsv_nm"] != 2) & (tsr_all3["gcsv_nm"] != 3) & (tsr_all3["gcsv_nm"] != 4) & (tsr_all3["gcsv_nm"] != 5)]  = np.nan
tsr_all3["gcsm_nm"][(tsr_all3["gcsm_nm"] != 1) & (tsr_all3["gcsm_nm"] != 2) & (tsr_all3["gcsm_nm"] != 3) & (tsr_all3["gcsm_nm"] != 4) & (tsr_all3["gcsm_nm"] != 5) & (tsr_all3["gcsm_nm"] != 6)]  = np.nan
tsr_all3["discharged_mrs"][(tsr_all3["discharged_mrs"] != 0) & (tsr_all3["discharged_mrs"] != 1) & (tsr_all3["discharged_mrs"] != 2) & (tsr_all3["discharged_mrs"] != 3) & (tsr_all3["discharged_mrs"] != 4) & (tsr_all3["discharged_mrs"] != 5) & (tsr_all3["discharged_mrs"] != 6)]  = np.nan

#boolean
for i in tsr_all3[boolean]:
    tsr_all3[i].replace("1",1, inplace=True)
    tsr_all3[i].replace("0",0, inplace=True)
    tsr_all3[i].replace("Y",1, inplace=True)
    tsr_all3[i].replace("N",0, inplace=True)
    tsr_all3[i][(tsr_all3[i] != 1) & (tsr_all3[i] != 0)] = np.nan
    
#continuous
for i in tsr_all3[continuous]:
    q1 = tsr_all3[i].quantile(0.25)
    q3 = tsr_all3[i].quantile(0.75)
    iqr = q3 - q1
    inner_fence = 1.5 * iqr

    inner_fence_low = q1 - inner_fence
    inner_fence_upp = q3 + inner_fence
    tsr_all3[i][(tsr_all3[i] < inner_fence_low) | (tsr_all3[i] > inner_fence_upp)] = np.nan
    #choose one imputation
    tsr_all3[i] = tsr_all3[i].fillna(tsr_all3[i].mean())

tsr_all3[continuous] = tsr_all3[continuous].fillna(9999)
    
#barthel
for i in tsr_all3[barthel]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")
    
tsr_all3["feeding"][(tsr_all3["feeding"] < 0) | (tsr_all3["feeding"] > 10)] = np.nan
tsr_all3["transfers"][(tsr_all3["transfers"] < 0) | (tsr_all3["transfers"] > 15)] = np.nan
tsr_all3["bathing"][(tsr_all3["bathing"] < 0) | (tsr_all3["bathing"] > 5)] = np.nan
tsr_all3["toilet_use"][(tsr_all3["toilet_use"] < 0) | (tsr_all3["toilet_use"] > 10)] = np.nan
tsr_all3["grooming"][(tsr_all3["grooming"] < 0) | (tsr_all3["grooming"] > 5)] = np.nan
tsr_all3["mobility"][(tsr_all3["mobility"] < 0) | (tsr_all3["mobility"] > 15)] = np.nan
tsr_all3["stairs"][(tsr_all3["stairs"] < 0) | (tsr_all3["stairs"] > 10)] = np.nan
tsr_all3["dressing"][(tsr_all3["dressing"] < 0) | (tsr_all3["dressing"] > 10)] = np.nan
tsr_all3["bowel_control"][(tsr_all3["bowel_control"] < 0) | (tsr_all3["bowel_control"] > 10)] = np.nan
tsr_all3["bladder_control"][(tsr_all3["bladder_control"] < 0) | (tsr_all3["bladder_control"] > 10)] = np.nan

#choose one imputation
for i in tsr_all3[barthel]:
    tsr_all3[i] = tsr_all3[i].fillna(tsr_all3[i].mode()[0])

tsr_all3[barthel] = tsr_all3[barthel].fillna(9999)

## total scores of barthel

#nihss_in
for i in tsr_all3[nihss_in]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")

tsr_all3["nihs_1a_in"][(tsr_all3["nihs_1a_in"] < 0) | (tsr_all3["nihs_1a_in"] > 3)] = np.nan
tsr_all3["nihs_1b_in"][(tsr_all3["nihs_1b_in"] < 0) | (tsr_all3["nihs_1b_in"] > 2)] = np.nan
tsr_all3["nihs_1c_in"][(tsr_all3["nihs_1c_in"] < 0) | (tsr_all3["nihs_1c_in"] > 2)] = np.nan
tsr_all3["nihs_2_in"][(tsr_all3["nihs_2_in"] < 0) | (tsr_all3["nihs_2_in"] > 2)] = np.nan
tsr_all3["nihs_3_in"][(tsr_all3["nihs_3_in"] < 0) | (tsr_all3["nihs_3_in"] > 3)] = np.nan
tsr_all3["nihs_4_in"][(tsr_all3["nihs_4_in"] < 0) | (tsr_all3["nihs_4_in"] > 3)] = np.nan
tsr_all3["nihs_5al_in"][(tsr_all3["nihs_5al_in"] < 0) | (tsr_all3["nihs_5al_in"] > 4)] = np.nan
tsr_all3["nihs_5br_in"][(tsr_all3["nihs_5br_in"] < 0) | (tsr_all3["nihs_5br_in"] > 4)] = np.nan
tsr_all3["nihs_6al_in"][(tsr_all3["nihs_6al_in"] < 0) | (tsr_all3["nihs_6al_in"] > 4)] = np.nan
tsr_all3["nihs_6br_in"][(tsr_all3["nihs_6br_in"] < 0) | (tsr_all3["nihs_6br_in"] > 4)] = np.nan
tsr_all3["nihs_7_in"][(tsr_all3["nihs_7_in"] < 0) | (tsr_all3["nihs_7_in"] > 2)] = np.nan
tsr_all3["nihs_8_in"][(tsr_all3["nihs_8_in"] < 0) | (tsr_all3["nihs_8_in"] > 2)] = np.nan
tsr_all3["nihs_9_in"][(tsr_all3["nihs_9_in"] < 0) | (tsr_all3["nihs_9_in"] > 3)] = np.nan
tsr_all3["nihs_10_in"][(tsr_all3["nihs_10_in"] < 0) | (tsr_all3["nihs_10_in"] > 2)] = np.nan
tsr_all3["nihs_11_in"][(tsr_all3["nihs_11_in"] < 0) | (tsr_all3["nihs_11_in"] > 2)] = np.nan

#choose one imputation
for i in tsr_all3[nihss_in]:
    tsr_all3[i] = tsr_all3[i].fillna(tsr_all3[i].mode()[0])

tsr_all3[nihss_in] = tsr_all3[nihss_in].fillna(9999)

## total scores of nihss_in

#nihss_out
for i in tsr_all3[nihss_out]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")

tsr_all3["nihs_1a_out"][(tsr_all3["nihs_1a_out"] < 0) | (tsr_all3["nihs_1a_out"] > 3)] = np.nan
tsr_all3["nihs_1b_out"][(tsr_all3["nihs_1b_out"] < 0) | (tsr_all3["nihs_1b_out"] > 2)] = np.nan
tsr_all3["nihs_1c_out"][(tsr_all3["nihs_1c_out"] < 0) | (tsr_all3["nihs_1c_out"] > 2)] = np.nan
tsr_all3["nihs_2_out"][(tsr_all3["nihs_2_out"] < 0) | (tsr_all3["nihs_2_out"] > 2)] = np.nan
tsr_all3["nihs_3_out"][(tsr_all3["nihs_3_out"] < 0) | (tsr_all3["nihs_3_out"] > 3)] = np.nan
tsr_all3["nihs_4_out"][(tsr_all3["nihs_4_out"] < 0) | (tsr_all3["nihs_4_out"] > 3)] = np.nan
tsr_all3["nihs_5al_out"][(tsr_all3["nihs_5al_out"] < 0) | (tsr_all3["nihs_5al_out"] > 4)] = np.nan
tsr_all3["nihs_5br_out"][(tsr_all3["nihs_5br_out"] < 0) | (tsr_all3["nihs_5br_out"] > 4)] = np.nan
tsr_all3["nihs_6al_out"][(tsr_all3["nihs_6al_out"] < 0) | (tsr_all3["nihs_6al_out"] > 4)] = np.nan
tsr_all3["nihs_6br_out"][(tsr_all3["nihs_6br_out"] < 0) | (tsr_all3["nihs_6br_out"] > 4)] = np.nan
tsr_all3["nihs_7_out"][(tsr_all3["nihs_7_out"] < 0) | (tsr_all3["nihs_7_out"] > 2)] = np.nan
tsr_all3["nihs_8_out"][(tsr_all3["nihs_8_out"] < 0) | (tsr_all3["nihs_8_out"] > 2)] = np.nan
tsr_all3["nihs_9_out"][(tsr_all3["nihs_9_out"] < 0) | (tsr_all3["nihs_9_out"] > 3)] = np.nan
tsr_all3["nihs_10_out"][(tsr_all3["nihs_10_out"] < 0) | (tsr_all3["nihs_10_out"] > 2)] = np.nan
tsr_all3["nihs_11_out"][(tsr_all3["nihs_11_out"] < 0) | (tsr_all3["nihs_11_out"] > 2)] = np.nan

#choose one imputation
for i in tsr_all3[nihss_out]:
    tsr_all3[i] = tsr_all3[i].fillna(tsr_all3[i].mode()[0])

tsr_all3[nihss_out] = tsr_all3[nihss_out].fillna(9999)

## total scores of nihss_out

#hour & minute
for i in tsr_all3[hour]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")
    tsr_all3[i][(tsr_all3[i] < 0) | (tsr_all3[i] > 24)] = np.nan
    tsr_all3[i].replace(24, 0, inplace=True)

tsr_all3[hour] = tsr_all3[hour].fillna(9999)

for i in tsr_all3[minute]:
    tsr_all3[i] = pd.to_numeric(tsr_all3[i], errors = "coerce")
    tsr_all3[i][(tsr_all3[i] < 0) | (tsr_all3[i] > 60)] = np.nan
    tsr_all3[i].replace(60, 0, inplace=True)

tsr_all3[minute] = tsr_all3[minute].fillna(9999)

# date
for i in tsr_all3[date]:
    tsr_all3[i] = pd.to_datetime(tsr_all3[i], errors="coerce", format="%Y-%m-%d")
    tsr_all3[i][(tsr_all3[i].dt.year < 2006) | (tsr_all3[i].dt.year > 2021)] = np.nan
    tsr_all3[i] = tsr_all3[i].fillna(tsr_all3[i].mode()[0])

#SAVE FILE
#TSR_ALL3_TIDY = pd.concat([tsr_all3, nominal_onehot], axis = 1)
TSR_ALL3_TIDY = tsr_all3.dropna()

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3_TIDY.csv")
TSR_ALL3_TIDY.to_csv(csv_save, index=False)
