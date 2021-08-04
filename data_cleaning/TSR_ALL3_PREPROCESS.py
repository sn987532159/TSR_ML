import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def ischemic_stroke_cases(df):
    df = df[(df["icd_id"] == 1) | (df["icd_id"] == 2)]
    print(df.shape)
    return df

def remove_unrelated_features(df, ur_f):
    df = df.drop(ur_f, axis=1)
    print(df.shape)
    return df

def remove_timestamp_features(df, d, h, m):
    df = df.drop(d + h + m, axis=1)
    print(df.shape)
    return df

def categorical_features(df, nom_f, ord_f, bl_f, b_i, ni_in, ni_out):
    # nominal_features
    df["gender_tx"][df["gender_tx"] == "M"] = 1
    df["gender_tx"][df["gender_tx"] == "F"] = 0
    for i in df[nom_f]:
        df[i] = pd.to_numeric(df[i], errors="coerce")
    df["opc_id"][(df["opc_id"] != 1) & (df["opc_id"] != 2) & (df["opc_id"] != 3)] = np.nan
    df["toast_id"][(df["toast_id"] != 1) & (df["toast_id"] != 2) & (df["toast_id"] != 3) & (
            df["toast_id"] != 4) & (df["toast_id"] != 5)] = np.nan
    df["offdt_id"][(df["offdt_id"] != 1) & (df["offdt_id"] != 2) & (df["offdt_id"] != 3) & (
            df["offdt_id"] != 4) & (df["offdt_id"] != 5)] = np.nan
    df["gender_tx"][(df["gender_tx"] != 1) & (df["gender_tx"] != 0)] = np.nan

    for i in df.loc[:, "hd_id":"ca_id"]:
        df[i] = pd.to_numeric(df[i], errors="coerce")
        df[i][(df[i] != 0) & (df[i] != 1) & (df[i] != 2)] = np.nan

    for i in df.loc[:, "fahiid_parents_1":"fahiid_parents_4"]:
        df[i] = pd.to_numeric(df[i], errors="coerce")
        df[i][(df[i] != 0) & (df[i] != 1) & (df[i] != 2)] = np.nan

    for i in df.loc[:, "fahiid_brsi_1":"fahiid_brsi_4"]:
        df[i] = pd.to_numeric(df[i], errors="coerce")
        df[i][(df[i] != 0) & (df[i] != 1) & (df[i] != 2) & (df[i] != 9)] = np.nan

    # ordinal_features
    for i in df[ord_f]:
        df[i] = pd.to_numeric(df[i], errors="coerce")

    df["mrs_tx_1"][(df["mrs_tx_1"] != 0) & (df["mrs_tx_1"] != 1) & (df["mrs_tx_1"] != 2) & (
            df["mrs_tx_1"] != 3) & (df["mrs_tx_1"] != 4) & (df["mrs_tx_1"] != 5) & (
                           df["mrs_tx_1"] != 6) & (df["mrs_tx_1"] != 9)] = np.nan
    df["gcse_nm"][(df["gcse_nm"] != 1) & (df["gcse_nm"] != 2) & (df["gcse_nm"] != 3) & (
            df["gcse_nm"] != 4)] = np.nan
    df["gcsv_nm"][(df["gcsv_nm"] != 1) & (df["gcsv_nm"] != 2) & (df["gcsv_nm"] != 3) & (
            df["gcsv_nm"] != 4) & (df["gcsv_nm"] != 5)] = np.nan
    df["gcsm_nm"][(df["gcsm_nm"] != 1) & (df["gcsm_nm"] != 2) & (df["gcsm_nm"] != 3) & (
            df["gcsm_nm"] != 4) & (df["gcsm_nm"] != 5) & (df["gcsm_nm"] != 6)] = np.nan
    df["discharged_mrs"][
        (df["discharged_mrs"] != 0) & (df["discharged_mrs"] != 1) & (df["discharged_mrs"] != 2) & (
                df["discharged_mrs"] != 3) & (df["discharged_mrs"] != 4) & (
                df["discharged_mrs"] != 5) & (df["discharged_mrs"] != 6)] = np.nan

    # boolean
    for i in df[bl_f]:
        df[i].replace("1", 1, inplace=True)
        df[i].replace("0", 0, inplace=True)
        df[i].replace("Y", 1, inplace=True)
        df[i].replace("N", 0, inplace=True)
        df[i][(df[i] != 1) & (df[i] != 0)] = np.nan

    # barthel
    for i in df[b_i]:
        df[i] = pd.to_numeric(df[i], errors="coerce")

    df["feeding"][(df["feeding"] != 0) & (df["feeding"] != 5) & (df["feeding"] != 10)] = np.nan
    df["transfers"][(df["transfers"] != 0) & (df["transfers"] != 5) & (df["transfers"] != 10) & (
            df["transfers"] != 15)] = np.nan
    df["bathing"][(df["bathing"] != 0) & (df["bathing"] != 5)] = np.nan
    df["toilet_use"][(df["toilet_use"] != 0) & (df["toilet_use"] != 5) & (df["toilet_use"] != 10)] = np.nan
    df["grooming"][(df["grooming"] != 0) & (df["grooming"] != 5)] = np.nan
    df["mobility"][
        (df["mobility"] != 0) & (df["mobility"] != 5) & (df["mobility"] != 10) & (df["mobility"] != 15)] = np.nan
    df["stairs"][(df["stairs"] != 0) & (df["stairs"] != 5) & (df["stairs"] != 10)] = np.nan
    df["dressing"][(df["dressing"] != 0) & (df["dressing"] != 5) & (df["dressing"] != 10)] = np.nan
    df["bowel_control"][
        (df["bowel_control"] != 0) & (df["bowel_control"] != 5) & (df["bowel_control"] != 10)] = np.nan
    df["bladder_control"][
        (df["bladder_control"] != 0) & (df["bladder_control"] != 5) & (df["bladder_control"] != 10)] = np.nan

    # nihss_in
    for i in df[ni_in]:
        df[i] = pd.to_numeric(df[i], errors="coerce")

    df["nihs_1a_in"][(df["nihs_1a_in"] < 0) | (df["nihs_1a_in"] > 3)] = np.nan
    df["nihs_1b_in"][(df["nihs_1b_in"] < 0) | (df["nihs_1b_in"] > 2)] = np.nan
    df["nihs_1c_in"][(df["nihs_1c_in"] < 0) | (df["nihs_1c_in"] > 2)] = np.nan
    df["nihs_2_in"][(df["nihs_2_in"] < 0) | (df["nihs_2_in"] > 2)] = np.nan
    df["nihs_3_in"][(df["nihs_3_in"] < 0) | (df["nihs_3_in"] > 3)] = np.nan
    df["nihs_4_in"][(df["nihs_4_in"] < 0) | (df["nihs_4_in"] > 3)] = np.nan
    df["nihs_5al_in"][(df["nihs_5al_in"] < 0) | (df["nihs_5al_in"] > 4)] = np.nan
    df["nihs_5br_in"][(df["nihs_5br_in"] < 0) | (df["nihs_5br_in"] > 4)] = np.nan
    df["nihs_6al_in"][(df["nihs_6al_in"] < 0) | (df["nihs_6al_in"] > 4)] = np.nan
    df["nihs_6br_in"][(df["nihs_6br_in"] < 0) | (df["nihs_6br_in"] > 4)] = np.nan
    df["nihs_7_in"][(df["nihs_7_in"] < 0) | (df["nihs_7_in"] > 2)] = np.nan
    df["nihs_8_in"][(df["nihs_8_in"] < 0) | (df["nihs_8_in"] > 2)] = np.nan
    df["nihs_9_in"][(df["nihs_9_in"] < 0) | (df["nihs_9_in"] > 3)] = np.nan
    df["nihs_10_in"][(df["nihs_10_in"] < 0) | (df["nihs_10_in"] > 2)] = np.nan
    df["nihs_11_in"][(df["nihs_11_in"] < 0) | (df["nihs_11_in"] > 2)] = np.nan

    # nihss_out
    for i in df[ni_out]:
        df[i] = pd.to_numeric(df[i], errors="coerce")

    df["nihs_1a_out"][(df["nihs_1a_out"] < 0) | (df["nihs_1a_out"] > 3)] = np.nan
    df["nihs_1b_out"][(df["nihs_1b_out"] < 0) | (df["nihs_1b_out"] > 2)] = np.nan
    df["nihs_1c_out"][(df["nihs_1c_out"] < 0) | (df["nihs_1c_out"] > 2)] = np.nan
    df["nihs_2_out"][(df["nihs_2_out"] < 0) | (df["nihs_2_out"] > 2)] = np.nan
    df["nihs_3_out"][(df["nihs_3_out"] < 0) | (df["nihs_3_out"] > 3)] = np.nan
    df["nihs_4_out"][(df["nihs_4_out"] < 0) | (df["nihs_4_out"] > 3)] = np.nan
    df["nihs_5al_out"][(df["nihs_5al_out"] < 0) | (df["nihs_5al_out"] > 4)] = np.nan
    df["nihs_5br_out"][(df["nihs_5br_out"] < 0) | (df["nihs_5br_out"] > 4)] = np.nan
    df["nihs_6al_out"][(df["nihs_6al_out"] < 0) | (df["nihs_6al_out"] > 4)] = np.nan
    df["nihs_6br_out"][(df["nihs_6br_out"] < 0) | (df["nihs_6br_out"] > 4)] = np.nan
    df["nihs_7_out"][(df["nihs_7_out"] < 0) | (df["nihs_7_out"] > 2)] = np.nan
    df["nihs_8_out"][(df["nihs_8_out"] < 0) | (df["nihs_8_out"] > 2)] = np.nan
    df["nihs_9_out"][(df["nihs_9_out"] < 0) | (df["nihs_9_out"] > 3)] = np.nan
    df["nihs_10_out"][(df["nihs_10_out"] < 0) | (df["nihs_10_out"] > 2)] = np.nan
    df["nihs_11_out"][(df["nihs_11_out"] < 0) | (df["nihs_11_out"] > 2)] = np.nan

    print(df.shape)
    return df

def continuous_features(df, cont):
    # continuous
    for i in df[cont]:
        df[i][df[i] == 999.9] = np.nan
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 - q1
        inner_fence = 1.5 * iqr

        inner_fence_low = q1 - inner_fence
        inner_fence_upp = q3 + inner_fence
        df[i][(df[i] < inner_fence_low) | (df[i] > inner_fence_upp)] = np.nan
        df[i][df[i] < 0] = np.nan

    print(df.shape)
    return df

def remove_high_missing_features(df):
    missing_ratio = df.isnull().sum() / len(df) * 100
    missing_ratio_index = missing_ratio[missing_ratio > 40].index
    df = df.drop(missing_ratio_index, axis=1)
    df = df.sort_values(by=["icase_id", "idcase_id"])
    print(df.shape)
    return df

def delete_error_cases(df):
    df_1 = df[["icase_id", "idcase_id", "feeding", "transfers", "bathing", "toilet_use", "grooming", "mobility",
               "stairs", "dressing", "bowel_control", "bladder_control", "nihs_1a_out", "nihs_1b_out", "nihs_1c_out",
               "nihs_2_out", "nihs_3_out", "nihs_4_out", "nihs_5al_out", "nihs_5br_out", "nihs_6al_out", "nihs_6br_out",
               "nihs_7_out", "nihs_8_out", "nihs_9_out", "nihs_10_out", "nihs_11_out", "discharged_mrs"]]
    print(df_1.shape)

    df_1[
        "bi_total"] = df_1.feeding + df_1.transfers + df_1.bathing + df_1.toilet_use + df_1.grooming + df_1.mobility + df_1.stairs + df_1.dressing + df_1.bowel_control + df_1.bladder_control
    df_1[
        "nihss_total"] = df_1.nihs_1a_out + df_1.nihs_1b_out + df_1.nihs_1c_out + df_1.nihs_2_out + df_1.nihs_3_out + df_1.nihs_4_out + df_1.nihs_5al_out + df_1.nihs_5br_out + df_1.nihs_6al_out + df_1.nihs_6br_out + df_1.nihs_7_out + df_1.nihs_8_out + df_1.nihs_9_out + df_1.nihs_10_out + df_1.nihs_11_out

    df_1["index"] = range(0, len(df_1), 1)
    df_1.set_index("index", inplace=True)
    all_0_index = df_1[(df_1["discharged_mrs"] == 0) & (df_1["bi_total"] == 0) & (df_1["nihss_total"] == 0)].index
    df_1 = df_1.drop(df_1.index[all_0_index])
    print(df_1.shape)

    df_1["index"] = range(0, len(df_1), 1)
    df_1.set_index("index", inplace=True)
    db_0_index = df_1[(df_1["discharged_mrs"] == 0) & (df_1["bi_total"] == 0)].index
    df_1 = df_1.drop(df_1.index[db_0_index])
    print(df_1.shape)

    df_1["index"] = range(0, len(df_1), 1)
    df_1.set_index("index", inplace=True)
    bn_0_index = df_1[(df_1["bi_total"] == 0) & (df_1["nihss_total"] == 0)].index
    df_1 = df_1.drop(df_1.index[bn_0_index])
    print(df_1.shape)

    df_1 = df_1.dropna()
    print(df_1.shape)
    df_1["index"] = range(1, len(df_1) + 1, 1)
    df_1.set_index("index", inplace=True)

    return df_1

def outlier_detection(df, var1, var2):
    outlier_index = []
    for i in set(df[var1]):
        selected_df = df[df[var1] == i]
        selected_df_mean = selected_df[var2].mean()
        selected_df_sd = selected_df[var2].std()
        selected_df_del_index = selected_df[(selected_df[var2] < selected_df_mean - 2 * selected_df_sd) | (
                    selected_df[var2] > selected_df_mean + 2 * selected_df_sd)].index.values.tolist()
        outlier_index = outlier_index + selected_df_del_index
    print(len(outlier_index))
    return outlier_index


if __name__ == '__main__':
    # Grouping Features
    # "icase_id", "idcase_id"
    unrelated_features = ["edu_id", "pro_id", "icd_id", "off_id", "fstatus_id_1", "location_id_1", "torg_id_1",
                          "flu_id_1", "fluorg_id_1", "fluorg_tx_1", "fluresult_tx_1", "death_dt_1", "death_id_1",
                          "deathsk_id_1", "deatho_tx_1", "veihdorg_id_1", "versorg_id_1", "torg_tx_1", "versorg_tx_1",
                          "veihdorg_tx_1", "fstatus_id_3", "location_id_3", "torg_id_3", "flu_id_3", "fluorg_id_3",
                          "fluorg_tx_3", "fluresult_tx_3", "deatho_tx_3", "versorg_id_3", "veihdorg_id_3", "torg_tx_3",
                          "versorg_tx_3", "veihdorg_tx_3", "fstatus_id_6", "rfur_dt_6", "location_id_6", "torg_id_6",
                          "flu_id_6", "fluorg_id_6", "fluorg_tx_6", "fluresult_tx_6", "death_dt_6", "death_id_6",
                          "deathsk_id_6", "deatho_tx_6", "ve_id_6", "vers_fl_6", "verscich_id_6", "vers_dt_6",
                          "versorg_id_6", "veihd_fl_6", "veihd_id_6", "veihd_dt_6", "veihdorg_id_6", "mrs_tx_6",
                          "torg_tx_6", "versorg_tx_6", "veihdorg_tx_6", "fstatus_id_12", "rfur_dt_12", "location_id_12",
                          "torg_id_12", "flu_id_12", "fluorg_id_12", "fluorg_tx_12", "fluresult_tx_12", "death_dt_12",
                          "death_id_12", "deathsk_id_12", "deatho_tx_12", "ve_id_12", "vers_fl_12", "verscich_id_12",
                          "vers_dt_12", "versorg_id_12", "veihd_fl_12", "veihd_id_12", "veihd_dt_12", "veihdorg_id_12",
                          "mrs_tx_12", "torg_tx_12", "versorg_tx_12", "veihdorg_tx_12", "index", "iprotocol_id",
                          "icase_id.1", "idcase_id.1", "cstatus_id", "org_id", "dctype24_id", "patient_id", "input_nm",
                          "age_nm", "proot_tx", "itown_id", "addr_tx", "telh_tx", "telp_tx", "telf_tx", "ftitle_tx",
                          "casememo_tx", "ivtpath_fl", "ivtpaah_fl", "nivtpa99_tx", "icd_tx", "icdtia_id", "icdo_tx",
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
                          "nivtpa11_fl", "omad_fl", "dethoh_fl", "ecg_id", "mra_fl", "cta_fl", "dsa_fl", "cdr_id",
                          "cdl_id", "tccsr_id", "tccsl_id", "tccsba_id", "mcdr_id", "mcdl_id", "mcdba_id",
                          "mcdri_id", "mcdli_id", "vers_dt_1", "veihd_dt_1", "death_dt_3", "vers_dt_3", "veihd_dt_3",
                          "det_id"]

    date = ["rfur_dt_1", "rfur_dt_3", "oh_dt", "onset_dt", "ot_dt", "flook_dt", "fct_dt", "nihsin_dt",
            "nihsot_dt", "ct_dt", "mri_dt"]

    hour = ["onseth_nm", "ottih_nm", "flookh_nm", "fcth_nm", "nihsinh_nm", "nihsoth_nm", "cth_nm", "mrih_nm"]

    minute = ["onsetm_nm", "ottim_nm", "flookm_nm", "fctm_nm", "nihsinm_nm", "nihsotm_nm", "ctm_nm", "mrim_nm"]

    nominal_features = ["opc_id", "toast_id", "offdt_id", "gender_tx", "hd_id", "pcva_id",
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
                  "cre_nm", "ua_nm", "tcho_nm", "tg_nm", "hdl_nm", "alb_nm", "crp_nm", "hbac_nm", "ac_nm", "got_nm",
                  "ldl_nm", "gpt_nm", "age", "hospitalised_time"]

    continuous_n = ["height_nm", "weight_nm", "sbp_nm", "dbp_nm", "bt_nm", "hr_nm", "rr_nm", "hb_nm",
                  "hct_nm", "platelet_nm", "wbc_nm", "ptt1_nm", "ptt2_nm", "ptinr_nm", "er_nm", "bun_nm",
                  "cre_nm", "ua_nm", "tcho_nm", "tg_nm", "hdl_nm",
                  "ldl_nm", "gpt_nm", "age", "hospitalised_time"]

    barthel = ["feeding", "transfers", "bathing", "toilet_use", "grooming", "mobility", "stairs", "dressing",
               "bowel_control", "bladder_control"]

    nihss_in = ["nihs_1a_in", "nihs_1b_in", "nihs_1c_in", "nihs_2_in", "nihs_3_in", "nihs_4_in", "nihs_5al_in",
                "nihs_5br_in", "nihs_6al_in", "nihs_6br_in", "nihs_7_in", "nihs_8_in", "nihs_9_in", "nihs_10_in",
                "nihs_11_in"]

    nihss_out = ["nihs_1a_out", "nihs_1b_out", "nihs_1c_out", "nihs_2_out", "nihs_3_out",
                 "nihs_4_out", "nihs_5al_out", "nihs_5br_out", "nihs_6al_out", "nihs_6br_out", "nihs_7_out",
                 "nihs_8_out", "nihs_9_out", "nihs_10_out", "nihs_11_out"]

    # import data
    csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3.csv")
    TSR_ALL3_df = pd.read_csv(csv_path, low_memory=False)

    # pre_procesing
    TSR_ALL3_df1 = ischemic_stroke_cases(TSR_ALL3_df)
    TSR_ALL3_df2 = remove_unrelated_features(TSR_ALL3_df1, unrelated_features)
    TSR_ALL3_df3 = remove_timestamp_features(TSR_ALL3_df2, date, hour, minute)
    TSR_ALL3_df4 = categorical_features(TSR_ALL3_df3, nominal_features, ordinal_features, boolean, barthel, nihss_in, nihss_out)
    TSR_ALL3_df5 = continuous_features(TSR_ALL3_df4, continuous)
    TSR_ALL3_df6 = remove_high_missing_features(TSR_ALL3_df5)

    # save pre_processed dataset
    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_preprocessed.csv")
    TSR_ALL3_df6.to_csv(csv_save, index=False)

    # delete error cases
    TSR_ALL3_score_df = delete_error_cases(TSR_ALL3_df6)

    # save error cases deleted dataset
    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_score.csv")
    TSR_ALL3_score_df.to_csv(csv_save, index=False)

    # mRS & BI * NIHSS outlier detection
    TSR_ALL3_score_df = TSR_ALL3_score_df.reset_index(drop=True)
    db_outlier = outlier_detection(TSR_ALL3_score_df, "discharged_mrs", "bi_total")
    dn_outlier = outlier_detection(TSR_ALL3_score_df, "discharged_mrs", "nihss_total")
    #bn_outlier = outlier_detection(TSR_ALL3_score_df, "bi_total", "nihss_total")

    # delete_union = len(set(db_outlier) & set(dn_outlier) & set(bn_outlier))
    # delete_intersection = len(set(db_outlier) | set(dn_outlier) | set(bn_outlier))
    # delete_union_num = len(set(db_outlier) & set(dn_outlier))
    # delete_intersection_num = len(set(db_outlier) | set(dn_outlier))
    delete_union_index = list(set(db_outlier) & set(dn_outlier))
    delete_intersection_index = list(set(db_outlier) | set(dn_outlier))

    TSR_ALL3_score_cleaned_df = TSR_ALL3_score_df.drop(TSR_ALL3_score_df.index[delete_intersection_index])

    # save outlier cases deleted dataset
    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_score_cleaned.csv")
    TSR_ALL3_score_cleaned_df.to_csv(csv_save, index=False)

    # merge TSR_ALL3_score_cleaned_df to the original dataset
    TSR_ALL3_AMPUTATED_DF = pd.merge(TSR_ALL3_df6, TSR_ALL3_score_cleaned_df.iloc[:, 0:2], on=["icase_id", "idcase_id"])
    print(TSR_ALL3_AMPUTATED_DF.shape)

    # sum(TSR_ALL3_AMPUTATED_DF["gender_tx"] == 1) / len(TSR_ALL3_AMPUTATED_DF["gender_tx"])  # male
    # TSR_ALL3_AMPUTATED_DF["age"].mean(), TSR_ALL3_AMPUTATED_DF["age"].std()  # age
    # TSR_ALL3_AMPUTATED_DF["discharged_mrs"].mean(), TSR_ALL3_AMPUTATED_DF["discharged_mrs"].std()  # Discharged mRS
    # TSR_ALL3_AMPUTATED_DF["mrs_tx_1"].mean(), TSR_ALL3_AMPUTATED_DF["mrs_tx_1"].std() # 1-month follow-up mRS
    # TSR_ALL3_AMPUTATED_DF.loc[:, "feeding" : "bladder_control"].sum(axis=1).mean(), TSR_ALL3_AMPUTATED_DF.loc[:, "feeding" : "bladder_control"].sum(axis=1).std()
    # TSR_ALL3_AMPUTATED_DF.loc[:, "nihs_1a_in": "nihs_11_in"].sum(axis=1).mean(), TSR_ALL3_AMPUTATED_DF.loc[:,"nihs_1a_in": "nihs_11_in"].sum(axis=1).std()
    # TSR_ALL3_AMPUTATED_DF.loc[:, "nihs_1a_out": "nihs_11_out"].sum(axis=1).mean(), TSR_ALL3_AMPUTATED_DF.loc[:,"nihs_1a_out": "nihs_11_out"].sum(axis=1).std()

    TSR_ALL3_AMPUTATED_DF[continuous_n] = TSR_ALL3_AMPUTATED_DF[continuous_n].fillna(9999)
    TSR_ALL3_AMPUTATED_DF = TSR_ALL3_AMPUTATED_DF.dropna()
    print(TSR_ALL3_AMPUTATED_DF.shape)

    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_AMPUTATED.csv")
    TSR_ALL3_AMPUTATED_DF.to_csv(csv_save, index=False)

    TSR_ALL3_AMPUTATED_DF["ih_dt"] = pd.to_datetime(TSR_ALL3_AMPUTATED_DF["ih_dt"], errors='coerce')
    TSR_ALL3_AMPUTATED_DF["ih_dt"][
        (TSR_ALL3_AMPUTATED_DF["ih_dt"].dt.year < 2006) | (TSR_ALL3_AMPUTATED_DF["ih_dt"].dt.year > 2020)] = np.nan

    TSR_ALL3_TRAIN = TSR_ALL3_AMPUTATED_DF[
        TSR_ALL3_AMPUTATED_DF["ih_dt"].dt.year.isin([2006, 2007, 2008, 2009, 2010, 2011])]
    TSR_ALL3_VALIDATION = TSR_ALL3_AMPUTATED_DF[
        TSR_ALL3_AMPUTATED_DF["ih_dt"].dt.year.isin([2012, 2013])]
    TSR_ALL3_TEST = TSR_ALL3_AMPUTATED_DF[
        ~TSR_ALL3_AMPUTATED_DF["ih_dt"].dt.year.isin([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013])]

    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_TRAIN.csv")
    TSR_ALL3_TRAIN.to_csv(csv_save, index=False)
    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_VALIDATION.csv")
    TSR_ALL3_VALIDATION.to_csv(csv_save, index=False)
    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_TEST.csv")
    TSR_ALL3_TEST.to_csv(csv_save, index=False)
