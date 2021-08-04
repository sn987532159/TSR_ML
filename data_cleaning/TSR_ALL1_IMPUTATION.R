install.packages("mice")
install.packages("VIM")
install.packages("Gmisc")
library(mice)
library(VIM)
library(Gmisc)
library(dplyr)

setwd("C:/Users/Jacky C/PycharmProjects/tsr_ml/data_cleaning")
file_path <- pathJoin("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TRAIN.csv")
TSR_ALL1_TRAIN <- read.csv(file_path)
TSR_ALL1_TRAIN_1 <- TSR_ALL1_TRAIN %>% select(-icase_id, -idcase_id, -ih_dt, -mrs_tx_1)
TSR_ALL1_TRAIN_1[TSR_ALL1_TRAIN_1 == 9999] = NA

file_path <- pathJoin("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_VALIDATION.csv")
TSR_ALL1_VALIDATION <- read.csv(file_path)
TSR_ALL1_VALIDATION_1 <- TSR_ALL1_VALIDATION %>% select(-icase_id, -idcase_id, -ih_dt, -mrs_tx_1)
TSR_ALL1_VALIDATION_1[TSR_ALL1_VALIDATION_1 == 9999] = NA

file_path <- pathJoin("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TEST.csv")
TSR_ALL1_TEST <- read.csv(file_path)
TSR_ALL1_TEST_1 <- TSR_ALL1_TEST %>% select(-icase_id, -idcase_id, -ih_dt, -mrs_tx_1)
TSR_ALL1_TEST_1[TSR_ALL1_TEST_1 == 9999] = NA

nominal_features = c("opc_id", "toast_id", "offdt_id", "gender_tx", "hd_id", "pcva_id", 
                     "pcvaci_id", "pcvach_id", "po_id", "ur_id", "sm_id", "ptia_id", "hc_id", "hcht_id", 
                     "hchc_id", "ht_id", "dm_id", "pad_id", "al_id", "ca_id", "fahiid_parents_1", 
                     "fahiid_parents_2", "fahiid_parents_3", "fahiid_parents_4", "fahiid_brsi_1", 
                     "fahiid_brsi_2", "fahiid_brsi_3", "fahiid_brsi_4")

ordinal_features = c("gcse_nm", "gcsv_nm", "gcsm_nm", "discharged_mrs")

boolean = c("toastle_fl", "toastli_fl", "toastsce_fl", "toastsmo_fl", "toastsra_fl", "toastsdi_fl", 
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
            "old_stroke_mrich")

continuous = c("height_nm", "weight_nm", "sbp_nm", "dbp_nm", "bt_nm", "hr_nm", "rr_nm", "hb_nm",
               "hct_nm", "platelet_nm", "wbc_nm", "ptt1_nm", "ptt2_nm", "ptinr_nm", "er_nm", "bun_nm", 
               "cre_nm", "ua_nm", "tcho_nm", "tg_nm", "hdl_nm", 
               "ldl_nm", "gpt_nm", "age", "hospitalised_time")

barthel = c("feeding", "transfers", "bathing", "toilet_use", "grooming", "mobility", "stairs", "dressing", 
            "bowel_control", "bladder_control")

nihss_in = c("nihs_1a_in", "nihs_1b_in", "nihs_1c_in", "nihs_2_in", "nihs_3_in", "nihs_4_in", "nihs_5al_in", 
             "nihs_5br_in", "nihs_6al_in", "nihs_6br_in", "nihs_7_in", "nihs_8_in", "nihs_9_in", "nihs_10_in", 
             "nihs_11_in")

nihss_out = c("nihs_1a_out", "nihs_1b_out", "nihs_1c_out", "nihs_2_out", "nihs_3_out", 
              "nihs_4_out", "nihs_5al_out", "nihs_5br_out", "nihs_6al_out", "nihs_6br_out", "nihs_7_out", 
              "nihs_8_out", "nihs_9_out", "nihs_10_out", "nihs_11_out")

TSR_ALL1_TRAIN_1[nominal_features] <- lapply(TSR_ALL1_TRAIN_1[nominal_features], as.factor)
TSR_ALL1_TRAIN_1[ordinal_features] <- lapply(TSR_ALL1_TRAIN_1[ordinal_features], as.factor)
TSR_ALL1_TRAIN_1[boolean] <- lapply(TSR_ALL1_TRAIN_1[boolean], as.factor)
TSR_ALL1_TRAIN_1[continuous] <- lapply(TSR_ALL1_TRAIN_1[continuous], as.numeric)
TSR_ALL1_TRAIN_1[barthel] <- lapply(TSR_ALL1_TRAIN_1[barthel], as.factor)
TSR_ALL1_TRAIN_1[nihss_in] <- lapply(TSR_ALL1_TRAIN_1[nihss_in], as.factor)
TSR_ALL1_TRAIN_1[nihss_out] <- lapply(TSR_ALL1_TRAIN_1[nihss_out], as.factor)

TSR_ALL1_VALIDATION_1[nominal_features] <- lapply(TSR_ALL1_VALIDATION_1[nominal_features], as.factor)
TSR_ALL1_VALIDATION_1[ordinal_features] <- lapply(TSR_ALL1_VALIDATION_1[ordinal_features], as.factor)
TSR_ALL1_VALIDATION_1[boolean] <- lapply(TSR_ALL1_VALIDATION_1[boolean], as.factor)
TSR_ALL1_VALIDATION_1[continuous] <- lapply(TSR_ALL1_VALIDATION_1[continuous], as.numeric)
TSR_ALL1_VALIDATION_1[barthel] <- lapply(TSR_ALL1_VALIDATION_1[barthel], as.factor)
TSR_ALL1_VALIDATION_1[nihss_in] <- lapply(TSR_ALL1_VALIDATION_1[nihss_in], as.factor)
TSR_ALL1_VALIDATION_1[nihss_out] <- lapply(TSR_ALL1_VALIDATION_1[nihss_out], as.factor)

TSR_ALL1_TEST_1[nominal_features] <- lapply(TSR_ALL1_TEST_1[nominal_features], as.factor)
TSR_ALL1_TEST_1[ordinal_features] <- lapply(TSR_ALL1_TEST_1[ordinal_features], as.factor)
TSR_ALL1_TEST_1[boolean] <- lapply(TSR_ALL1_TEST_1[boolean], as.factor)
TSR_ALL1_TEST_1[continuous] <- lapply(TSR_ALL1_TEST_1[continuous], as.numeric)
TSR_ALL1_TEST_1[barthel] <- lapply(TSR_ALL1_TEST_1[barthel], as.factor)
TSR_ALL1_TEST_1[nihss_in] <- lapply(TSR_ALL1_TEST_1[nihss_in], as.factor)
TSR_ALL1_TEST_1[nihss_out] <- lapply(TSR_ALL1_TEST_1[nihss_out], as.factor)

#mice imputation
methods_TSR_ALL1 <- c(height_nm = "pmm", weight_nm = "pmm", opc_id = "polyreg", gcse_nm = "polr", gcsv_nm = "polr", gcsm_nm = "polr", sbp_nm = "pmm", dbp_nm = "pmm", bt_nm = "pmm", hr_nm = "pmm", rr_nm = "pmm",
                      toast_id = "polyreg", toastle_fl = "logreg",toastli_fl = "logreg",toastsce_fl = "logreg",toastsmo_fl = "logreg",toastsra_fl = "logreg",toastsdi_fl = "logreg",toastsmi_fl = "logreg",toastsantip_fl = "logreg",toastsau_fl = "logreg", 
                      toastshy_fl = "logreg",toastspr_fl = "logreg",toastsantit_fl = "logreg",toastsho_fl = "logreg",toastshys_fl = "logreg",toastsca_fl = "logreg",thda_fl = "logreg",thdh_fl = "logreg",thdi_fl = "logreg",thdam_fl = "logreg",thdv_fl = "logreg",
                      thde_fl = "logreg",thdm_fl = "logreg",thdr_fl = "logreg",thdp_fl = "logreg",hb_nm = "pmm", hct_nm = "pmm", platelet_nm = "pmm", wbc_nm = "pmm", ptt1_nm = "pmm", ptt2_nm = "pmm", ptinr_nm = "pmm", er_nm = "pmm", bun_nm = "pmm", cre_nm = "pmm",
                      ua_nm = "pmm", tcho_nm = "pmm", tg_nm = "pmm", hdl_nm = "pmm", ldl_nm = "pmm", gpt_nm = "pmm", trman_fl = "logreg",trmas_fl = "logreg",trmti_fl = "logreg",trmhe_fl = "logreg",trmwa_fl = "logreg",
                      trmia_fl = "logreg",trmfo_fl = "logreg",trmta_fl = "logreg",trmsd_fl = "logreg",trmre_fl = "logreg",trmen_fl = "logreg",trmag_fl = "logreg",trmcl_fl = "logreg",trmpl_fl = "logreg",trmlm_fl = "logreg",trmiv_fl = "logreg",trmve_fl = "logreg",
                      trmng_fl = "logreg",trmdy_fl = "logreg",trmicu_fl = "logreg",trmsm_fl = "logreg",trmed_fl = "logreg",trmop_fl = "logreg",om_fl = "logreg",omas_fl = "logreg",omag_fl = "logreg",omti_fl = "logreg",omcl_fl = "logreg",omwa_fl = "logreg",ompl_fl = "logreg",omanh_fl = "logreg",
                      omand_fl = "logreg",omli_fl = "logreg",am_fl = "logreg",amas_fl = "logreg",amag_fl = "logreg",amti_fl = "logreg",amcl_fl = "logreg",amwa_fl = "logreg",ampl_fl = "logreg",amanh_fl = "logreg",amand_fl = "logreg",amli_fl = "logreg",compn_fl = "logreg",comut_fl = "logreg",
                      comug_fl = "logreg",compr_fl = "logreg",compu_fl = "logreg",comac_fl = "logreg",comse_fl = "logreg",comde_fl = "logreg",detst_fl = "logreg",dethe_fl = "logreg",detho_fl = "logreg",detha_fl = "logreg",detva_fl = "logreg",detre_fl = "logreg",
                      detme_fl = "logreg",offdt_id = "polyreg", ct_fl = "logreg",mri_fl = "logreg",ecgl_fl = "logreg",ecga_fl = "logreg",ecgq_fl = "logreg",feeding = "pmm", transfers = "pmm", bathing = "pmm", toilet_use = "pmm", grooming = "pmm", mobility = "pmm", stairs = "pmm", 
                      dressing = "pmm", bowel_control = "pmm", bladder_control = "pmm", discharged_mrs = "polr", cortical_aca_ctr = "logreg", cortical_mca_ctr = "logreg", subcortical_aca_ctr = "logreg",
                      subcortical_mca_ctr = "logreg", pca_cortex_ctr = "logreg", thalamus_ctr = "logreg", brainstem_ctr = "logreg", cerebellum_ctr = "logreg", watershed_ctr = "logreg", hemorrhagic_infarct_ctr = "logreg",
                      old_stroke_ctci = "logreg", cortical_aca_ctl = "logreg", cortical_mca_ctl = "logreg", subcortical_aca_ctl = "logreg", subcortical_mca_ctl = "logreg", pca_cortex_ctl = "logreg", thalamus_ctl = "logreg",
                      brainstem_ctl = "logreg", cerebellum_ctl = "logreg", watershed_ctl = "logreg", hemorrhagic_infarct_ctl = "logreg", old_stroke_ctch = "logreg", cortical_aca_mrir = "logreg", cortical_mca_mrir = "logreg",
                      subcortical_aca_mrir = "logreg", subcortical_mca_mrir = "logreg", pca_cortex_mrir = "logreg", thalamus_mrir = "logreg", brainstem_mrir = "logreg", cerebellum_mrir = "logreg", watershed_mrir = "logreg",
                      hemorrhagic_infarct_mrir = "logreg", old_stroke_mrici = "logreg", cortical_aca_mril = "logreg", cortical_mca_mril = "logreg", subcortical_aca_mril = "logreg", subcortical_mca_mril = "logreg",
                      pca_cortex_mril = "logreg", thalamus_mril = "logreg", brainstem_mril = "logreg", cerebellum_mril = "logreg", watershed_mril = "logreg", hemorrhagic_infarct_mril = "logreg", old_stroke_mrich = "logreg",
                      hd_id = "polyreg", pcva_id = "polyreg", pcvaci_id = "polyreg", pcvach_id = "polyreg", po_id = "polyreg", ur_id = "polyreg", sm_id = "polyreg", ptia_id = "polyreg", hc_id = "polyreg", hcht_id = "polyreg", hchc_id = "polyreg", ht_id = "polyreg", dm_id = "polyreg", pad_id = "polyreg", al_id = "polyreg", ca_id = "polyreg", fahiid_parents_1 = "polyreg", 
                      fahiid_parents_2 = "polyreg", fahiid_parents_3 = "polyreg", fahiid_parents_4 = "polyreg", fahiid_brsi_1 = "polyreg", fahiid_brsi_2 = "polyreg", fahiid_brsi_3 = "polyreg", fahiid_brsi_4 = "polyreg", nihs_1a_in = "pmm", nihs_1b_in = "pmm",
                      nihs_1c_in = "pmm", nihs_2_in = "pmm", nihs_3_in = "pmm", nihs_4_in = "pmm", nihs_5al_in = "pmm", nihs_5br_in = "pmm", nihs_6al_in = "pmm", nihs_6br_in = "pmm", nihs_7_in = "pmm", nihs_8_in = "pmm", nihs_9_in = "pmm", nihs_10_in = "pmm",
                      nihs_11_in = "pmm", nihs_1a_out = "pmm", nihs_1b_out = "pmm", nihs_1c_out = "pmm", nihs_2_out = "pmm", nihs_3_out = "pmm", nihs_4_out = "pmm", nihs_5al_out = "pmm", nihs_5br_out = "pmm", nihs_6al_out= "pmm",
                      nihs_6br_out = "pmm", nihs_7_out = "pmm", nihs_8_out = "pmm", nihs_9_out = "pmm", nihs_10_out = "pmm", nihs_11_out = "pmm", gender_tx = "polyreg", age = "pmm", hospitalised_time = "pmm")

TSR_ALL1_TRAIN_imp <- mice(TSR_ALL1_TRAIN_1, maxit =20, m = 5, method = methods_TSR_ALL1, print = TRUE, seed = 19)
TSR_ALL1_TRAIN_mice <- TSR_ALL1_TRAIN %>% select(icase_id, idcase_id, mrs_tx_1) %>% cbind(complete(TSR_ALL1_TRAIN_imp, 5))
save_file <- pathJoin("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TRAIN_MICE5.csv")
write.csv(TSR_ALL1_TRAIN_mice, save_file, row.names=FALSE)

TSR_ALL1_VALIDATION_imp<- mice.mids(TSR_ALL1_TRAIN_imp, TSR_ALL1_VALIDATION_1)
TSR_ALL1_VALIDATION_mice <- TSR_ALL1_VALIDATION %>% select(icase_id, idcase_id, mrs_tx_1) %>% cbind(complete(TSR_ALL1_VALIDATION_imp, 5))
save_file <- pathJoin("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_VALIDATION_MICE5.csv")
write.csv(TSR_ALL1_VALIDATION_mice, save_file, row.names=FALSE)

TSR_ALL1_TEST_imp<- mice.mids(TSR_ALL1_TRAIN_imp, TSR_ALL1_TEST_1)
TSR_ALL1_TEST_mice <- TSR_ALL1_TEST %>% select(icase_id, idcase_id, mrs_tx_1) %>% cbind(complete(TSR_ALL1_TEST_imp, 5))
save_file <- pathJoin("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TEST_MICE5.csv")
write.csv(TSR_ALL1_TEST_mice, save_file, row.names=FALSE)
