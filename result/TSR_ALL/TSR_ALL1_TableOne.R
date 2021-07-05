install.packages("table1")
install.packages("tableone")
install.packages("MBESS")
library(table1)
library(tableone)
library(Gmisc)
library(dplyr)
library(fastDummies)
library(MBESS)

pvalue <- function(x, ...) {
  # Construct vectors of data y, and groups (strata) g
  y <- unlist(x)
  g <- factor(rep(1:length(x), times=sapply(x, length)))
  if (is.numeric(y)) {
    # For numeric variables, perform a standard 2-sample t-test
    p <- t.test(y ~ g)$p.value
  } else {
    # For categorical variables, perform a chi-squared test of independence
    p <- chisq.test(table(y, g))$p.value
  }
  # Format the p-value, using an HTML entity for the less-than sign.
  # The initial empty string places the output on the line below the variable label.
  c("", sub("<", "&lt;", format.pval(p, digits=3, eps=0.001)))
}

setwd("C:/Users/Jacky C/PycharmProjects/tsr_ml/result/TSR_ALL")
file_path <- pathJoin("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_MICE2.csv")
TSR_ALL1 <- read.csv(file_path)
TSR_ALL1 <- TSR_ALL1 %>% select(-icase_id, -idcase_id)
nominal_features = c("edu_id", "pro_id", "opc_id", "toast_id", "offdt_id", "gender_tx", "hd_id", "pcva_id", 
                     "pcvaci_id", "pcvach_id", "po_id", "ur_id", "sm_id", "ptia_id", "hc_id", "hcht_id", 
                     "hchc_id", "ht_id", "dm_id", "pad_id", "al_id", "ca_id", "fahiid_parents_1", 
                     "fahiid_parents_2", "fahiid_parents_3", "fahiid_parents_4", "fahiid_brsi_1", 
                     "fahiid_brsi_2", "fahiid_brsi_3", "fahiid_brsi_4")

TSR_ALL1_df <- dummy_cols(TSR_ALL1, nominal_features, remove_selected_columns = FALSE)

### GOOD when discharged
GOOD_when_discharged <- TSR_ALL1_df[(TSR_ALL1_df$discharged_mrs == 0) | (TSR_ALL1_df$discharged_mrs == 1) | (TSR_ALL1_df$discharged_mrs == 2), ] %>%
  select(mrs_tx_1, discharged_mrs, fahiid_brsi_1, fahiid_brsi_4, pro_id, subcortical_mca_mrir, dressing, grooming, 
         mobility, stairs, toilet_use, transfers, nihs_10_out)
GOOD_when_discharged[2:13] <- lapply(GOOD_when_discharged[2:13] , factor)

GOOD_when_discharged$mrs_tx_1[(GOOD_when_discharged$mrs_tx_1==0) | (GOOD_when_discharged$mrs_tx_1==1) | (GOOD_when_discharged$mrs_tx_1==2)] = "Not changed"
GOOD_when_discharged$mrs_tx_1[GOOD_when_discharged$mrs_tx_1!="Not changed"] = "Changed"
label(GOOD_when_discharged$mrs_tx_1) <- "1-month mRS"
label(GOOD_when_discharged$discharged_mrs) <- "Discharged mRS"
label(GOOD_when_discharged$fahiid_brsi_1) <- "Siblings having hypertension"
label(GOOD_when_discharged$fahiid_brsi_4) <- "Siblings having stroke or TIA"
label(GOOD_when_discharged$pro_id) <- "Profession"
label(GOOD_when_discharged$subcortical_mca_mrir) <- "MRI_subcortical MCA_right"
label(GOOD_when_discharged$dressing) <- "BI_dressing"
label(GOOD_when_discharged$grooming) <- "BI_grooming"
label(GOOD_when_discharged$mobility) <- "BI_mobility"
label(GOOD_when_discharged$stairs) <- "BI_stairs"
label(GOOD_when_discharged$toilet_use) <- "BI_toilet_use"
label(GOOD_when_discharged$transfers) <- "BI_transfers"
label(GOOD_when_discharged$nihs_10_out) <- "Discharged NIHSS_10"
                                                                              
table1(~ .|mrs_tx_1, data=GOOD_when_discharged, overall=F, extra.col=list("P-value"=pvalue))
table_smd<- CreateTableOne(strata = "mrs_tx_1", data = GOOD_when_discharged, test = TRUE)
print(table_smd, smd = TRUE)

### BAD when discharged
BAD_when_discharged <- TSR_ALL1_df[(TSR_ALL1_df$discharged_mrs != 0) & (TSR_ALL1_df$discharged_mrs != 1) & (TSR_ALL1_df$discharged_mrs != 2), ] %>%
  select(mrs_tx_1, age, hospitalised_time, discharged_mrs, offdt_id, omas_fl, trmfo_fl, trmng_fl, bathing, bladder_control, 
         bowel_control, dressing, feeding, mobility, stairs, toilet_use, transfers, nihs_5al_out, nihs_5br_out, nihs_6al_out,
         nihs_6br_out, nihs_10_out)
BAD_when_discharged[4:22] <- lapply(BAD_when_discharged[4:22] , factor)
BAD_when_discharged[2:3] <- lapply(BAD_when_discharged[2:3] , as.numeric)

BAD_when_discharged$mrs_tx_1[(BAD_when_discharged$mrs_tx_1==0) | (BAD_when_discharged$mrs_tx_1==1) | (BAD_when_discharged$mrs_tx_1==2)] = "Changed"
BAD_when_discharged$mrs_tx_1[BAD_when_discharged$mrs_tx_1!="Changed"] = "Not changed"
label(BAD_when_discharged$mrs_tx_1) <- "1-month mRS"
label(BAD_when_discharged$age) <- "Age" 
label(BAD_when_discharged$hospitalised_time) <- "Hospitalised duration"
label(BAD_when_discharged$discharged_mrs) <- "Discharged mRS" 
label(BAD_when_discharged$offdt_id) <- "Destination after discharged"
label(BAD_when_discharged$omas_fl) <- "Aspirin after discharged"
label(BAD_when_discharged$trmfo_fl) <- "Hospitalised treatment-foley"
label(BAD_when_discharged$trmng_fl) <- "Hospitalised treatment-nasogastric tube"
label(BAD_when_discharged$bathing) <- "BI_bathing"
label(BAD_when_discharged$bladder_control) <- "BI_bladder_control"
label(BAD_when_discharged$bowel_control) <- "BI_bowel_control"
label(BAD_when_discharged$dressing) <- "BI_dressing"
label(BAD_when_discharged$feeding) <- "BI_feeding"
label(BAD_when_discharged$mobility) <- "BI_mobility"
label(BAD_when_discharged$stairs) <- "BI_stairs"
label(BAD_when_discharged$toilet_use) <- "BI_toilet_use"
label(BAD_when_discharged$transfers) <- "BI_transfers"
label(BAD_when_discharged$nihs_5al_out) <- "Discharged NIHSS_5al"
label(BAD_when_discharged$nihs_5br_out) <- "Discharged NIHSS_5br"
label(BAD_when_discharged$nihs_6al_out) <- "Discharged NIHSS_6al"
label(BAD_when_discharged$nihs_6br_out) <- "Discharged NIHSS_6br"
label(BAD_when_discharged$nihs_10_out) <- "Discharged NIHSS_10"

table1(~ .|mrs_tx_1, data=BAD_when_discharged, overall=F, extra.col=list("P-value"=pvalue))
table_smd<- CreateTableOne(strata = "mrs_tx_1", data = BAD_when_discharged, test = TRUE)
print(table_smd, smd = TRUE)

