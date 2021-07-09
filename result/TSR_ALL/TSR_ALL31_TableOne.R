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
file_path <- pathJoin("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3_MICE5.csv")
TSR_ALL31 <- read.csv(file_path)
TSR_ALL31_df <- TSR_ALL31 %>% select(-icase_id, -idcase_id)

### GOOD when discharged
GOOD_when_discharged <- TSR_ALL31_df[(TSR_ALL31_df$discharged_mrs == 0) | (TSR_ALL31_df$discharged_mrs == 1) | (TSR_ALL31_df$discharged_mrs == 2), ] %>%
  select(mrs_tx_3, age, mrs_tx_1, omas_fl, discharged_mrs, pcva_id, pro_id, mobility, stairs, nihs_5al_out, nihs_6br_out)

GOOD_when_discharged[3:11] <- lapply(GOOD_when_discharged[3:11] , as.factor)

GOOD_when_discharged$mrs_tx_3[(GOOD_when_discharged$mrs_tx_3==0) | (GOOD_when_discharged$mrs_tx_3==1) | (GOOD_when_discharged$mrs_tx_3==2)] = "Not changed"
GOOD_when_discharged$mrs_tx_3[GOOD_when_discharged$mrs_tx_3!="Not changed"] = "Changed"
label(GOOD_when_discharged$mrs_tx_3) <- "3-month mRS"
label(GOOD_when_discharged$age) <- "Age"
label(GOOD_when_discharged$mrs_tx_1) <- "1-month mRS"
label(GOOD_when_discharged$omas_fl) <- "Aspirin after discharged"
label(GOOD_when_discharged$discharged_mrs) <- "Discharged mRS"
label(GOOD_when_discharged$pcva_id) <- "Previous CVA"
label(GOOD_when_discharged$pro_id) <- "Profession"
label(GOOD_when_discharged$mobility) <- "BI_mobility"
label(GOOD_when_discharged$stairs) <- "BI_stairs"
label(GOOD_when_discharged$nihs_5al_out) <- "Discharged NIHSS_5al"
label(GOOD_when_discharged$nihs_6br_out) <- "Discharged NIHSS_6br"

table1(~ .|mrs_tx_3, data=GOOD_when_discharged, overall=F, extra.col=list("P-value"=pvalue))
table_smd<- CreateTableOne(strata = "mrs_tx_3", data = GOOD_when_discharged, test = TRUE)
print(table_smd, smd = TRUE)

### BAD when discharged
BAD_when_discharged <- TSR_ALL31_df[(TSR_ALL31_df$discharged_mrs != 0) & (TSR_ALL31_df$discharged_mrs != 1) & (TSR_ALL31_df$discharged_mrs != 2), ] %>%
  select(mrs_tx_3, mrs_tx_1, discharged_mrs, pcvaci_id, pcva_id, pro_id, bladder_control, feeding, mobility,toilet_use, transfers, 
         nihs_6al_out)

BAD_when_discharged[2:11] <- lapply(BAD_when_discharged[2:11] , factor)

BAD_when_discharged$mrs_tx_3[(BAD_when_discharged$mrs_tx_3==0) | (BAD_when_discharged$mrs_tx_3==1) | (BAD_when_discharged$mrs_tx_3==2)] = "Changed"
BAD_when_discharged$mrs_tx_3[(BAD_when_discharged$mrs_tx_3!="Changed")] = "Not changed"
label(BAD_when_discharged$mrs_tx_3) <- "3-month mRS"
label(BAD_when_discharged$mrs_tx_1) <- "1-month mRS"
label(BAD_when_discharged$discharged_mrs) <- "Discharged mRS"
label(BAD_when_discharged$pcvaci_id) <- "Previous cerebral infraction"
label(BAD_when_discharged$pcva_id) <- "Previous CVA"
label(BAD_when_discharged$pro_id) <- "Profession"
label(BAD_when_discharged$bladder_control) <- "BI_bladder controll" 
label(BAD_when_discharged$feeding) <- "BI_feeding"
label(BAD_when_discharged$mobility) <- "BI_mobility"
label(BAD_when_discharged$toilet_use) <- "BI_toilet_use"
label(BAD_when_discharged$transfers) <- "BI_transfers"
label(BAD_when_discharged$nihs_6al_out) <- "Discharged NIHSS_6al"

table1(~ .|mrs_tx_3, data=BAD_when_discharged, overall=F, extra.col=list("P-value"=pvalue))
table_smd<- CreateTableOne(strata = "mrs_tx_3", data = BAD_when_discharged, test = TRUE)
print(table_smd, smd = TRUE)

