install.packages("tableone")
library(Gmisc)
library(dplyr)
library(tableone)

setwd("C:/Users/Jacky C/PycharmProjects/tsr_ml/result/TSR_ALL")
file_path <- pathJoin("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TRAIN.csv")
TSR_ALL1_TRAIN <- read.csv(file_path)
file_path <- pathJoin("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_VALIDATION.csv")
TSR_ALL1_VALIDATION <- read.csv(file_path)
file_path <- pathJoin("..", "..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1_TEST.csv")
TSR_ALL1_TEST <- read.csv(file_path)

TSR_ALL1_TRAIN_df <- TSR_ALL1_TRAIN %>%
  mutate(BI = feeding+transfers+bathing+toilet_use+grooming+mobility+stairs+dressing+bowel_control+bladder_control)%>%
  mutate(NIHSS_IN = nihs_1a_in+nihs_1b_in+nihs_1c_in+nihs_2_in+nihs_3_in+nihs_4_in+nihs_5al_in+nihs_5br_in+nihs_6al_in+nihs_6br_in+nihs_7_in+nihs_8_in+nihs_9_in+nihs_10_in+nihs_11_in)%>%
  mutate(NIHSS_OUT = nihs_1a_out+nihs_1b_out+nihs_1c_out+nihs_2_out+nihs_3_out+nihs_4_out+nihs_5al_out+nihs_5br_out+nihs_6al_out+nihs_6br_out+nihs_7_out+nihs_8_out+nihs_9_out+nihs_10_out+nihs_11_out)%>%
  select(gender_tx, age, mrs_tx_1, discharged_mrs, BI, NIHSS_IN, NIHSS_OUT)
TSR_ALL1_TRAIN_df["TABLEONE"] <- "TRAIN"

TSR_ALL1_VALIDATION_df <- TSR_ALL1_VALIDATION %>%
  mutate(BI = feeding+transfers+bathing+toilet_use+grooming+mobility+stairs+dressing+bowel_control+bladder_control)%>%
  mutate(NIHSS_IN = nihs_1a_in+nihs_1b_in+nihs_1c_in+nihs_2_in+nihs_3_in+nihs_4_in+nihs_5al_in+nihs_5br_in+nihs_6al_in+nihs_6br_in+nihs_7_in+nihs_8_in+nihs_9_in+nihs_10_in+nihs_11_in)%>%
  mutate(NIHSS_OUT = nihs_1a_out+nihs_1b_out+nihs_1c_out+nihs_2_out+nihs_3_out+nihs_4_out+nihs_5al_out+nihs_5br_out+nihs_6al_out+nihs_6br_out+nihs_7_out+nihs_8_out+nihs_9_out+nihs_10_out+nihs_11_out)%>%
  select(gender_tx, age, mrs_tx_1, discharged_mrs, BI, NIHSS_IN, NIHSS_OUT)
TSR_ALL1_VALIDATION_df["TABLEONE"] <- "VALIDATION"

TSR_ALL1_TEST_df <- TSR_ALL1_TEST %>%
  mutate(BI = feeding+transfers+bathing+toilet_use+grooming+mobility+stairs+dressing+bowel_control+bladder_control)%>%
  mutate(NIHSS_IN = nihs_1a_in+nihs_1b_in+nihs_1c_in+nihs_2_in+nihs_3_in+nihs_4_in+nihs_5al_in+nihs_5br_in+nihs_6al_in+nihs_6br_in+nihs_7_in+nihs_8_in+nihs_9_in+nihs_10_in+nihs_11_in)%>%
  mutate(NIHSS_OUT = nihs_1a_out+nihs_1b_out+nihs_1c_out+nihs_2_out+nihs_3_out+nihs_4_out+nihs_5al_out+nihs_5br_out+nihs_6al_out+nihs_6br_out+nihs_7_out+nihs_8_out+nihs_9_out+nihs_10_out+nihs_11_out)%>%
  select(gender_tx, age, mrs_tx_1, discharged_mrs, BI, NIHSS_IN, NIHSS_OUT)
TSR_ALL1_TEST_df["TABLEONE"] <- "TEST"

TSR_ALL1_df <- rbind(TSR_ALL1_TRAIN_df, TSR_ALL1_VALIDATION_df, TSR_ALL1_TEST_df)
TSR_ALL1_df[TSR_ALL1_df == 9999] = NA
#write.csv(TSR_ALL1_df, file = "C:/Users/Jacky C/PycharmProjects/tsr_ml/result/TSR_ALL/try.csv", row.names = FALSE)

TSR_ALL1_tableone<- CreateTableOne(data = TSR_ALL1_df, strata = c("TABLEONE"))
TSR_ALL1_tableone_1<- print(TSR_ALL1_tableone, nonnormal = "age", smd = TRUE)
write.csv(TSR_ALL1_tableone_1, file = "C:/Users/Jacky C/PycharmProjects/tsr_ml/result/TSR_ALL/myTable.csv")
