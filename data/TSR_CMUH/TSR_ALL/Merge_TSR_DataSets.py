import pandas as pd

CASEDBMRS_1 = pd.read_csv("CASEDBMRS_1.csv", low_memory=False)
CASEDBMRS_2 = pd.read_csv("CASEDBMRS_2.csv", low_memory=False)

CASEDBMRS = pd.concat([CASEDBMRS_1,CASEDBMRS_2],axis=0)
CASEDBMRS.to_csv("CASEDBMRS.csv",index=False)

print(CASEDBMRS_1.shape, CASEDBMRS_2.shape)
print("Total:", CASEDBMRS.shape)

CASEDCTMR_1 = pd.read_csv("CASEDCTMR_1.csv", low_memory=False)
CASEDCTMR_2 = pd.read_csv("CASEDCTMR_2.csv", low_memory=False)

CASEDCTMR = pd.concat([CASEDCTMR_1,CASEDCTMR_2],axis=0)
CASEDCTMR.to_csv("CASEDCTMR.csv",index=False)

print(CASEDCTMR_1.shape, CASEDCTMR_2.shape)
print("Total:", CASEDCTMR.shape)

CASEDNIHS_1 = pd.read_csv("CASEDNIHS_1.csv", low_memory=False)
CASEDNIHS_2 = pd.read_csv("CASEDNIHS_2.csv", low_memory=False)
CASEDNIHS_3 = pd.read_csv("CASEDNIHS_3.csv", low_memory=False)

CASEDNIHS = pd.concat([CASEDNIHS_1,CASEDNIHS_2, CASEDNIHS_3],axis=0)
CASEDNIHS.to_csv("CASEDNIHS.csv",index=False)

print(CASEDNIHS_1.shape, CASEDNIHS_2.shape, CASEDNIHS_3.shape)
print("Total:", CASEDNIHS.shape)