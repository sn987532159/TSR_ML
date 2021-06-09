import pandas as pd
import os

def mRS_records_deletion(filename):
    print(filename.shape)

    died_before_discharge = filename[filename["off_id"] != 3].index
    filename = filename.drop(filename.index[died_before_discharge])
    filename = filename.reset_index(drop=True)
    died_before_1_month = filename[(filename["death_dt_1"] != "1900/1/1") & (filename["death_dt_1"].notnull())].index
    filename = filename.drop(filename.index[died_before_1_month])
    print(filename.shape)

    filename = filename.reset_index(drop=True)
    no_mrs_1 = filename[filename["mrs_tx_1"].isnull()].index
    filename = filename.drop(filename.index[no_mrs_1])
    print(filename.shape)

    filename = filename.reset_index(drop=True)
    no_mrs_3 = filename[filename["mrs_tx_3"].isnull()].index
    filename = filename.drop(filename.index[no_mrs_3])
    print(filename.shape)
    #return __main__

##def special_cases_deletion(filename):
    filename = filename.reset_index(drop=True)
    other_stroke = filename[(filename["icd_id"] != 1) & (filename["icd_id"] != 2) & (filename["icd_id"] != 3) & (filename["icd_id"] != 4)].index
    filename = filename.drop(filename.index[other_stroke])
    print(filename.shape)

    filename = filename.reset_index(drop=True)
    smaller_than18 = filename[(filename["age"] >= 0) & ((filename["age"] < 18))].index
    filename = filename.drop(filename.index[smaller_than18])
    print(filename.shape)

    print(filename[(filename["icd_id"] == 1) | (filename["icd_id"] == 2)].shape)
    print(filename[(filename["icd_id"] == 3) | (filename["icd_id"] == 4)].shape)

    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3.csv")
    filename.to_csv(csv_save, index=False)


if __name__ == '__main__':
    csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL.csv")
    tsr_all = pd.read_csv(csv_path, low_memory=False)

    mRS_records_deletion(tsr_all)
    #special_cases_deletion(tsr_all)