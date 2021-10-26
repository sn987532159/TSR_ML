import pandas as pd
import os
import numpy as np

def delete_duplicated_rows(df):
    print(df.shape)
    df = df.drop_duplicates(["icase_id", "idcase_id"], keep="first")
    df = df.reset_index(drop=True)
    print(df.shape)
    return df

def mRS_records_deletion(df):
    df[df["mrs_tx_1"] == 9] = np.nan
    df["mrs_tx_1"][(df["death_dt_1"] != "1900/1/1") & (df["death_dt_1"].notnull())] = 9
    died_before_discharge = df[df["off_id"] != 3].index
    df = df.drop(df.index[died_before_discharge])
    print(df.shape)

    df = df.reset_index(drop=True)
    no_mrs_1 = df[df["mrs_tx_1"].isnull()].index
    df = df.drop(df.index[no_mrs_1])
    print(df.shape)
    return df

def special_cases_deletion(df):
    df = df.reset_index(drop=True)
    smaller_than18 = df[(df["age"] >= 0) & ((df["age"] < 18))].index
    df = df.drop(df.index[smaller_than18])
    print(df.shape)

    df = df.reset_index(drop=True)
    ischemic = df[df["icd_id"].isin([1,2])]
    print(ischemic.shape)

    df = df.reset_index(drop=True)
    other = df[~df["icd_id"].isin([1,2])]
    print(other.shape)
    return ischemic


if __name__ == '__main__':
    csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL.csv")
    tsr_all_df = pd.read_csv(csv_path, low_memory=False)

    tsr_all_df1 = delete_duplicated_rows(tsr_all_df)
    tsr_all_df2 = mRS_records_deletion(tsr_all_df1)
    tsr_all_df3 = special_cases_deletion(tsr_all_df2)

    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL1", "TSR_ALL1.csv")
    #tsr_all_df3.to_csv(csv_save, index=False)