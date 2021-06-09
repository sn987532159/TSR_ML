import pandas as pd
import os

def mRS_records_deletion(df):
    print(df.shape)

    died_before_discharge = df[df["off_id"] != 3].index
    df = df.drop(df.index[died_before_discharge])
    df = df.reset_index(drop=True)
    died_before_1_month = df[(df["death_dt_1"] != "1900/1/1") & (df["death_dt_1"].notnull())].index
    df = df.drop(df.index[died_before_1_month])
    print(df.shape)

    df = df.reset_index(drop=True)
    no_mrs_1 = df[df["mrs_tx_1"].isnull()].index
    df = df.drop(df.index[no_mrs_1])
    print(df.shape)

    df = df.reset_index(drop=True)
    no_mrs_3 = df[df["mrs_tx_3"].isnull()].index
    df = df.drop(df.index[no_mrs_3])
    print(df.shape)
    return df

def special_cases_deletion(df):
    df = df.reset_index(drop=True)
    other_stroke = df[(df["icd_id"] != 1) & (df["icd_id"] != 2) & (df["icd_id"] != 3) & (df["icd_id"] != 4)].index
    df = df.drop(df.index[other_stroke])
    print(df.shape)

    df = df.reset_index(drop=True)
    smaller_than18 = df[(df["age"] >= 0) & ((df["age"] < 18))].index
    df = df.drop(df.index[smaller_than18])
    print(df.shape)

    print(df[(df["icd_id"] == 1) | (df["icd_id"] == 2)].shape)
    print(df[(df["icd_id"] == 3) | (df["icd_id"] == 4)].shape)

    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3.csv")
    df.to_csv(csv_save, index=False)


if __name__ == '__main__':
    csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL.csv")
    tsr_all_df = pd.read_csv(csv_path, low_memory=False)

    tsr_all_df1 = mRS_records_deletion(tsr_all_df)
    tsr_all_df2 = special_cases_deletion(tsr_all_df1)

    csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3.csv")
    tsr_all_df2.to_csv(csv_save, index=False)