import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.font_manager import FontProperties
import numpy as np

os.chdir('/data_cleansing')

font_path = os.path.join("..", "kaiu.ttf")
myfont = FontProperties(fname=font_path, size=14)
sns.set(font=myfont.get_name())

plt.rcParams['figure.figsize'] = (10, 5)
plt.rc('axes', unicode_minus=False)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_EHR", "TSR_6.csv")
tsr_6 = pd.read_csv(csv_path, low_memory=False, encoding='unicode_escape')
tsr_6.head()

tsr_6.describe()

# icase_id
# idcase_id
# fstatus_id

# 追蹤日期_1

rfur_dt_1 = tsr_6.loc[:, "rfur_dt_1"]
rfur_dt_1 = pd.to_datetime(rfur_dt_1)
# print(rfur_dt_1)
# print(rfur_dt_1.value_counts() / len(rfur_dt_1))
print(rfur_dt_1.describe())

rfur_dt_1.value_counts().plot()
plt.title("Date of 1-month follow-up - Lineplot")
plt.xlabel('Date of 1-month follow-up')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 目前所在地_1

location_id_1 = tsr_6.loc[:, "location_id_1"]
# print(location_id_1)
print(location_id_1.value_counts() / len(location_id_1))
# print(location_id_1.describe())

# location_id_1_labels = ["住家","護理之家","呼吸病房","本院住院中","轉至其他院","失聯"]
sns.countplot(location_id_1)  # .set_xticklabels(location_id_1_labels)
plt.title("LOCATION_ID_1 - Barplot")
plt.xlabel('LOCATION_ID_1')
plt.ylabel('Number', rotation=0)
plt.show()

# 轉至醫院_1

torg_id_1 = tsr_6.loc[:, "torg_id_1"]
# print(torg_id_1)
print(torg_id_1.value_counts() / len(torg_id_1))
# print(torg_id_1.describe())

sns.countplot(torg_id_1)
plt.title("TORG_ID_1 - Barplot")
plt.xlabel('TORG_ID_1')
plt.ylabel('Number', rotation=0)
plt.show()

# 相關醫療紀錄_1

flu_id_1 = tsr_6.loc[:, "flu_id_1"]
# print(flu_id_1)
print(flu_id_1.value_counts() / len(flu_id_1))
# print(flu_id_1.describe())

# flu_id_1_labels = ["本院或他院門診繼續服藥","拒回診","死亡"]
sns.countplot(flu_id_1)  # .set_xticklabels(flu_id_1_labels)
plt.title("FLU_ID_1 - Barplot")
plt.xlabel('FLU_ID_1')
plt.ylabel('Number', rotation=0)
plt.show()

# 門診醫院_1

fluorg_id_1 = tsr_6.loc[:, "fluorg_id_1"]
# print(fluorg_id_1)
print(fluorg_id_1.value_counts() / len(fluorg_id_1))
# print(fluorg_id_1.describe())

sns.countplot(fluorg_id_1)
plt.title("FLUORG_ID_1 - Barplot")
plt.xlabel('FLUORG_ID_1')
plt.ylabel('Number', rotation=0)
plt.show()

# FLUORG_TX_1
# FLURESULT_TX_1
# DEATH_DT_1
# DEATH_ID_1
# DEATHSK_ID_1
# DEATHO_TX_1

# 相關醫療紀錄2_1

ve_id_1 = tsr_6.loc[:, 've_id_1']
ve_id_1[(ve_id_1 == str(0)) | (ve_id_1 == 0)] = "N"

# print(ve_id_1)
print(ve_id_1.value_counts() / len(ve_id_1))
# print(ve_id_1.describe())

sns.countplot(ve_id_1)
plt.title("VE_ID_1 - Barplot")
plt.xlabel('VE_ID_1')
plt.ylabel('Number', rotation=0)
plt.show()

# 再中風_1

vers_fl_1 = tsr_6.loc[:, 'vers_fl_1']
# print(vers_fl_1)
print(vers_fl_1.value_counts() / len(vers_fl_1))
# print(vers_fl_1.describe())

sns.countplot(vers_fl_1)
plt.title("VERS_FL_1 - Barplot")
plt.xlabel('VERS_FL_1')
plt.ylabel('Number', rotation=0)
plt.show()

# 再中風 (CI / CH)_1

verscich_id_1 = tsr_6.loc[:, 'verscich_id_1']
verscich_id_1[verscich_id_1 == str(1)] = 1
verscich_id_1[verscich_id_1 == str(2)] = 2
verscich_id_1[(verscich_id_1 != 1) & (verscich_id_1 != 2)] = np.nan
# print(verscich_id_1)
print(verscich_id_1.value_counts() / len(verscich_id_1))
# print(verscich_id_1.describe())

# verscich_id_1_labels = ["CI","CH"]
sns.countplot(verscich_id_1)  # .set_xticklabels(verscich_id_1_labels)
plt.title("VERSCICH_ID_1 - Barplot")
plt.xlabel('VERSCICH_ID_1')
plt.ylabel('Number', rotation=0)
plt.show()
verscich_id_1 = verscich_id_1.fillna(999)

# 再中風日期_1

vers_dt_1 = tsr_6.loc[:, 'vers_dt_1']
vers_dt_1 = pd.to_datetime(vers_dt_1, errors='coerce')
vers_dt_1[(vers_dt_1.dt.year < 2006) | (vers_dt_1.dt.year > 2021)] = np.nan
# print(vers_dt_1)
# print(vers_dt_1.value_counts() / len(vers_dt_1))
print(vers_dt_1.describe())

vers_dt_1.value_counts().plot()
plt.title("VERS_DT_1 - Lineplot")
plt.xlabel('VERS_DT_1')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 再中風醫院_1

versorg_id_1 = tsr_6.loc[:, 'versorg_id_1']
# print(versorg_id_1)
print(versorg_id_1.value_counts() / len(versorg_id_1))
# print(versorg_id_1.describe())

sns.countplot(versorg_id_1)
plt.title("VERSORG_ID_1 - Barplot")
plt.xlabel('VERSORG_ID_1')
plt.ylabel('Number', rotation=0)
plt.show()

# Ischemic Heart Disease_1

veihd_fl_1 = tsr_6.loc[:, 'veihd_fl_1']
# print(veihd_fl_1)
print(veihd_fl_1.value_counts() / len(veihd_fl_1))
# print(veihd_fl_1.describe())

sns.countplot(veihd_fl_1)
plt.title("VEIHD_FL_1 - Barplot")
plt.xlabel('VEIHD_FL_1')
plt.ylabel('Number', rotation=0)
plt.show()

# Types of Ischemic Heart Disease_1

veihd_id_1 = tsr_6.loc[:, 'veihd_id_1']
# print(veihd_id_1)
print(veihd_id_1.value_counts() / len(veihd_id_1))
# print(veihd_id_1.describe())

# veihd_id_1_labels = ["AMI","Angina"]
sns.countplot(veihd_id_1)  # .set_xticklabels(veihd_id_1_labels)
plt.title("VEIHD_ID_1 - Barplot")
plt.xlabel('VEIHD_ID_1')
plt.ylabel('Number', rotation=0)
plt.show()

# Ischemic Heart Disease日期_1

veihd_dt_1 = tsr_6.loc[:, 'veihd_dt_1']
veihd_dt_1 = pd.to_datetime(veihd_dt_1, errors='coerce')
veihd_dt_1[(veihd_dt_1.dt.year < 2006) | (veihd_dt_1.dt.year > 2021)] = np.nan
# print(veihd_dt_1)
# print(veihd_dt_1.value_counts() / len(veihd_dt_1))
print(veihd_dt_1.describe())

veihd_dt_1.value_counts().plot()
plt.title("VEIHD_DT_1 - Lineplot")
plt.xlabel('VEIHD_DT_1')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# Ischemic Heart Disease醫院_1

veihdorg_id_1 = tsr_6.loc[:, 'veihdorg_id_1']
# print(veihdorg_id_1)
print(veihdorg_id_1.value_counts() / len(veihdorg_id_1))
# print(veihdorg_id_1.describe())

sns.countplot(veihdorg_id_1)
plt.title("VEIHDORG_ID_1 - Barplot")
plt.xlabel('VEIHDORG_ID_1')
plt.ylabel('Number', rotation=0)
plt.show()

# mRS_1

mrs_tx_1 = tsr_6.loc[:, 'mrs_tx_1']
mrs_tx_1 = mrs_tx_1.fillna(mrs_tx_1.mode()[0])
# print(mrs_tx_1)
print(mrs_tx_1.value_counts() / len(mrs_tx_1))
# print(mrs_tx_1.describe())

sns.countplot(mrs_tx_1)
plt.title("MRS_TX_1 - Barplot")
plt.xlabel('MRS_TX_1')
plt.ylabel('Number', rotation=0)
plt.show()

# TORG_TX_1
# VERSORG_TX_1
# VEIHDORG_TX_1

# 追蹤日期_3

rfur_dt_3 = tsr_6.loc[:, "rfur_dt_3"]
rfur_dt_3 = pd.to_datetime(rfur_dt_3, errors="coerce")
# print(rfur_dt_3)
# print(rfur_dt_3.value_counts() / len(rfur_dt_3))
print(rfur_dt_3.describe())

rfur_dt_3.value_counts().plot()
plt.title("Date of 3-month follow-up - Lineplot")
plt.xlabel('Date of 3-month follow-up')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 目前所在地_3

location_id_3 = tsr_6.loc[:, "location_id_3"]
# print(location_id_3)
print(location_id_3.value_counts() / len(location_id_3))
# print(location_id_3.describe())

# location_id_3_labels = ["住家","護理之家","呼吸病房","本院住院中","轉至其他院","失聯"]
sns.countplot(location_id_3)  # .set_xticklabels(location_id_3_labels)
plt.title("LOCATION_ID_3 - Barplot")
plt.xlabel('LOCATION_ID_3')
plt.ylabel('Number', rotation=0)
plt.show()

# 轉至醫院_3

torg_id_3 = tsr_6.loc[:, "torg_id_3"]
# print(torg_id_3)
print(torg_id_3.value_counts() / len(torg_id_3))
# print(torg_id_3.describe())

sns.countplot(torg_id_3)
plt.title("TORG_ID_3 - Barplot")
plt.xlabel('TORG_ID_3')
plt.ylabel('Number', rotation=0)
plt.show()

# 相關醫療紀錄_3

flu_id_3 = tsr_6.loc[:, "flu_id_3"]
# print(flu_id_3)
print(flu_id_3.value_counts() / len(flu_id_3))
# print(flu_id_3.describe())

# flu_id_3_labels = ["本院或他院門診繼續服藥","拒回診","死亡"]
sns.countplot(flu_id_3)  # .set_xticklabels(flu_id_3_labels)
plt.title("FLU_ID_3 - Barplot")
plt.xlabel('FLU_ID_3')
plt.ylabel('Number', rotation=0)
plt.show()

# 門診醫院_3

fluorg_id_3 = tsr_6.loc[:, "fluorg_id_3"]
# print(fluorg_id_3)
print(fluorg_id_3.value_counts() / len(fluorg_id_3))
# print(fluorg_id_3.describe())

sns.countplot(fluorg_id_3)
plt.title("FLUORG_ID_3 - Barplot")
plt.xlabel('FLUORG_ID_3')
plt.ylabel('Number', rotation=0)
plt.show()

# FLUORG_TX_3
# FLURESULT_TX_3
# DEATH_DT_3
# DEATH_ID_3
# DEATHSK_ID_3
# DEATHO_TX_3

# 相關醫療紀錄2_3

ve_id_3 = tsr_6.loc[:, 've_id_3']
ve_id_3[(ve_id_3 == str(0)) | (ve_id_3 == 0)] = "N"

# print(ve_id_3)
print(ve_id_3.value_counts() / len(ve_id_3))
# print(ve_id_3.describe())

sns.countplot(ve_id_3)
plt.title("VE_ID_3 - Barplot")
plt.xlabel('VE_ID_3')
plt.ylabel('Number', rotation=0)
plt.show()

# 再中風_3

vers_fl_3 = tsr_6.loc[:, 'vers_fl_3']
vers_fl_3[(vers_fl_3 != "N") & (vers_fl_3 != "Y")] = np.nan
# print(vers_fl_3)
print(vers_fl_3.value_counts() / len(vers_fl_3))
# print(vers_fl_3.describe())

sns.countplot(vers_fl_3)
plt.title("VERS_FL_3 - Barplot")
plt.xlabel('VERS_FL_3')
plt.ylabel('Number', rotation=0)
plt.show()

# 再中風 (CI / CH)_3

verscich_id_3 = tsr_6.loc[:, 'verscich_id_3']
verscich_id_3[verscich_id_3 == str(1)] = 1
verscich_id_3[verscich_id_3 == str(2)] = 2
verscich_id_3[(verscich_id_3 != 1) & (verscich_id_3 != 2)] = np.nan
# print(verscich_id_3)
print(verscich_id_3.value_counts() / len(verscich_id_3))
# print(verscich_id_3.describe())

# verscich_id_3_labels = ["CI","CH"]
sns.countplot(verscich_id_3)  # .set_xticklabels(verscich_id_3_labels)
plt.title("VERSCICH_ID_3 - Barplot")
plt.xlabel('VERSCICH_ID_3')
plt.ylabel('Number', rotation=0)
plt.show()
verscich_id_3 = verscich_id_3.fillna(999)

# 再中風日期_3

vers_dt_3 = tsr_6.loc[:, 'vers_dt_3']
vers_dt_3 = pd.to_datetime(vers_dt_3, errors='coerce')
vers_dt_3[(vers_dt_3.dt.year < 2006) | (vers_dt_3.dt.year > 2023)] = np.nan
# print(vers_dt_3)
# print(vers_dt_3.value_counts() / len(vers_dt_3))
print(vers_dt_3.describe())

vers_dt_3.value_counts().plot()
plt.title("VERS_DT_3 - Lineplot")
plt.xlabel('VERS_DT_3')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 再中風醫院_3

versorg_id_3 = tsr_6.loc[:, 'versorg_id_3']
# print(versorg_id_3)
print(versorg_id_3.value_counts() / len(versorg_id_3))
# print(versorg_id_3.describe())

sns.countplot(versorg_id_3)
plt.title("VERSORG_ID_3 - Barplot")
plt.xlabel('VERSORG_ID_3')
plt.ylabel('Number', rotation=0)
plt.show()

# Ischemic Heart Disease_3

veihd_fl_3 = tsr_6.loc[:, 'veihd_fl_3']
# print(veihd_fl_3)
print(veihd_fl_3.value_counts() / len(veihd_fl_3))
# print(veihd_fl_3.describe())

sns.countplot(veihd_fl_3)
plt.title("VEIHD_FL_3 - Barplot")
plt.xlabel('VEIHD_FL_3')
plt.ylabel('Number', rotation=0)
plt.show()

# Types of Ischemic Heart Disease_3

veihd_id_3 = tsr_6.loc[:, 'veihd_id_3']
# print(veihd_id_3)
print(veihd_id_3.value_counts() / len(veihd_id_3))
# print(veihd_id_3.describe())

# veihd_id_3_labels = ["AMI","Angina"]
sns.countplot(veihd_id_3)  # .set_xticklabels(veihd_id_3_labels)
plt.title("VEIHD_ID_3 - Barplot")
plt.xlabel('VEIHD_ID_3')
plt.ylabel('Number', rotation=0)
plt.show()

# Ischemic Heart Disease日期_3

veihd_dt_3 = tsr_6.loc[:, 'veihd_dt_3']
veihd_dt_3 = pd.to_datetime(veihd_dt_3, errors='coerce')
veihd_dt_3[(veihd_dt_3.dt.year < 2006) | (veihd_dt_3.dt.year > 2023)] = np.nan
# print(veihd_dt_3)
# print(veihd_dt_3.value_counts() / len(veihd_dt_3))
print(veihd_dt_3.describe())

veihd_dt_3.value_counts().plot()
plt.title("VEIHD_DT_3 - Lineplot")
plt.xlabel('VEIHD_DT_3')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# Ischemic Heart Disease醫院_3

veihdorg_id_3 = tsr_6.loc[:, 'veihdorg_id_3']
# print(veihdorg_id_3)
print(veihdorg_id_3.value_counts() / len(veihdorg_id_3))
# print(veihdorg_id_3.describe())

sns.countplot(veihdorg_id_3)
plt.title("VEIHDORG_ID_3 - Barplot")
plt.xlabel('VEIHDORG_ID_3')
plt.ylabel('Number', rotation=0)
plt.show()

# mRS_3

mrs_tx_3 = tsr_6.loc[:, 'mrs_tx_3']
mrs_tx_3 = mrs_tx_3.fillna(mrs_tx_3.mode()[0])
# print(mrs_tx_3)
print(mrs_tx_3.value_counts() / len(mrs_tx_3))
# print(mrs_tx_3.describe())

sns.countplot(mrs_tx_3)
plt.title("MRS_TX_3 - Barplot")
plt.xlabel('MRS_TX_3')
plt.ylabel('Number', rotation=0)
plt.show()

# TORG_TX_3
# VERSORG_TX_3
# VEIHDORG_TX_3


# 追蹤日期_6

rfur_dt_6 = tsr_6.loc[:, "rfur_dt_6"]
rfur_dt_6 = pd.to_datetime(rfur_dt_6, errors="coerce")
# print(rfur_dt_6)
# print(rfur_dt_6.value_counts() / len(rfur_dt_6))
print(rfur_dt_6.describe())

rfur_dt_6.value_counts().plot()
plt.title("Date of 3-month follow-up - Lineplot")
plt.xlabel('Date of 3-month follow-up')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 目前所在地_6

location_id_6 = tsr_6.loc[:, "location_id_6"]
# print(location_id_6)
print(location_id_6.value_counts() / len(location_id_6))
# print(location_id_6.describe())

# location_id_6_labels = ["住家","護理之家","呼吸病房","本院住院中","轉至其他院","失聯"]
sns.countplot(location_id_6)  # .set_xticklabels(location_id_6_labels)
plt.title("LOCATION_ID_6 - Barplot")
plt.xlabel('LOCATION_ID_6')
plt.ylabel('Number', rotation=0)
plt.show()

# 轉至醫院_6

torg_id_6 = tsr_6.loc[:, "torg_id_6"]
# print(torg_id_6)
print(torg_id_6.value_counts() / len(torg_id_6))
# print(torg_id_6.describe())

sns.countplot(torg_id_6)
plt.title("TORG_ID_6 - Barplot")
plt.xlabel('TORG_ID_6')
plt.ylabel('Number', rotation=0)
plt.show()

# 相關醫療紀錄_6

flu_id_6 = tsr_6.loc[:, "flu_id_6"]
# print(flu_id_6)
print(flu_id_6.value_counts() / len(flu_id_6))
# print(flu_id_6.describe())

# flu_id_6_labels = ["本院或他院門診繼續服藥","拒回診","死亡"]
sns.countplot(flu_id_6)  # .set_xticklabels(flu_id_6_labels)
plt.title("FLU_ID_6 - Barplot")
plt.xlabel('FLU_ID_6')
plt.ylabel('Number', rotation=0)
plt.show()

# 門診醫院_6

fluorg_id_6 = tsr_6.loc[:, "fluorg_id_6"]
# print(fluorg_id_6)
print(fluorg_id_6.value_counts() / len(fluorg_id_6))
# print(fluorg_id_6.describe())

sns.countplot(fluorg_id_6)
plt.title("FLUORG_ID_6 - Barplot")
plt.xlabel('FLUORG_ID_6')
plt.ylabel('Number', rotation=0)
plt.show()

# FLUORG_TX_6
# FLURESULT_TX_6
# DEATH_DT_6
# DEATH_ID_6
# DEATHSK_ID_6
# DEATHO_TX_6

# 相關醫療紀錄2_6

ve_id_6 = tsr_6.loc[:, 've_id_6']
ve_id_6[(ve_id_6 == str(0)) | (ve_id_6 == 0)] = "N"

# print(ve_id_6)
print(ve_id_6.value_counts() / len(ve_id_6))
# print(ve_id_6.describe())

sns.countplot(ve_id_6)
plt.title("VE_ID_6 - Barplot")
plt.xlabel('VE_ID_6')
plt.ylabel('Number', rotation=0)
plt.show()

# 再中風_6

vers_fl_6 = tsr_6.loc[:, 'vers_fl_6']
vers_fl_6[(vers_fl_6 != "N") & (vers_fl_6 != "Y")] = np.nan
# print(vers_fl_6)
print(vers_fl_6.value_counts() / len(vers_fl_6))
# print(vers_fl_6.describe())

sns.countplot(vers_fl_6)
plt.title("VERS_FL_6 - Barplot")
plt.xlabel('VERS_FL_6')
plt.ylabel('Number', rotation=0)
plt.show()

# 再中風 (CI / CH)_6

verscich_id_6 = tsr_6.loc[:, 'verscich_id_6']
verscich_id_6[verscich_id_6 == str(1)] = 1
verscich_id_6[verscich_id_6 == str(2)] = 2
verscich_id_6[(verscich_id_6 != 1) & (verscich_id_6 != 2)] = np.nan
# print(verscich_id_6)
print(verscich_id_6.value_counts() / len(verscich_id_6))
# print(verscich_id_6.describe())

# verscich_id_6_labels = ["CI","CH"]
sns.countplot(verscich_id_6)  # .set_xticklabels(verscich_id_6_labels)
plt.title("VERSCICH_ID_6 - Barplot")
plt.xlabel('VERSCICH_ID_6')
plt.ylabel('Number', rotation=0)
plt.show()
verscich_id_6 = verscich_id_6.fillna(999)

# 再中風日期_6

vers_dt_6 = tsr_6.loc[:, 'vers_dt_6']
vers_dt_6 = pd.to_datetime(vers_dt_6, errors='coerce')
vers_dt_6[(vers_dt_6.dt.year < 2006) | (vers_dt_6.dt.year > 2023)] = np.nan
# print(vers_dt_6)
# print(vers_dt_6.value_counts() / len(vers_dt_6))
print(vers_dt_6.describe())

vers_dt_6.value_counts().plot()
plt.title("VERS_DT_6 - Lineplot")
plt.xlabel('VERS_DT_6')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 再中風醫院_6

versorg_id_6 = tsr_6.loc[:, 'versorg_id_6']
# print(versorg_id_6)
print(versorg_id_6.value_counts() / len(versorg_id_6))
# print(versorg_id_6.describe())

sns.countplot(versorg_id_6)
plt.title("VERSORG_ID_6 - Barplot")
plt.xlabel('VERSORG_ID_6')
plt.ylabel('Number', rotation=0)
plt.show()

# Ischemic Heart Disease_6

veihd_fl_6 = tsr_6.loc[:, 'veihd_fl_6']
# print(veihd_fl_6)
print(veihd_fl_6.value_counts() / len(veihd_fl_6))
# print(veihd_fl_6.describe())

sns.countplot(veihd_fl_6)
plt.title("VEIHD_FL_6 - Barplot")
plt.xlabel('VEIHD_FL_6')
plt.ylabel('Number', rotation=0)
plt.show()

# Types of Ischemic Heart Disease_6

veihd_id_6 = tsr_6.loc[:, 'veihd_id_6']
# print(veihd_id_6)
print(veihd_id_6.value_counts() / len(veihd_id_6))
# print(veihd_id_6.describe())

# veihd_id_6_labels = ["AMI","Angina"]
sns.countplot(veihd_id_6)  # .set_xticklabels(veihd_id_6_labels)
plt.title("VEIHD_ID_6 - Barplot")
plt.xlabel('VEIHD_ID_6')
plt.ylabel('Number', rotation=0)
plt.show()

# Ischemic Heart Disease日期_6

veihd_dt_6 = tsr_6.loc[:, 'veihd_dt_6']
veihd_dt_6 = pd.to_datetime(veihd_dt_6, errors='coerce')
veihd_dt_6[(veihd_dt_6.dt.year < 2006) | (veihd_dt_6.dt.year > 2023)] = np.nan
# print(veihd_dt_6)
# print(veihd_dt_6.value_counts() / len(veihd_dt_6))
print(veihd_dt_6.describe())

veihd_dt_6.value_counts().plot()
plt.title("VEIHD_DT_6 - Lineplot")
plt.xlabel('VEIHD_DT_6')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# Ischemic Heart Disease醫院_6

veihdorg_id_6 = tsr_6.loc[:, 'veihdorg_id_6']
# print(veihdorg_id_6)
print(veihdorg_id_6.value_counts() / len(veihdorg_id_6))
# print(veihdorg_id_6.describe())

sns.countplot(veihdorg_id_6)
plt.title("VEIHDORG_ID_6 - Barplot")
plt.xlabel('VEIHDORG_ID_6')
plt.ylabel('Number', rotation=0)
plt.show()

# mRS_6 (Outcome)

mrs_tx_6 = tsr_6.loc[:, 'mrs_tx_6']
mrs_tx_6 = mrs_tx_6.fillna(mrs_tx_6.mode()[0])
# print(mrs_tx_6)
print(mrs_tx_6.value_counts() / len(mrs_tx_6))
# print(mrs_tx_6.describe())

sns.countplot(mrs_tx_6)
plt.title("MRS_TX_6 - Barplot")
plt.xlabel('MRS_TX_6')
plt.ylabel('Number', rotation=0)
plt.show()

# TORG_TX_6
# VERSORG_TX_6
# VEIHDORG_TX_6
# index
# icase_id
# idcase_id
# cstatus_id
# org_id
# dctype24_id
# patient_id
# input_nm
# age_nm

# 身高

height_nm = tsr_6.loc[:, "height_nm"]

q1 = height_nm.quantile(0.25)
q3 = height_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
height_nm[(height_nm < inner_fence_low) | (height_nm > inner_fence_upp)] = np.nan

height_nm = height_nm.fillna(round(height_nm.mean(), 3))

# print(height_nm)
# print(height_nm.value_counts() / len(height_nm))
print(height_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

height_nm.plot.box(ax=ax1)
ax1.set_title("Height - Boxplot")
ax1.set_xlabel('Height')
ax1.set_ylabel('cm', rotation=0)
ax1.set_xticks([])

height_nm.plot.hist(ax=ax2, bins=20)
ax2.set_title("Height - Histogram")
ax2.set_xlabel('Height(cm)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# 體重

weight_nm = tsr_6.loc[:, "weight_nm"]

q1 = weight_nm.quantile(0.25)
q3 = weight_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
weight_nm[(weight_nm < inner_fence_low) | (weight_nm > inner_fence_upp)] = np.nan

weight_nm = weight_nm.fillna(round(weight_nm.mean(), 3))

# print(weight_nm)
# print(weight_nm.value_counts()len(weight_nm))
print(weight_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

weight_nm.plot.box(ax=ax1)
ax1.set_title("Weight - Boxplot")
ax1.set_xlabel('Weight')
ax1.set_ylabel('kg', rotation=0)
ax1.set_xticks([])

weight_nm.plot.hist(ax=ax2, bins=20)
ax2.set_title("Weight - Histogram")
ax2.set_xlabel('Weight(kg)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# 教育程度

edu_id = tsr_6.loc[:, "edu_id"]
edu_id = pd.to_numeric(edu_id, errors='coerce')
edu_id[(edu_id != 1) & (edu_id != 2) & (edu_id != 3) & (edu_id != 4) & (edu_id != 5) & (edu_id != 6) & (edu_id != 7)] = np.nan
edu_id = edu_id.fillna((edu_id.mode()[0]))
# print(edu_id)
print(edu_id.value_counts() / len(edu_id))
# print(edu_id.describe())

edu_id_labels = ["無", "國小", "國中", "高中職", "大學", "研究所", "不詳"]
sns.countplot(edu_id).set_xticklabels(edu_id_labels)
plt.title("edu_idcation Level - Barplot")
plt.xlabel('edu_idcation Level')
plt.ylabel('Number', rotation=0)
plt.show()

# 職業

pro_id = tsr_6.loc[:, "pro_id"]
pro_id = pd.to_numeric(pro_id, errors='coerce')
pro_id[(pro_id != 1) & (pro_id != 2) & (pro_id != 3) & (pro_id != 4) & (pro_id != 5) & (pro_id != 6) & (pro_id != 7) & (pro_id != 8) & (pro_id != 9) & (pro_id != 10) & (pro_id != 11) & (pro_id != 12)] = np.nan
pro_id = pro_id.fillna((pro_id.mode()[0]))
# print(pro_id)
print(pro_id.value_counts() / len(pro_id))
# print(pro_id.describe())

pro_id_labels = ["無", "軍", "公", "教", "工", "農", "商", "服務業", "學生", "退休", "不詳", "其他"]
sns.countplot(pro_id).set_xticklabels(pro_id_labels)
plt.title("Occupation - Barplot")
plt.xlabel('Occupation')
plt.ylabel('Number', rotation=0)
plt.show()

# proot_tx
# casememo_tx

# 入院方式

opc_id = tsr_6.loc[:, "opc_id"]
opc_id = pd.to_numeric(opc_id, errors="coerce")
opc_id[(opc_id != 1) & (opc_id != 2) & (opc_id != 3)] = np.nan
opc_id = opc_id.fillna((opc_id.mode()[0]))
# print(opc_id)
print(opc_id.value_counts() / len(opc_id))
# print(opc_id.describe())

opc_id_labels = ["Inpatient", "Outpatient", "Emergency Room"]
sns.countplot(opc_id).set_xticklabels(opc_id_labels)
plt.title("Types of Hospital Admission - Barplot")
plt.xlabel('Types of Hospital Admission')
plt.ylabel('Number', rotation=0)
plt.show()

# 不住院

ih_fl = tsr_6.loc[:, "ih_fl"]
ih_fl[(ih_fl != "1") & (ih_fl != np.nan)] = 0
ih_fl = ih_fl.fillna((ih_fl.mode()[0]))
# print(ih_fl)
print(ih_fl.value_counts() / len(ih_fl))
# print(ih_fl.describe())

ih_fl_labels = ["Hospitalised", "Not Hospitalised"]
sns.countplot(ih_fl).set_xticklabels(ih_fl_labels)
plt.title("Whether Being Hospitalised or Not - Barplot")
plt.xlabel('Whether Being Hospitalised or Not')
plt.ylabel('Number', rotation=0)
plt.show()

# 住院日期

ih_dt = tsr_6.loc[:, "ih_dt"]
ih_dt = pd.to_datetime(ih_dt, errors='coerce')
ih_dt = ih_dt.fillna((ih_dt.mode()[0]))
# print(ih_dt)
# print(ih_dt.value_counts() / len(ih_dt))
print(ih_dt.describe())

ih_dt.value_counts().plot()
plt.title("Date of Hospital Admission - Lineplot")
plt.xlabel('Date of Hospital Admission')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 離院日期

oh_dt = tsr_6.loc[:, "oh_dt"]
oh_dt = pd.to_datetime(oh_dt, errors='coerce')
oh_dt = oh_dt.fillna((oh_dt.mode()[0]))
oh_dt[(oh_dt.dt.year < 2006) | (oh_dt.dt.year > 2021) | (oh_dt < ih_dt)] = np.nan
# print(oh_dt)
# print(oh_dt.value_counts() / len(oh_dt))
# print(oh_dt.describe())

hospitalised_time = oh_dt - ih_dt
hospitalised_time = hospitalised_time.dt.days

q1 = hospitalised_time.quantile(0.25)
q3 = hospitalised_time.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
hospitalised_time[(hospitalised_time < inner_fence_low) | (hospitalised_time > inner_fence_upp)] = np.nan

hospitalised_time = hospitalised_time.fillna(round(hospitalised_time.mean(), 3))

# print(hospitalised_time.value_counts().sort_values(ascending= True))
print(hospitalised_time.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

hospitalised_time.plot.box(ax=ax1)
ax1.set_title("Days at Hospitals - Boxplot")
ax1.set_xlabel('Days at Hospitals')
ax1.set_ylabel('Days', rotation=0)
ax1.set_xticks([])

# hospitalised_time.plot.hist(ax = ax2, bins=100)
# plt.show()
hospitalised_time.plot.hist(ax=ax2, bins=100)
ax2.set_title("Days at Hospitals - Histogram")
ax2.set_xlabel('Days at Hospitals')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# 發病日期

onset_time = tsr_6.loc[:, ["onset_dt", "onseth_nm", "onsetm_nm"]]
onset_time.onset_dt = pd.to_datetime(onset_time.onset_dt, errors="coerce", format="%Y-%m-%d")
onset_time.onset_dt[(onset_time.onset_dt.dt.year < 2006) | (onset_time.onset_dt.dt.year > 2021)] = np.nan

onset_time['onseth_nm'] = pd.to_numeric(onset_time['onseth_nm'], errors='coerce')
onset_time['onseth_nm'][(onset_time['onseth_nm'] < 0) | (onset_time['onseth_nm'] > 24)] = np.nan
onset_time['onseth_nm'][onset_time['onseth_nm'] == 24] = 0
onset_time['onsetm_nm'] = pd.to_numeric(onset_time['onsetm_nm'], errors='coerce')
onset_time['onsetm_nm'][(onset_time['onsetm_nm'] < 0) | (onset_time['onsetm_nm'] > 60)] = np.nan
onset_time['onsetm_nm'][onset_time['onsetm_nm'] == 60] = 0

onset_time['onset_dt'] = onset_time['onset_dt'].fillna(onset_time['onset_dt'].mode()[0])
onset_time['onseth_nm'] = onset_time['onseth_nm'].fillna(onset_time['onseth_nm'].mean())
onset_time['onsetm_nm'] = onset_time['onsetm_nm'].fillna(onset_time['onsetm_nm'].mean())

onset = onset_time['onset_dt'].astype(str) + ' ' + onset_time['onseth_nm'].astype(int).map(str) + ':' + onset_time[
    'onsetm_nm'].astype(int).map(str)
onset_day = pd.to_datetime(onset, format='%Y/%m/%d %H:%M', errors='coerce')
# print(onset_day.value_counts() / len(onset_day))
print(onset_day.describe())

onset_day.value_counts().plot()
plt.title("Onset Date - Lineplot")
plt.xlabel('Onset Date')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 發病時間-時

onseth_nm = tsr_6.loc[:, "onseth_nm"]
onseth_nm = pd.to_numeric(onseth_nm, errors='coerce')
onseth_nm[(onseth_nm < 0) | (onseth_nm > 24)] = np.nan
onseth_nm[onseth_nm == 24] = 0
onseth_nm = onseth_nm.fillna(onseth_nm.mean())
# print(onseth_nm)
# print(onseth_nm.value_counts() / len(onseth_nm))
print(onseth_nm.describe())

# 發病時間-分

onsetm_nm = tsr_6.loc[:, "onsetm_nm"]
onsetm_nm = pd.to_numeric(onsetm_nm, errors='coerce')
onsetm_nm[(onsetm_nm < 0) | (onsetm_nm > 60)] = np.nan
onsetm_nm[onsetm_nm == 60] = 0
onsetm_nm = onsetm_nm.fillna(onsetm_nm.mean())
# print(onsetm_nm)
# print(onsetm_nm.value_counts()/len(onsetm_nm))
print(onsetm_nm.describe())

# 不確定發病時間

onset_fl = tsr_6.loc[:, "onset_fl"]
onset_fl[(onset_fl != "N") & (onset_fl != "Y")] = np.nan
onset_fl = onset_fl.fillna((onset_fl.mode()[0]))
# print(onset_fl)
print(onset_fl.value_counts() / len(onset_fl))
# print(onset_fl.describe())

sns.countplot(onset_fl)
plt.title("Onset Date Unknown - Barplot")
plt.xlabel('Onset Date Unknown')
plt.ylabel('Number', rotation=0)
plt.show()

# 門診/急診日期

ot_time = tsr_6.loc[:, ["ot_dt", "ottih_nm", "ottim_nm"]]
ot_time.ot_dt = pd.to_datetime(ot_time.ot_dt, errors="coerce", format="%Y-%m-%d")
ot_time.ot_dt[(ot_time.ot_dt.dt.year < 2006) | (ot_time.ot_dt.dt.year > 2021)] = np.nan

ot_time['ottih_nm'] = pd.to_numeric(ot_time['ottih_nm'], errors='coerce')
ot_time['ottih_nm'][(ot_time['ottih_nm'] < 0) | (ot_time['ottih_nm'] > 24)] = np.nan
ot_time['ottih_nm'][ot_time['ottih_nm'] == 24] = 0
ot_time['ottim_nm'] = pd.to_numeric(ot_time['ottim_nm'], errors='coerce')
ot_time['ottim_nm'][(ot_time['ottim_nm'] < 0) | (ot_time['ottim_nm'] > 60)] = np.nan
ot_time['ottim_nm'][ot_time['ottim_nm'] == 60] = 0

ot_time['ot_dt'] = ot_time['ot_dt'].fillna(ot_time['ot_dt'].mode()[0])
ot_time['ottih_nm'] = ot_time['ottih_nm'].fillna(ot_time['ottih_nm'].mean())
ot_time['ottim_nm'] = ot_time['ottim_nm'].fillna(ot_time['ottim_nm'].mean())

otset = ot_time['ot_dt'].astype(str) + ' ' + ot_time['ottih_nm'].astype(int).map(str) + ':' + ot_time[
    'ottim_nm'].astype(int).map(str)

otset_day = pd.to_datetime(otset, format='%Y/%m/%d %H:%M', errors='coerce')
# print(otset_day.value_counts() / len(otset_day))
print(otset_day.describe())

otset_day.value_counts().plot()
plt.title("Outpatient/ER Date - Lineplot")
plt.xlabel('Outpatient/ER Date')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 門診/急診時間-時

ottih_nm = tsr_6.loc[:, "ottih_nm"]
ottih_nm = pd.to_numeric(ottih_nm, errors='coerce')
ottih_nm[(ottih_nm < 0) | (ottih_nm > 24)] = np.nan
ottih_nm[ottih_nm == 24] = 0
ottih_nm = ottih_nm.fillna(ottih_nm.mean())
# print(ottih_nm)
# print(ottih_nm.value_counts() / len(ottih_nm))
print(ottih_nm.describe())

# 門診/急診時間-秒

ottim_nm = tsr_6.loc[:, "ottim_nm"]
ottim_nm = pd.to_numeric(ottim_nm, errors='coerce')
ottim_nm[(ottim_nm < 0) | (ottim_nm > 60)] = np.nan
ottim_nm[ottim_nm == 60] = 0
ottim_nm = ottim_nm.fillna(ottim_nm.mean())
# print(ottim_nm)
# print(ottim_nm.value_counts() / len(ottim_nm))
print(ottim_nm.describe())

# 不確定門/急診時間

ot_fl = tsr_6.loc[:, "ot_fl"]
ot_fl[(ot_fl != "N") & (ot_fl != "Y")] = np.nan
ot_fl = ot_fl.fillna(ot_fl.mode()[0])
# print(ot_fl)
print(ot_fl.value_counts() / len(ot_fl))
# print(ot_fl.describe())

sns.countplot(ot_fl)
plt.title("Outpatient/ER Date Unknown - Barplot")
plt.xlabel('Outpatient/ER Date Unknown')
plt.ylabel('Number', rotation=0)
plt.show()

# 第一次醫師檢視日期

flook_time = tsr_6.loc[:, ["flook_dt", "flookh_nm", "flookm_nm"]]
flook_time.flook_dt = pd.to_datetime(flook_time.flook_dt, errors="coerce", format="%Y-%m-%d")
flook_time.flook_dt[(flook_time.flook_dt.dt.year < 2006) | (flook_time.flook_dt.dt.year > 2021)] = np.nan

flook_time['flookh_nm'] = pd.to_numeric(flook_time['flookh_nm'], errors='coerce')
flook_time['flookh_nm'][(flook_time['flookh_nm'] < 0) | (flook_time['flookh_nm'] > 24)] = np.nan
flook_time['flookh_nm'][flook_time['flookh_nm'] == 24] = 0
flook_time['flookm_nm'] = pd.to_numeric(flook_time['flookm_nm'], errors='coerce')
flook_time['flookm_nm'][(flook_time['flookm_nm'] < 0) | (flook_time['flookm_nm'] > 60)] = np.nan
flook_time['flookm_nm'][flook_time['flookm_nm'] == 60] = 0

flook_time['flook_dt'] = flook_time['flook_dt'].fillna(flook_time['flook_dt'].mode()[0])
flook_time['flookh_nm'] = flook_time['flookh_nm'].fillna(flook_time['flookh_nm'].mean())
flook_time['flookm_nm'] = flook_time['flookm_nm'].fillna(flook_time['flookm_nm'].mean())

flookset = flook_time['flook_dt'].astype(str) + ' ' + flook_time['flookh_nm'].astype(int).map(str) + ':' + flook_time[
    'flookm_nm'].astype(int).map(str)

flookset_day = pd.to_datetime(flookset, format='%Y/%m/%d %H:%M', errors='coerce')
# print(flookset_day.value_counts() / len(flookset_day))
print(flookset_day.describe())

flookset_day.value_counts().plot()
plt.title("Doctors First Check Date - Lineplot")
plt.xlabel('Doctors First Check Date')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 第一次醫師檢視時間-時

flookh_nm = tsr_6.loc[:, "flookh_nm"]
flookh_nm = pd.to_numeric(flookh_nm, errors='coerce')
flookh_nm[(flookh_nm < 0) | (flookh_nm > 24)] = np.nan
flookh_nm[flookh_nm == 24] = 0
flookh_nm = flookh_nm.fillna(flookh_nm.mean())
# print(flookh_nm)
# print(flookh_nm.value_counts() / len(flookh_nm))
print(flookh_nm.describe())

# 第一次醫師檢視時間-分

flookm_nm = tsr_6.loc[:, "flookm_nm"]
flookm_nm = pd.to_numeric(flookm_nm, errors='coerce')
flookm_nm[(flookm_nm < 0) | (flookm_nm > 60)] = np.nan
flookm_nm[flookm_nm == 60] = 0
flookm_nm = flookm_nm.fillna(flookm_nm.mean())
# print(flookm_nm)
# print(flookm_nm.value_counts() / len(flookm_nm))
print(flookm_nm.describe())

# 不確定第一次醫師檢視時間

flook_fl = tsr_6.loc[:, "flook_fl"]
flook_fl[(flook_fl != "N") & (flook_fl != "Y")] = np.nan
flook_fl = flook_fl.fillna(flook_fl.mode()[0])
# print(flook_fl)
print(flook_fl.value_counts() / len(flook_fl))
# print(flook_fl.describe())

sns.countplot(flook_fl)
plt.title("Doctors First Check Date Unknown - Barplot")
plt.xlabel('Doctors First Check Date Unknown')
plt.ylabel('Number', rotation=0)
plt.show()

# 1st CT 日期

fct_time = tsr_6.loc[:, ["fct_dt", "fcth_nm", "fctm_nm"]]
fct_time.fct_dt = pd.to_datetime(fct_time.fct_dt, errors="coerce", format="%Y-%m-%d")
fct_time.fct_dt[(fct_time.fct_dt.dt.year < 2006) | (fct_time.fct_dt.dt.year > 2021)] = np.nan

fct_time['fcth_nm'] = pd.to_numeric(fct_time['fcth_nm'], errors='coerce')
fct_time['fcth_nm'][(fct_time['fcth_nm'] < 0) | (fct_time['fcth_nm'] > 24)] = np.nan
fct_time['fcth_nm'][fct_time['fcth_nm'] == 24] = 0
fct_time['fctm_nm'] = pd.to_numeric(fct_time['fctm_nm'], errors='coerce')
fct_time['fctm_nm'][(fct_time['fctm_nm'] < 0) | (fct_time['fctm_nm'] > 60)] = np.nan
fct_time['fctm_nm'][fct_time['fctm_nm'] == 60] = 0

fct_time['fct_dt'] = fct_time['fct_dt'].fillna(fct_time['fct_dt'].mode()[0])
fct_time['fcth_nm'] = fct_time['fcth_nm'].fillna(fct_time['fcth_nm'].mean())
fct_time['fctm_nm'] = fct_time['fctm_nm'].fillna(fct_time['fctm_nm'].mean())

fctset = fct_time['fct_dt'].astype(str) + ' ' + fct_time['fcth_nm'].astype(int).map(str) + ':' + fct_time[
    'fctm_nm'].astype(int).map(str)

fctset_day = pd.to_datetime(fctset, format='%Y/%m/%d %H:%M', errors='coerce')
# print(fctset_day.value_counts() / len(fctset_day))
print(fctset_day.describe())

fctset_day.value_counts().plot()
plt.title("First CT Date - Lineplot")
plt.xlabel('First CT Date')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 1st CT 時間-時

fcth_nm = tsr_6.loc[:, "fcth_nm"]
fcth_nm = pd.to_numeric(fcth_nm, errors='coerce')
fcth_nm[(fcth_nm < 0) | (fcth_nm > 24)] = np.nan
fcth_nm[fcth_nm == 24] = 0
fcth_nm = fcth_nm.fillna(fcth_nm.mean())
# print(fcth_nm)
# print(fcth_nm.value_counts() / len(fcth_nm))
print(fcth_nm.describe())

# 1st CT 時間-分

fctm_nm = tsr_6.loc[:, "fctm_nm"]
fctm_nm = pd.to_numeric(fctm_nm, errors='coerce')
fctm_nm[(fctm_nm < 0) | (fctm_nm > 60)] = np.nan
fctm_nm[fctm_nm == 60] = 0
fctm_nm = fctm_nm.fillna(fctm_nm.mean())
# print(fctm_nm)
# print(fctm_nm.value_counts() / len(fctm_nm))
print(fctm_nm.describe())

# 1st CT 外片

fctoh_fl = tsr_6.loc[:, "fctoh_fl"]
fctoh_fl[(fctoh_fl != "N") & (fctoh_fl != "Y")] = np.nan
fctoh_fl = fctoh_fl.fillna(fctoh_fl.mode()[0])
# print(fctoh_fl)
print(fctoh_fl.value_counts() / len(fctoh_fl))
# print(fctoh_fl.describe())

sns.countplot(fctoh_fl)
plt.title("First CT From Another Hospital - Barplot")
plt.xlabel('First CT From Another Hospital')
plt.ylabel('Number', rotation=0)
plt.show()

# ivtpath_fl
# ivtpaah_fl

# IV-tPA

ivtpath_id = tsr_6.loc[:, "ivtpath_id"]
ivtpath_id = pd.to_numeric(ivtpath_id, errors='coerce')
ivtpath_id = ivtpath_id.fillna(ivtpath_id.mode()[0])
# print(ivtpath_id)
print(ivtpath_id.value_counts() / len(ivtpath_id))
# print(ivtpath_id.describe())

ivtpath_id_labels = ["This hospital", "Another hospital"]
sns.countplot(ivtpath_id).set_xticklabels(ivtpath_id_labels)
plt.title("IV-tPA - Barplot")
plt.xlabel('IV-tPA')
plt.ylabel('Number', rotation=0)
plt.show()

# IV-tPA 日期

ivtpa_time = tsr_6.loc[:, ["ivtpa_dt", "ivtpah_nm", "ivtpam_nm"]]
ivtpa_time.ivtpa_dt = pd.to_datetime(ivtpa_time.ivtpa_dt, errors="coerce", format="%Y-%m-%d")
ivtpa_time.ivtpa_dt[(ivtpa_time.ivtpa_dt.dt.year < 2006) | (ivtpa_time.ivtpa_dt.dt.year > 2021)] = np.nan

ivtpa_time['ivtpah_nm'] = pd.to_numeric(ivtpa_time['ivtpah_nm'], errors='coerce')
ivtpa_time['ivtpah_nm'][(ivtpa_time['ivtpah_nm'] < 0) | (ivtpa_time['ivtpah_nm'] > 24)] = np.nan
ivtpa_time['ivtpah_nm'][ivtpa_time['ivtpah_nm'] == 24] = 0
ivtpa_time['ivtpam_nm'] = pd.to_numeric(ivtpa_time['ivtpam_nm'], errors='coerce')
ivtpa_time['ivtpam_nm'][(ivtpa_time['ivtpam_nm'] < 0) | (ivtpa_time['ivtpam_nm'] > 24)] = np.nan
ivtpa_time['ivtpam_nm'][ivtpa_time['ivtpam_nm'] == 24] = 0

ivtpa_time['ivtpa_dt'] = ivtpa_time['ivtpa_dt'].fillna(ivtpa_time['ivtpa_dt'].mode()[0])
ivtpa_time['ivtpah_nm'] = ivtpa_time['ivtpah_nm'].fillna(ivtpa_time['ivtpah_nm'].mean())
ivtpa_time['ivtpam_nm'] = ivtpa_time['ivtpam_nm'].fillna(ivtpa_time['ivtpam_nm'].mean())

ivtpaset = ivtpa_time['ivtpa_dt'].astype(str) + ' ' + ivtpa_time['ivtpah_nm'].astype(int).map(str) + ':' + ivtpa_time[
    'ivtpam_nm'].astype(int).map(str)

ivtpaset_day = pd.to_datetime(ivtpaset, format='%Y/%m/%d %H:%M', errors='coerce')
# print(ivtpaset_day.value_counts() / len(ivtpaset_day))
print(ivtpaset_day.describe())

ivtpaset_day.value_counts().plot()
plt.title("IV-tPA Date - Lineplot")
plt.xlabel('IV-tPA Date')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# IV-tPA 時間-時

ivtpah_nm = tsr_6.loc[:, "ivtpah_nm"]
ivtpah_nm = pd.to_numeric(ivtpah_nm, errors='coerce')
ivtpah_nm[(ivtpah_nm < 0) | (ivtpah_nm > 24)] = np.nan
ivtpah_nm[ivtpah_nm == 24] = 0
ivtpah_nm = ivtpah_nm.fillna(ivtpah_nm.mean())
# print(ivtpah_nm)
# print(ivtpah_nm.value_counts() / len(ivtpah_nm))
print(ivtpah_nm.describe())

# IV-tPA 時間-分

ivtpam_nm = tsr_6.loc[:, "ivtpam_nm"]
ivtpam_nm = pd.to_numeric(ivtpam_nm, errors='coerce')
ivtpam_nm[(ivtpam_nm < 0) | (ivtpam_nm > 60)] = np.nan
ivtpam_nm[ivtpam_nm == 60] = 0
ivtpam_nm = ivtpam_nm.fillna(ivtpam_nm.mean())
# print(ivtpam_nm)
# print(ivtpam_nm.value_counts() / len(ivtpam_nm))
print(ivtpam_nm.describe())

# Start IV-tPA mg

ivtpamg_nm = tsr_6.loc[:, "ivtpamg_nm"]
ivtpamg_nm = pd.to_numeric(ivtpamg_nm, errors='coerce')

q1 = ivtpamg_nm.quantile(0.25)
q3 = ivtpamg_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
ivtpamg_nm[(ivtpamg_nm < inner_fence_low) | (ivtpamg_nm > inner_fence_upp)] = np.nan

# ivtpamg_nm =ivtpamg_nm.fillna(ivtpamg_nm.mean())
ivtpamg_nm = ivtpamg_nm.fillna(0)

# print(ivtpamg_nm)
# print(ivtpamg_nm.value_counts() / len(ivtpamg_nm))
print(ivtpamg_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ivtpamg_nm.plot.box(ax=ax1)
ax1.set_title("Start IV-tPA - Boxplot")
ax1.set_xlabel('Start IV-tPA')
ax1.set_ylabel('mg', rotation=0)
ax1.set_xticks([])

# ivtpamg_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
ivtpamg_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("Start IV-tPA - Histogram")
ax2.set_xlabel('Start IV-tPA(mg)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# 未打IV-tPA 原因

nivtpa_id = tsr_6.loc[:, "nivtpa_id"]
nivtpa_id = pd.to_numeric(nivtpa_id, errors="coerce")
nivtpa_id[(nivtpa_id != 1) & (nivtpa_id != 2) & (nivtpa_id != 3)] = np.nan
nivtpa_id = nivtpa_id.fillna(0)
# print(nivtpa_id)
print(nivtpa_id.value_counts() / len(nivtpa_id))
print(nivtpa_id.describe())

nivtpa_id[nivtpa_id == 1] = "輸注本藥前，急性缺血性中風發作的時間已超過 3 小時或症狀發作時間不明"
nivtpa_id[nivtpa_id == 2] = "缺血性中風發作至到院2 小時內，但不符合下列施打條件"
nivtpa_id[nivtpa_id == 3] = "缺血性中風發作至到院2-3 小時內，但不符合下列施打條件"

sns.countplot(nivtpa_id, hue=nivtpa_id)
plt.title("IV-tPA not Injected Reasons - Barplot")
plt.xlabel('IV-tPA not Injected Reasons')
plt.ylabel('Number', rotation=0)
plt.xticks([])
plt.legend(loc=6, title="IV-tPA not Injected Reasons")
plt.show()

nivtpa_id[nivtpa_id == "輸注本藥前，急性缺血性中風發作的時間已超過 3 小時或症狀發作時間不明"] = 1
nivtpa_id[nivtpa_id == "缺血性中風發作至到院2 小時內，但不符合下列施打條件"] = 2
nivtpa_id[nivtpa_id == "缺血性中風發作至到院2-3 小時內，但不符合下列施打條件"] = 3
nivtpa_id = nivtpa_id.fillna(999)

# 輸注本藥前，急性缺血性腦中風的症狀已迅速改善或症狀輕微。（如NIHSS<6分）

nivtpa1_fl = tsr_6.loc[:, "nivtpa1_fl"]
nivtpa1_fl[nivtpa1_fl == str(0)] = int(0)
nivtpa1_fl[nivtpa1_fl == str(1)] = int(1)
nivtpa1_fl[nivtpa1_fl == int(0)] = "N"
nivtpa1_fl[nivtpa1_fl == int(1)] = "Y"
nivtpa1_fl[(nivtpa1_fl != "N") & (nivtpa1_fl != "Y")] = np.nan
# print(nivtpa1_fl)
print(nivtpa1_fl.value_counts() / len(nivtpa1_fl))
# print(nivtpa1_fl.describe())

sns.countplot(nivtpa1_fl)
plt.title("輸注本藥前，急性缺血性腦中風的症狀已迅速改善或症狀輕微。（如NIHSS<6分）- Barplot")
plt.xlabel('輸注本藥前，急性缺血性腦中風的症狀已迅速改善或症狀輕微。（如NIHSS<6分）')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa1_fl = nivtpa1_fl.fillna(999)

# 臨床或影像判定為嚴重之中風（如NIHSS>25）

nivtpa2_fl = tsr_6.loc[:, "nivtpa2_fl"]
nivtpa2_fl[nivtpa2_fl == str(0)] = int(0)
nivtpa2_fl[nivtpa2_fl == str(1)] = int(1)
nivtpa2_fl[nivtpa2_fl == int(0)] = "N"
nivtpa2_fl[nivtpa2_fl == int(1)] = "Y"
# print(nivtpa2_fl)
print(nivtpa2_fl.value_counts() / len(nivtpa2_fl))
# print(nivtpa2_fl.describe())

sns.countplot(nivtpa2_fl)
plt.title("臨床或影像判定為嚴重之中風（如NIHSS>25）- Barplot")
plt.xlabel('臨床或影像判定為嚴重之中風（如NIHSS>25）')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa2_fl = nivtpa2_fl.fillna(999)

# 年齡在18歲以下，或80歲以上

nivtpa3_fl = tsr_6.loc[:, "nivtpa3_fl"]
nivtpa3_fl[nivtpa3_fl == str(0)] = int(0)
nivtpa3_fl[nivtpa3_fl == str(1)] = int(1)
nivtpa3_fl[nivtpa3_fl == int(0)] = "N"
nivtpa3_fl[nivtpa3_fl == int(1)] = "Y"
# print(nivtpa3_fl)
print(nivtpa3_fl.value_counts() / len(nivtpa3_fl))
# print(nivtpa3_fl.describe())

sns.countplot(nivtpa3_fl)
plt.title("年齡在18歲以下，或80歲以上 - Barplot")
plt.xlabel('年齡在18歲以下，或80歲以上')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa3_fl = nivtpa3_fl.fillna(999)

# 過去曾中風且合併糖尿病

nivtpa4_fl = tsr_6.loc[:, "nivtpa4_fl"]
nivtpa4_fl[nivtpa4_fl == str(0)] = int(0)
nivtpa4_fl[nivtpa4_fl == str(1)] = int(1)
nivtpa4_fl[nivtpa4_fl == int(0)] = "N"
nivtpa4_fl[nivtpa4_fl == int(1)] = "Y"
# print(nivtpa4_fl)
print(nivtpa4_fl.value_counts() / len(nivtpa4_fl))
# print(nivtpa4_fl.describe())

sns.countplot(nivtpa4_fl)
plt.title("過去曾中風且合併糖尿病 - Barplot")
plt.xlabel('過去曾中風且合併糖尿病')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa4_fl = nivtpa4_fl.fillna(999)

# 最近3個月內有中風病史或有嚴重性頭部創傷

nivtpa5_fl = tsr_6.loc[:, "nivtpa5_fl"]
nivtpa5_fl[nivtpa5_fl == str(0)] = int(0)
nivtpa5_fl[nivtpa5_fl == str(1)] = int(1)
nivtpa5_fl[nivtpa5_fl == int(0)] = "N"
nivtpa5_fl[nivtpa5_fl == int(1)] = "Y"
# print(nivtpa5_fl)
print(nivtpa5_fl.value_counts() / len(nivtpa5_fl))
# print(nivtpa5_fl.describe())

sns.countplot(nivtpa5_fl)
plt.title("最近3個月內有中風病史或有嚴重性頭部創傷 - Barplot")
plt.xlabel('最近3個月內有中風病史或有嚴重性頭部創傷')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa5_fl = nivtpa5_fl.fillna(999)

# 中風發作時併發癲癇

nivtpa6_fl = tsr_6.loc[:, "nivtpa6_fl"]
nivtpa6_fl[nivtpa6_fl == str(0)] = int(0)
nivtpa6_fl[nivtpa6_fl == str(1)] = int(1)
nivtpa6_fl[nivtpa6_fl == int(0)] = "N"
nivtpa6_fl[nivtpa6_fl == int(1)] = "Y"
# print(nivtpa6_fl)
print(nivtpa6_fl.value_counts() / len(nivtpa6_fl))
# print(nivtpa6_fl.describe())

sns.countplot(nivtpa6_fl)
plt.title("中風發作時併發癲癇 - Barplot")
plt.xlabel('中風發作時併發癲癇')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa6_fl = nivtpa6_fl.fillna(999)

# 病人正接受口服抗凝血劑，如warfarin sodium（INR>1.3）

nivtpa7_fl = tsr_6.loc[:, "nivtpa7_fl"]
nivtpa7_fl[nivtpa7_fl == str(0)] = int(0)
nivtpa7_fl[nivtpa7_fl == str(1)] = int(1)
nivtpa7_fl[nivtpa7_fl == int(0)] = "N"
nivtpa7_fl[nivtpa7_fl == int(1)] = "Y"
# print(nivtpa7_fl)
print(nivtpa7_fl.value_counts() / len(nivtpa7_fl))
# print(nivtpa7_fl.describe())

sns.countplot(nivtpa7_fl)
plt.title("病人正接受口服抗凝血劑，如warfarin sodium（INR>1.3）- Barplot")
plt.xlabel('病人正接受口服抗凝血劑，如warfarin sodium（INR>1.3）')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa7_fl = nivtpa7_fl.fillna(999)

# 最近3個月內曾患胃腸道潰瘍

nivtpa8_fl = tsr_6.loc[:, "nivtpa8_fl"]
nivtpa8_fl[nivtpa8_fl == str(0)] = int(0)
nivtpa8_fl[nivtpa8_fl == str(1)] = int(1)
nivtpa8_fl[nivtpa8_fl == int(0)] = "N"
nivtpa8_fl[nivtpa8_fl == int(1)] = "Y"
# print(nivtpa8_fl)
print(nivtpa8_fl.value_counts() / len(nivtpa8_fl))
# print(nivtpa8_fl.describe())

sns.countplot(nivtpa8_fl)
plt.title("最近3個月內曾患胃腸道潰瘍- Barplot")
plt.xlabel('最近3個月內曾患胃腸道潰瘍')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa8_fl = nivtpa8_fl.fillna(999)

# 家屬拒絕

nivtpa9_fl = tsr_6.loc[:, "nivtpa9_fl"]
nivtpa9_fl[nivtpa9_fl == str(0)] = int(0)
nivtpa9_fl[nivtpa9_fl == str(1)] = int(1)
nivtpa9_fl[nivtpa9_fl == int(0)] = "N"
nivtpa9_fl[nivtpa9_fl == int(1)] = "Y"
# print(nivtpa9_fl)
print(nivtpa9_fl.value_counts() / len(nivtpa9_fl))
# print(nivtpa9_fl.describe())

sns.countplot(nivtpa9_fl)
plt.title("家屬拒絕 - Barplot")
plt.xlabel('家屬拒絕')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa9_fl = nivtpa9_fl.fillna(999)

# 輸注本藥前，缺血性腦中風發作的時間已超過3小時

nivtpa10_fl = tsr_6.loc[:, "nivtpa10_fl"]
nivtpa10_fl[nivtpa10_fl == str(0)] = int(0)
nivtpa10_fl[nivtpa10_fl == str(1)] = int(1)
nivtpa10_fl[nivtpa10_fl == int(0)] = "N"
nivtpa10_fl[nivtpa10_fl == int(1)] = "Y"
# print(nivtpa10_fl)
print(nivtpa10_fl.value_counts() / len(nivtpa10_fl))
# print(nivtpa10_fl.describe())

sns.countplot(nivtpa10_fl)
plt.title("輸注本藥前，缺血性腦中風發作的時間已超過3小時 - Barplot")
plt.xlabel('輸注本藥前，缺血性腦中風發作的時間已超過3小時')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa10_fl = nivtpa10_fl.fillna(999)

# 收縮壓 > 185mmhg或舒張壓 > 110mmhg

nivtpa11_fl = tsr_6.loc[:, "nivtpa11_fl"]
nivtpa11_fl[nivtpa11_fl == str(0)] = int(0)
nivtpa11_fl[nivtpa11_fl == str(1)] = int(1)
nivtpa11_fl[nivtpa11_fl == int(0)] = "N"
nivtpa11_fl[nivtpa11_fl == int(1)] = "Y"
nivtpa11_fl[(nivtpa11_fl != "N") & (nivtpa11_fl != "Y")] = np.nan
# print(nivtpa11_fl)
print(nivtpa11_fl.value_counts() / len(nivtpa11_fl))
# print(nivtpa11_fl.describe())

sns.countplot(nivtpa11_fl)
plt.title("收縮壓>185mmhg或舒張壓>110mmhg - Barplot")
plt.xlabel('收縮壓>185mmhg或舒張壓>110mmhg')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa11_fl = nivtpa11_fl.fillna(999)

# 不符合施打條件其他(未打IV-tPA 原因)

nivtpa99_fl = tsr_6.loc[:, "nivtpa99_fl"]
nivtpa99_fl[nivtpa99_fl == str(0)] = int(0)
nivtpa99_fl[nivtpa99_fl == str(1)] = int(1)
nivtpa99_fl[nivtpa99_fl == int(0)] = "N"
nivtpa99_fl[nivtpa99_fl == int(1)] = "Y"
nivtpa99_fl[(nivtpa99_fl != "N") & (nivtpa99_fl != "Y")] = np.nan
# print(nivtpa99_fl)
print(nivtpa99_fl.value_counts() / len(nivtpa99_fl))
# print(nivtpa99_fl.describe())

sns.countplot(nivtpa99_fl)
plt.title("不符合施打條件其他(未打IV-tPA 原因) - Barplot")
plt.xlabel('不符合施打條件其他(未打IV-tPA 原因)')
plt.ylabel('Number', rotation=0)
plt.show()
nivtpa99_fl = nivtpa99_fl.fillna(999)

# nivtpa99_tx

# GCS-E (eye opening)

gcse_nm = tsr_6.loc[:, "gcse_nm"]
gcse_nm = pd.to_numeric(gcse_nm, errors="coerce")
gcse_nm[(gcse_nm != 1) & (gcse_nm != 2) & (gcse_nm != 3) & (gcse_nm != 4)] = np.nan
gcse_nm = gcse_nm.fillna(gcse_nm.mode()[0])
# print(gcse_nm)
print(gcse_nm.value_counts() / len(gcse_nm))
print(gcse_nm.describe())

sns.countplot(gcse_nm)
plt.title("GCS-E - Barplot")
plt.xlabel('GCS-E')
plt.ylabel('Number', rotation=0)
plt.show()

# GCS-V (verbal response)

gcsv_nm = tsr_6.loc[:, "gcsv_nm"]
gcsv_nm = pd.to_numeric(gcsv_nm, errors="coerce")
gcsv_nm[(gcsv_nm != 1) & (gcsv_nm != 2) & (gcsv_nm != 3) & (gcsv_nm != 4) & (gcsv_nm != 5)] = np.nan
gcsv_nm = gcsv_nm.fillna(gcsv_nm.mode()[0])
# print(gcsv_nm)
print(gcsv_nm.value_counts() / len(gcsv_nm))
# print(gcsv_nm.describe())

sns.countplot(gcsv_nm)
plt.title("GCS-V - Barplot")
plt.xlabel('GCS-V')
plt.ylabel('Number', rotation=0)
plt.show()

# GCS-M (motor response)

gcsm_nm = tsr_6.loc[:, "gcsm_nm"]
gcsm_nm = pd.to_numeric(gcsm_nm, errors="coerce")
gcsm_nm[(gcsm_nm != 1) & (gcsm_nm != 2) & (gcsm_nm != 3) & (gcsm_nm != 4) & (gcsm_nm != 5) & (gcsm_nm != 6)] = np.nan
gcsm_nm = gcsm_nm.fillna(gcsm_nm.mode()[0])
# print(gscmnm)
print(gcsm_nm.value_counts() / len(gcsm_nm))
# print(gscmnm.describe())

sns.countplot(gcsm_nm)
plt.title("GCS-M - Barplot")
plt.xlabel('GCS-M')
plt.ylabel('Number', rotation=0)
plt.show()

# SBP

sbp_nm = tsr_6.loc[:, "sbp_nm"]

q1 = sbp_nm.quantile(0.25)
q3 = sbp_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
sbp_nm[(sbp_nm < inner_fence_low) | (sbp_nm > inner_fence_upp)] = np.nan

sbp_nm = sbp_nm.fillna(round(sbp_nm.mean(), 3))

# print(sbp_nm)
# print(sbp_nm.value_counts() / len(sbp_nm))
print(sbp_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

sbp_nm.plot.box(ax=ax1)
ax1.set_title("SBP - Boxplot")
ax1.set_xlabel('SBP')
ax1.set_ylabel('mmHg', rotation=0)
ax1.set_xticks([])

sbp_nm.plot.hist(ax=ax2, bins=50)
ax2.set_title("SBP - Histogram")
ax2.set_xlabel('SBP(mmHg)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# DBP

dbp_nm = tsr_6.loc[:, "dbp_nm"]

q1 = dbp_nm.quantile(0.25)
q3 = dbp_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
dbp_nm[(dbp_nm < inner_fence_low) | (dbp_nm > inner_fence_upp)] = np.nan

dbp_nm = dbp_nm.fillna(round(dbp_nm.mean(), 3))

# print(dbp_nm)
# print(dbp_nm.value_counts() / len(dbp_nm))
print(dbp_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

dbp_nm.plot.box(ax=ax1)
ax1.set_title("DBP - Boxplot")
ax1.set_xlabel('DBP')
ax1.set_ylabel('mmHg', rotation=0)
ax1.set_xticks([])

dbp_nm.plot.hist(ax=ax2, bins=50)
ax2.set_title("DBP - Histogram")
ax2.set_xlabel('DBP(mmHg)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# BT (bleeding time)

bt_nm = tsr_6.loc[:, "bt_nm"]

q1 = bt_nm.quantile(0.25)
q3 = bt_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
bt_nm[(bt_nm < inner_fence_low) | (bt_nm > inner_fence_upp)] = np.nan

bt_nm = bt_nm.fillna(round(bt_nm.mean(), 3))

# print(bt_nm)
# print(bt_nm.value_counts() / len(bt_nm))
print(bt_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

bt_nm.plot.box(ax=ax1)
ax1.set_title("BT - Boxplot")
ax1.set_xlabel('BT')
ax1.set_ylabel('min', rotation=0)
ax1.set_xticks([])

bt_nm.plot.hist(ax=ax2, bins=20)
ax2.set_title("BT - Histogram")
ax2.set_xlabel('BT(min)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Heart Rate

hr_nm = tsr_6.loc[:, "hr_nm"]

q1 = hr_nm.quantile(0.25)
q3 = hr_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
hr_nm[(hr_nm < inner_fence_low) | (hr_nm > inner_fence_upp)] = np.nan

hr_nm = hr_nm.fillna(round(hr_nm.mean(), 3))

# print(hr_nm)
# print(hr_nm.value_counts() / len(hr_nm))
print(hr_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

hr_nm.plot.box(ax=ax1)
ax1.set_title("Heart Rate - Boxplot")
ax1.set_xlabel('Heart Rate')
ax1.set_ylabel('frequency', rotation=0)
ax1.set_xticks([])

hr_nm.plot.hist(ax=ax2, bins=50)
ax2.set_title("Heart Rate - Histogram")
ax2.set_xlabel('Heart Rate(frequency)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Respiratory Rate

rr_nm = tsr_6.loc[:, "rr_nm"]

q1 = rr_nm.quantile(0.25)
q3 = rr_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
rr_nm[(rr_nm < inner_fence_low) | (rr_nm > inner_fence_upp)] = np.nan

rr_nm = rr_nm.fillna(round(rr_nm.mean(), 3))

# print(rr_nm)
# print(rr_nm.value_counts() / len(rr_nm))
print(rr_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

rr_nm.plot.box(ax=ax1)
ax1.set_title("Respiratory Rate - Boxplot")
ax1.set_xlabel('Respiratory Rate')
ax1.set_ylabel('Frequency', rotation=0)
ax1.set_xticks([])

rr_nm.plot.hist(ax=ax2)
ax2.set_title("Respiratory Rate - Histogram")
ax2.set_xlabel('Respiratory Rate(Frequency)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Final Stroke Diagnosis

icd_id = tsr_6.loc[:, "icd_id"]
icd_id[(icd_id != 1) & (icd_id != 2) & (icd_id != 3) & (icd_id != 4) & (icd_id != 99)] = np.nan
icd_id = icd_id.fillna(icd_id.mode()[0])
# print(icd_id)
print(icd_id.value_counts() / len(icd_id))
# print(icd_id.describe())

icd_id_labels = ["Infarct", "TIA", "ICH", "SAH", "Others"]
sns.countplot(icd_id).set_xticklabels(icd_id_labels)
plt.title("Final Stroke Diagnosis - Barplot")
plt.xlabel('Final Stroke Diagnosis')
plt.ylabel('Number', rotation=0)
plt.show()

# ICD_TX

# Final Stroke Diagnosis (TIA Subtypes)

icdtia_id = tsr_6.loc[:, "icdtia_id"]
icdtia_id[(icdtia_id != 1) & (icdtia_id != 2)] = np.nan
icdtia_id = icdtia_id.fillna(icdtia_id.mode()[0])
# print(icdtia_id)
print(icdtia_id.value_counts() / len(icdtia_id))
# print(icdtia_id.describe())

icdtia_id_labels = ["Carotid", "VB"]
sns.countplot(icdtia_id).set_xticklabels(icdtia_id_labels)
plt.title("Final Stroke Diagnosis (TIA Subtypes) - Barplot")
plt.xlabel('Final Stroke Diagnosis (TIA Subtypes)')
plt.ylabel('Number', rotation=0)
plt.show()

# ICDO_TX

# Ischemic Subtype(TOAST 分類)

toast_id = tsr_6.loc[:, "toast_id"]
toast_id = pd.to_numeric(toast_id, errors="coerce")
toast_id[(toast_id != 1) & (toast_id != 2) & (toast_id != 3) & (toast_id != 4) & (toast_id != 5)] = np.nan
# print(toast_id)
print(toast_id.value_counts() / len(toast_id))
# print(toast_id.describe())

toast_id[toast_id == 1] = "Large artery atherosclerosis"
toast_id[toast_id == 2] = "Small vessel occlusion"
toast_id[toast_id == 3] = "Specific etiology"
toast_id[toast_id == 4] = "Cardioembolism"
toast_id[toast_id == 5] = "Undetermined etiology"

sns.countplot(toast_id, hue=toast_id)
plt.title("Ischemic Subtype(TOAST Groups) - Barplot")
plt.xlabel('Ischemic Subtype(TOAST Groups)')
plt.ylabel('Number', rotation=0)
plt.xticks([])
plt.legend(loc=1, title="Ischemic Subtype(TOAST Groups)")
plt.show()

toast_id[toast_id == "Large artery atherosclerosis"] = 1
toast_id[toast_id == "Small vessel occlusion"] = 2
toast_id[toast_id == "Specific etiology"] = 3
toast_id[toast_id == "Cardioembolism"] = 4
toast_id[toast_id == "Undetermined etiology"] = 5
toast_id = toast_id.fillna(999)

# Large Artery Atherosclerosis (Extra)

toastle_fl = tsr_6.loc[:, "toastle_fl"]
toastle_fl[toastle_fl == str(0)] = int(0)
toastle_fl[toastle_fl == str(1)] = int(1)
toastle_fl[toastle_fl == int(0)] = "N"
toastle_fl[toastle_fl == int(1)] = "Y"
toastle_fl = toastle_fl.fillna(toastle_fl.mode()[0])
# print(toastle_fl)
print(toastle_fl.value_counts() / len(toastle_fl))
# print(toastle_fl.describe())

sns.countplot(toastle_fl)
plt.title("Large Artery Atherosclerosis (Extra) - Barplot")
plt.xlabel('Large Artery Atherosclerosis (Extra)')
plt.ylabel('Number', rotation=0)
plt.show()

# Large Artery Atherosclerosis (Intra)

toastli_fl = tsr_6.loc[:, "toastli_fl"]
toastli_fl[toastli_fl == str(0)] = int(0)
toastli_fl[toastli_fl == str(1)] = int(1)
toastli_fl[toastli_fl == int(0)] = "N"
toastli_fl[toastli_fl == int(1)] = "Y"
toastli_fl = toastli_fl.fillna(toastli_fl.mode()[0])
# print(toastli_fl)
print(toastli_fl.value_counts() / len(toastli_fl))
# print(toastli_fl.describe())

sns.countplot(toastli_fl)
plt.title("Large Artery Atherosclerosis (Intra) - Barplot")
plt.xlabel('Large Artery Atherosclerosis (Intra)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Cerebral Venous Thrombosis)

toastsce_fl = tsr_6.loc[:, "toastsce_fl"]
toastsce_fl[toastsce_fl == str(0)] = int(0)
toastsce_fl[toastsce_fl == str(1)] = int(1)
toastsce_fl[toastsce_fl == int(0)] = "N"
toastsce_fl[toastsce_fl == int(1)] = "Y"
toastsce_fl = toastsce_fl.fillna(toastsce_fl.mode()[0])
# print(toastsce_fl)
print(toastsce_fl.value_counts() / len(toastsce_fl))
# print(toastsce_fl.describe())

sns.countplot(toastsce_fl)
plt.title("Specific Etiology (Cerebral Venous Thrombosis) - Barplot")
plt.xlabel('Specific Etiology (Cerebral Venous Thrombosis)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Moyamoya Syndrome)

toastsmo_fl = tsr_6.loc[:, "toastsmo_fl"]
toastsmo_fl[toastsmo_fl == str(0)] = int(0)
toastsmo_fl[toastsmo_fl == str(1)] = int(1)
toastsmo_fl[toastsmo_fl == int(0)] = "N"
toastsmo_fl[toastsmo_fl == int(1)] = "Y"
toastsmo_fl = toastsmo_fl.fillna(toastsmo_fl.mode()[0])
# print(toastsmo_fl)
print(toastsmo_fl.value_counts() / len(toastsmo_fl))
# print(toastsmo_fl.describe())

sns.countplot(toastsmo_fl)
plt.title("Specific Etiology (Moyamoya Syndrome) - Barplot")
plt.xlabel('Specific Etiology (Moyamoya Syndrome)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Radiation)

toastsra_fl = tsr_6.loc[:, "toastsra_fl"]
toastsra_fl[toastsra_fl == str(0)] = int(0)
toastsra_fl[toastsra_fl == str(1)] = int(1)
toastsra_fl[toastsra_fl == int(0)] = "N"
toastsra_fl[toastsra_fl == int(1)] = "Y"
toastsra_fl = toastsra_fl.fillna(toastsra_fl.mode()[0])
# print(toastsra_fl)
print(toastsra_fl.value_counts() / len(toastsra_fl))
# print(toastsra_fl.describe())

sns.countplot(toastsra_fl)
plt.title("Specific Etiology (Radiation) - Barplot")
plt.xlabel('Specific Etiology (Radiation)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Dissection)

toastsdi_fl = tsr_6.loc[:, "toastsdi_fl"]
toastsdi_fl[toastsdi_fl == str(0)] = int(0)
toastsdi_fl[toastsdi_fl == str(1)] = int(1)
toastsdi_fl[toastsdi_fl == int(0)] = "N"
toastsdi_fl[toastsdi_fl == int(1)] = "Y"
toastsdi_fl = toastsdi_fl.fillna(toastsdi_fl.mode()[0])
# print(toastsdi_fl)
print(toastsdi_fl.value_counts() / len(toastsdi_fl))
# print(toastsdi_fl.describe())

sns.countplot(toastsdi_fl)
plt.title("Specific Etiology (Dissection) - Barplot")
plt.xlabel('Specific Etiology (Dissection)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Migraine)

toastsmi_fl = tsr_6.loc[:, "toastsmi_fl"]
toastsmi_fl[toastsmi_fl == str(0)] = int(0)
toastsmi_fl[toastsmi_fl == int(0)] = "N"
toastsmi_fl = toastsmi_fl.fillna("Y")
# print(toastsmi_fl)
print(toastsmi_fl.value_counts() / len(toastsmi_fl))
# print(toastsmi_fl.describe())

sns.countplot(toastsmi_fl)
plt.title("Specific Etiology (Migraine) - Barplot")
plt.xlabel('Specific Etiology (Migraine)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Antiphospholipid Ab Synd)

toastsantip_fl = tsr_6.loc[:, "toastsantip_fl"]
toastsantip_fl[toastsantip_fl == str(0)] = int(0)
toastsantip_fl[toastsantip_fl == str(1)] = int(1)
toastsantip_fl[toastsantip_fl == int(0)] = "N"
toastsantip_fl[toastsantip_fl == int(1)] = "Y"
toastsantip_fl = toastsantip_fl.fillna(toastsantip_fl.mode()[0])
# print(toastsantip_fl)
print(toastsantip_fl.value_counts() / len(toastsantip_fl))
# print(toastsantip_fl.describe())

sns.countplot(toastsantip_fl)
plt.title("Specific Etiology (Antiphospholipid Ab Synd) - Barplot")
plt.xlabel('Specific Etiology (Antiphospholipid Ab Synd)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Autoimmune Disease)

toastsau_fl = tsr_6.loc[:, "toastsau_fl"]
toastsau_fl[toastsau_fl == str(0)] = int(0)
toastsau_fl[toastsau_fl == str(1)] = int(1)
toastsau_fl[toastsau_fl == int(0)] = "N"
toastsau_fl[toastsau_fl == int(1)] = "Y"
toastsau_fl = toastsau_fl.fillna(toastsau_fl.mode()[0])
# print(toastsau_fl)
print(toastsau_fl.value_counts() / len(toastsau_fl))
# print(toastsau_fl.describe())

sns.countplot(toastsau_fl)
plt.title("Specific Etiology (Autoimmune Disease) - Barplot")
plt.xlabel('Specific Etiology (Autoimmune Disease)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Hyperfibrinogenemia)

toastshy_fl = tsr_6.loc[:, "toastshy_fl"]
toastshy_fl[toastshy_fl == str(0)] = int(0)
toastshy_fl[toastshy_fl == int(0)] = "N"
toastshy_fl = toastshy_fl.fillna("Y")
# print(toastshy_fl)
print(toastshy_fl.value_counts() / len(toastshy_fl))
# print(toastshy_fl.describe())

sns.countplot(toastshy_fl)
plt.title("Specific Etiology (Hyperfibrinogenemia) - Barplot")
plt.xlabel('Specific Etiology (Hyperfibrinogenemia)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Prot C/Prot S Deficiency)

toastspr_fl = tsr_6.loc[:, "toastspr_fl"]
toastspr_fl[toastspr_fl == str(0)] = int(0)
toastspr_fl[toastspr_fl == str(1)] = int(1)
toastspr_fl[toastspr_fl == int(0)] = "N"
toastspr_fl[toastspr_fl == int(1)] = "Y"
toastspr_fl = toastspr_fl.fillna(toastspr_fl.mode()[0])
# print(toastspr_fl)
print(toastspr_fl.value_counts() / len(toastspr_fl))
# print(toastspr_fl.describe())

sns.countplot(toastspr_fl)
plt.title("Specific Etiology (Prot C/Prot S Deficiency) - Barplot")
plt.xlabel('Specific Etiology (Prot C/Prot S Deficiency)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Antithrombin III Deficiency)

toastsantit_fl = tsr_6.loc[:, "toastsantit_fl"]
toastsantit_fl[toastsantit_fl == str(0)] = int(0)
toastsantit_fl[toastsantit_fl == str(1)] = int(1)
toastsantit_fl[toastsantit_fl == int(0)] = "N"
toastsantit_fl[toastsantit_fl == int(1)] = "Y"
toastsantit_fl = toastsantit_fl.fillna(toastsantit_fl.mode()[0])
# print(toastsantit_fl)
print(toastsantit_fl.value_counts() / len(toastsantit_fl))
# print(toastsantit_fl.describe())

sns.countplot(toastsantit_fl)
plt.title("Specific Etiology (Antithrombin III Deficiency) - Barplot")
plt.xlabel('Specific Etiology (Antithrombin III Deficiency)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Homocystinuria)

toastsho_fl = tsr_6.loc[:, "toastsho_fl"]
toastsho_fl[toastsho_fl == str(0)] = int(0)
toastsho_fl[toastsho_fl == str(1)] = int(1)
toastsho_fl[toastsho_fl == int(0)] = "N"
toastsho_fl[toastsho_fl == int(1)] = "Y"
toastsho_fl = toastsho_fl.fillna(toastsho_fl.mode()[0])
# print(toastsho_fl)
print(toastsho_fl.value_counts() / len(toastsho_fl))
# print(toastsho_fl.describe())

sns.countplot(toastsho_fl)
plt.title("Specific Etiology (Homocystinuria) - Barplot")
plt.xlabel('Specific Etiology (Homocystinuria)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Hypercoagulable State)

toastshys_fl = tsr_6.loc[:, "toastshys_fl"]
toastshys_fl[toastshys_fl == str(0)] = int(0)
toastshys_fl[toastshys_fl == str(1)] = int(1)
toastshys_fl[toastshys_fl == int(0)] = "N"
toastshys_fl[toastshys_fl == int(1)] = "Y"
toastshys_fl = toastshys_fl.fillna(toastshys_fl.mode()[0])
# print(toastshys_fl)
print(toastshys_fl.value_counts() / len(toastshys_fl))
# print(toastshys_fl.describe())

sns.countplot(toastshys_fl)
plt.title("Specific Etiology (Hypercoagulable State) - Barplot")
plt.xlabel('Specific Etiology (Hypercoagulable State)')
plt.ylabel('Number', rotation=0)
plt.show()

# Specific Etiology (Cancer)

toastsca_fl = tsr_6.loc[:, "toastsca_fl"]
toastsca_fl[toastsca_fl == str(0)] = int(0)
toastsca_fl[toastsca_fl == str(1)] = int(1)
toastsca_fl[toastsca_fl == int(0)] = "N"
toastsca_fl[toastsca_fl == int(1)] = "Y"
toastsca_fl = toastsca_fl.fillna(toastsca_fl.mode()[0])
# print(toastsca_fl)
print(toastsca_fl.value_counts() / len(toastsca_fl))
# print(toastsca_fl.describe())

sns.countplot(toastsca_fl)
plt.title("Specific Etiology (Cancer) - Barplot")
plt.xlabel('Specific Etiology (Cancer)')
plt.ylabel('Number', rotation=0)
plt.show()

# toastscat_tx

# Specific Etiology (Others)

toastso_fl = tsr_6.loc[:, "toastso_fl"]
toastso_fl[toastso_fl == str(0)] = int(0)
toastso_fl[toastso_fl == str(1)] = int(1)
toastso_fl[toastso_fl == int(0)] = "N"
toastso_fl[toastso_fl == int(1)] = "Y"
toastso_fl[(toastso_fl != "N") & (toastso_fl != "Y")] = np.nan
toastso_fl = toastso_fl.fillna(toastso_fl.mode()[0])
# print(toastso_fl)
print(toastso_fl.value_counts() / len(toastso_fl))
# print(toastso_fl.describe())

sns.countplot(toastso_fl)
plt.title("Specific Etiology (Others) - Barplot")
plt.xlabel('Specific Etiology (Others)')
plt.ylabel('Number', rotation=0)
plt.show()

# toastso_tx

# Undetermined Etiology

toastu_id = tsr_6.loc[:, "toastu_id"]
toastu_id[(toastu_id != 1) & (toastu_id != 2) & (toastu_id != 3)] = np.nan
toastu_id = toastu_id.fillna(toastu_id.mode()[0])
# print(toastu_id)
print(toastu_id.value_counts() / len(toastu_id))
# print(toastu_id.describe())

toastu_id_labels = ["Imcomplete study", "Conflict data", "Complete study"]
sns.countplot(toastu_id).set_xticklabels(toastu_id_labels)
plt.title("Undetermined Etiology - Barplot")
plt.xlabel('Undetermined Etiology')
plt.ylabel('Number', rotation=0)
plt.show()

# Cause of ICH ???

cich_id = tsr_6.loc[:, "cich_id"]
cich_id[(cich_id != 0) & (cich_id != 1) & (cich_id != 2)] = np.nan
# print(cich_id)
print(cich_id.value_counts() / len(cich_id))
# print(cich_id.describe())

# cich_id_labels = ["Hypertension","Non-Hypertension","2"], but documement says that the labels are 1 and 2
# sns.countplot(cich_id).set_xticklabels(cich_id_labels)
sns.countplot(cich_id)
plt.title("Cause of ICH - Barplot")
plt.xlabel('Cause of ICH')
plt.ylabel('Number', rotation=0)
plt.show()
cich_id = cich_id.fillna(999)

# Cause of SAH (蜘蛛膜下腔出血(Subarachnoid hemorrhage, SAH))

csah_id = tsr_6.loc[:, "csah_id"]
csah_id[(csah_id != 1) & (csah_id != 2) & (csah_id != 3) & (csah_id != 4) & (csah_id != 99)] = np.nan
# csah_id = csah_id.fillna(csah_id.mode()[0])
# print(csah_id)
print(csah_id.value_counts() / len(csah_id))
# print(csah_id.describe())

csah_id_labels = ["Aneurysm", "AVM", "Angio negative", "Angio undone", "Others"]
sns.countplot(csah_id).set_xticklabels(csah_id_labels)
plt.title("Cause of SAH - Barplot")
plt.xlabel('Cause of SAH')
plt.ylabel('Number', rotation=0)
plt.show()
csah_id = csah_id.fillna(999)

# csaho_tx

# Type of Heart Disease (NONE)

thd_id = tsr_6.loc[:, "thd_id"]
thd_id[thd_id == str(0)] = int(0)
thd_id[thd_id == str(1)] = int(1)
thd_id[thd_id == int(0)] = "N"
thd_id[thd_id == int(1)] = "Y"
thd_id = thd_id.fillna(thd_id.mode()[0])
# print(thd_id)
print(thd_id.value_counts() / len(thd_id))
# print(thd_id.describe())

sns.countplot(thd_id)
plt.title("Type of heart disease (NONE) - Barplot")
plt.xlabel('Type of heart disease (NONE)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (Atrial Fibrillation)

thda_fl = tsr_6.loc[:, "thda_fl"]
thda_fl[thda_fl == str(0)] = int(0)
thda_fl[thda_fl == str(1)] = int(1)
thda_fl[thda_fl == int(0)] = "N"
thda_fl[thda_fl == int(1)] = "Y"
thda_fl[thd_id == "Y"] = "N"
thda_fl = thda_fl.fillna(thda_fl.mode()[0])
# print(thda_fl)
print(thda_fl.value_counts() / len(thda_fl))
# print(thda_fl.describe())

sns.countplot(thda_fl)
plt.title("Type of Heart Disease (Atrial Fibrillation) - Barplot")
plt.xlabel('Type of Heart Disease (Atrial Fibrillation)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (Heart Failure)

thdh_fl = tsr_6.loc[:, "thdh_fl"]
thdh_fl[thdh_fl == str(0)] = int(0)
thdh_fl[thdh_fl == str(1)] = int(1)
thdh_fl[thdh_fl == int(0)] = "N"
thdh_fl[thdh_fl == int(1)] = "Y"
thdh_fl[thd_id == "Y"] = "N"
thdh_fl = thdh_fl.fillna(thdh_fl.mode()[0])
# print(thdh_fl)
print(thdh_fl.value_counts() / len(thdh_fl))
# print(thdh_fl.describe())

sns.countplot(thdh_fl)
plt.title("Type of Heart Disease (Heart Failure) - Barplot")
plt.xlabel('Type of Heart Disease (Heart Failure)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (Ischemic Heart - CAD, old MI)

thdi_fl = tsr_6.loc[:, "thdi_fl"]
thdi_fl[thdi_fl == str(0)] = int(0)
thdi_fl[thdi_fl == str(1)] = int(1)
thdi_fl[thdi_fl == int(0)] = "N"
thdi_fl[thdi_fl == int(1)] = "Y"
thdi_fl[thd_id == "Y"] = "N"
thdi_fl = thdi_fl.fillna(thdi_fl.mode()[0])
# print(thdi_fl)
print(thdi_fl.value_counts() / len(thdi_fl))
# print(thdi_fl.describe())

sns.countplot(thdi_fl)
plt.title("Type of Heart Disease (Ischemic Heart - CAD, old MI) - Barplot")
plt.xlabel('Type of Heart Disease (Ischemic Heart - CAD, old MI)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (Heart Disease - AMI<4W)

thdam_fl = tsr_6.loc[:, "thdam_fl"]
thdam_fl[thdam_fl == str(0)] = int(0)
thdam_fl[thdam_fl == str(1)] = int(1)
thdam_fl[thdam_fl == int(0)] = "N"
thdam_fl[thdam_fl == int(1)] = "Y"
thdam_fl[thd_id == "Y"] = "N"
thdam_fl = thdam_fl.fillna(thdam_fl.mode()[0])
# print(thdam_fl)
print(thdam_fl.value_counts() / len(thdam_fl))
# print(thdam_fl.describe())

sns.countplot(thdam_fl)
plt.title("Type of Heart Disease (Heart Disease - AMI<4W) - Barplot")
plt.xlabel('Type of Heart Disease (Heart Disease - AMI<4W)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (Valvular Replacement)

thdv_fl = tsr_6.loc[:, "thdv_fl"]
thdv_fl[thdv_fl == str(0)] = int(0)
thdv_fl[thdv_fl == str(1)] = int(1)
thdv_fl[thdv_fl == int(0)] = "N"
thdv_fl[thdv_fl == int(1)] = "Y"
thdv_fl[thd_id == "Y"] = "N"
thdv_fl = thdv_fl.fillna(thdv_fl.mode()[0])
# print(thdv_fl)
print(thdv_fl.value_counts() / len(thdv_fl))
# print(thdv_fl.describe())

sns.countplot(thdv_fl)
plt.title("Type of Heart Disease (Valvular Replacement) - Barplot")
plt.xlabel('Type of Heart Disease (Valvular Replacement)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (Endocarditis)

thde_fl = tsr_6.loc[:, "thde_fl"]
thde_fl[thde_fl == str(0)] = int(0)
thde_fl[thde_fl == str(1)] = int(1)
thde_fl[thde_fl == int(0)] = "N"
thde_fl[thde_fl == int(1)] = "Y"
thde_fl[thd_id == "Y"] = "N"
thde_fl = thde_fl.fillna(thde_fl.mode()[0])
# print(thde_fl)
print(thde_fl.value_counts() / len(thde_fl))
# print(thde_fl.describe())

sns.countplot(thde_fl)
plt.title("Type of Heart Disease (Endocarditis) - Barplot")
plt.xlabel('Type of Heart Disease (Endocarditis)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (Myxoma)

thdm_fl = tsr_6.loc[:, "thdm_fl"]
thdm_fl[thdm_fl == str(0)] = int(0)
thdm_fl[thdm_fl == str(1)] = int(1)
thdm_fl[thdm_fl == int(0)] = "N"
thdm_fl[thdm_fl == int(1)] = "Y"
thdm_fl[thd_id == "Y"] = "N"
thdm_fl = thdm_fl.fillna(thdm_fl.mode()[0])
# print(thdm_fl)
print(thdm_fl.value_counts() / len(thdm_fl))
# print(thdm_fl.describe())

sns.countplot(thdm_fl)
plt.title("Type of Heart Disease (Myxoma) - Barplot")
plt.xlabel('Type of Heart Disease (Myxoma)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (RHD)

thdr_fl = tsr_6.loc[:, "thdr_fl"]
thdr_fl[thdr_fl == str(0)] = int(0)
thdr_fl[thdr_fl == str(1)] = int(1)
thdr_fl[thdr_fl == int(0)] = "N"
thdr_fl[thdr_fl == int(1)] = "Y"
thdr_fl[thd_id == "Y"] = "N"
thdr_fl = thdr_fl.fillna(thdr_fl.mode()[0])
# print(thdr_fl)
print(thdr_fl.value_counts() / len(thdr_fl))
# print(thdr_fl.describe())

sns.countplot(thdr_fl)
plt.title("Type of Heart Disease (RHD) - Barplot")
plt.xlabel('Type of Heart Disease (RHD)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (Patent Foramen Ovale)

thdp_fl = tsr_6.loc[:, "thdp_fl"]
thdp_fl[thdp_fl == str(0)] = int(0)
thdp_fl[thdp_fl == str(1)] = int(1)
thdp_fl[thdp_fl == int(0)] = "N"
thdp_fl[thdp_fl == int(1)] = "Y"
thdp_fl[(thdp_fl != "N") & (thdp_fl != "Y")] = np.nan
thdp_fl[thd_id == "Y"] = "N"
thdp_fl = thdp_fl.fillna(thdp_fl.mode()[0])
# print(thdp_fl)
print(thdp_fl.value_counts() / len(thdp_fl))
# print(thdp_fl.describe())

sns.countplot(thdp_fl)
plt.title("Type of Heart Disease (Patent Foramen Ovale) - Barplot")
plt.xlabel('Type of Heart Disease (Patent Foramen Ovale)')
plt.ylabel('Number', rotation=0)
plt.show()

# Type of Heart Disease (Others)

thdoo_fl = tsr_6.loc[:, "thdoo_fl"]
thdoo_fl[thdoo_fl == str(0)] = int(0)
thdoo_fl[thdoo_fl == str(1)] = int(1)
thdoo_fl[thdoo_fl == int(0)] = "N"
thdoo_fl[thdoo_fl == int(1)] = "Y"
thdoo_fl[(thdoo_fl != "N") & (thdoo_fl != "Y")] = np.nan
thdoo_fl[thd_id == "Y"] = "N"
thdoo_fl = thdoo_fl.fillna(thdoo_fl.mode()[0])
# print(thdoo_fl)
print(thdoo_fl.value_counts() / len(thdoo_fl))
# print(thdoo_fl.describe())

sns.countplot(thdoo_fl)
plt.title("Type of Heart Disease (Others) - Barplot")
plt.xlabel('Type of Heart Disease (Others)')
plt.ylabel('Number', rotation=0)
plt.show()

# THDOO_TX

# Hospitalised (None)

trm_id = tsr_6.loc[:, "trm_id"]
trm_id[trm_id == str(0)] = int(0)
trm_id[trm_id == str(1)] = int(1)
trm_id[trm_id == int(0)] = "N"
trm_id[trm_id == int(1)] = "Y"
trm_id[(trm_id != "N") & (trm_id != "Y")] = np.nan
trm_id = trm_id.fillna(trm_id.mode()[0])
# print(trm_id)
print(trm_id.value_counts() / len(trm_id))
# print(trm_id.describe())

sns.countplot(trm_id)
plt.title("Hospitalised (NONE) - Barplot")
plt.xlabel('Hospitalised (NONE)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Antithrombotic Drugs Dtart Within 48h)

trman_fl = tsr_6.loc[:, "trman_fl"]
trman_fl[(trman_fl != "N") & (trman_fl != "Y")] = np.nan
trman_fl[trm_id == "Y"] = "N"
trman_fl = trman_fl.fillna(trman_fl.mode()[0])
# print(trman_fl)
print(trman_fl.value_counts() / len(trman_fl))
# print(trman_fl.describe())

sns.countplot(trman_fl)
plt.title("Hospitalised (Antithrombotic Drugs Dtart Within 48h) - Barplot")
plt.xlabel('Hospitalised (Antithrombotic Drugs Dtart Within 48h)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Aspirin)

trmas_fl = tsr_6.loc[:, "trmas_fl"]
trmas_fl[trm_id == "Y"] = "N"
trmas_fl = trmas_fl.fillna(trmas_fl.mode()[0])
# print(trmas_fl)
print(trmas_fl.value_counts() / len(trmas_fl))
# print(trmas_fl.describe())

sns.countplot(trmas_fl)
plt.title("Hospitalised (Aspirin) - Barplot")
plt.xlabel('Hospitalised (Aspirin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Ticlopidine)

trmti_fl = tsr_6.loc[:, "trmti_fl"]
trmti_fl[trm_id == "Y"] = "N"
trmti_fl = trmti_fl.fillna(trmti_fl.mode()[0])
# print(trmti_fl)
print(trmti_fl.value_counts() / len(trmti_fl))
# print(trmti_fl.describe())

sns.countplot(trmti_fl)
plt.title("Hospitalised (Ticlopidine) - Barplot")
plt.xlabel('Hospitalised (Ticlopidine)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Heparin)

trmhe_fl = tsr_6.loc[:, "trmhe_fl"]
trmhe_fl[trm_id == "Y"] = "N"
trmhe_fl = trmhe_fl.fillna(trmhe_fl.mode()[0])
# print(trmhe_fl)
print(trmhe_fl.value_counts() / len(trmhe_fl))
# print(trmhe_fl.describe())

sns.countplot(trmhe_fl)
plt.title("Hospitalised (Heparin) - Barplot")
plt.xlabel('Hospitalised (Heparin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Warfarin)

trmwa_fl = tsr_6.loc[:, "trmwa_fl"]
trmwa_fl[trm_id == "Y"] = "N"
trmwa_fl = trmwa_fl.fillna(trmwa_fl.mode()[0])
# print(trmwa_fl)
print(trmwa_fl.value_counts() / len(trmwa_fl))
# print(trmwa_fl.describe())

sns.countplot(trmwa_fl)
plt.title("Hospitalised (Warfarin) - Barplot")
plt.xlabel('Hospitalised (Warfarin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (IA Thrombolysis)

trmia_fl = tsr_6.loc[:, "trmia_fl"]
trmia_fl[trm_id == "Y"] = "N"
trmia_fl = trmia_fl.fillna(trmia_fl.mode()[0])
# print(trmia_fl)
print(trmia_fl.value_counts() / len(trmia_fl))
# print(trmia_fl.describe())

sns.countplot(trmia_fl)
plt.title("Hospitalised (IA Thrombolysis) - Barplot")
plt.xlabel('Hospitalised (IA Thrombolysis)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Foley)

trmfo_fl = tsr_6.loc[:, "trmfo_fl"]
trmfo_fl[trm_id == "Y"] = "N"
trmfo_fl = trmfo_fl.fillna(trmfo_fl.mode()[0])
# print(trmfo_fl)
print(trmfo_fl.value_counts() / len(trmfo_fl))
# print(trmfo_fl.describe())

sns.countplot(trmfo_fl)
plt.title("Hospitalised (Foley) - Barplot")
plt.xlabel('Hospitalised (Foley)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Transarterial - Embolization)

trmta_fl = tsr_6.loc[:, "trmta_fl"]
trmta_fl[trm_id == "Y"] = "N"
trmta_fl = trmta_fl.fillna(trmta_fl.mode()[0])
# print(trmta_fl)
print(trmta_fl.value_counts() / len(trmta_fl))
# print(trmta_fl.describe())

sns.countplot(trmta_fl)
plt.title("Hospitalised (Transarterial - Embolization) - Barplot")
plt.xlabel('Hospitalised (Transarterial - Embolization)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Sign DNR)

trmsd_fl = tsr_6.loc[:, "trmsd_fl"]
trmsd_fl[trm_id == "Y"] = "N"
trmsd_fl = trmsd_fl.fillna(trmsd_fl.mode()[0])
# print(trmsd_fl)
print(trmsd_fl.value_counts() / len(trmsd_fl))
# print(trmsd_fl.describe())

sns.countplot(trmsd_fl)
plt.title("Hospitalised (Sign DNR) - Barplot")
plt.xlabel('Hospitalised (Sign DNR)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Rehab)

trmre_fl = tsr_6.loc[:, "trmre_fl"]
trmre_fl[trm_id == "Y"] = "N"
trmre_fl = trmre_fl.fillna(trmre_fl.mode()[0])
# print(trmre_fl)
print(trmre_fl.value_counts() / len(trmre_fl))
# print(trmre_fl.describe())

sns.countplot(trmre_fl)
plt.title("Hospitalised (Rehab) - Barplot")
plt.xlabel('Hospitalised (Rehab)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Endovascular Treatment)

trmen_fl = tsr_6.loc[:, "trmen_fl"]
trmen_fl[(trmen_fl != "N") & (trmen_fl != "Y")] = np.nan
trmen_fl[trm_id == "Y"] = "N"
trmen_fl = trmen_fl.fillna(trmen_fl.mode()[0])
# print(trmen_fl)
print(trmen_fl.value_counts() / len(trmen_fl))
# print(trmen_fl.describe())

sns.countplot(trmen_fl)
plt.title("Hospitalised (Endovascular Treatment) - Barplot")
plt.xlabel('Hospitalised (Endovascular Treatment)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Endovascular Treatment Options)

trmen_id = tsr_6.loc[:, "trmen_id"]
trmen_id[trmen_id == str(1)] = int(1)
trmen_id[trmen_id == str(2)] = int(2)
trmen_id[trmen_id == str(3)] = int(3)
trmen_id[(trmen_id != 1) & (trmen_id != 2) & (trmen_id != 3)] = np.nan
trmen_id = trmen_id.fillna(trmen_id.mode()[0])
trmen_id[trmen_fl == "N"] = np.nan
# print(trmen_id)
print(trmen_id.value_counts() / len(trmen_id))
# print(trmen_id.describe())

# trmen_id_labels = ["Aneurysm","AVM","Stenting"]
sns.countplot(trmen_id)  # .set_xticklabels(trmen_id_labels)
plt.title("Hospitalised (Endovascular Treatment Options) - Barplot")
plt.xlabel('Hospitalised (Endovascular Treatment Options)')
plt.ylabel('Number', rotation=0)
plt.show()
trmen_id = trmen_id.fillna(999)

# Hospitalised (Aggrenox)

trmag_fl = tsr_6.loc[:, "trmag_fl"]
trmag_fl[trm_id == "Y"] = "N"
trmag_fl = trmag_fl.fillna(trmag_fl.mode()[0])
# print(trmag_fl)
print(trmag_fl.value_counts() / len(trmag_fl))
# print(trmag_fl.describe())

sns.countplot(trmag_fl)
plt.title("Hospitalised (Aggrenox) - Barplot")
plt.xlabel('Hospitalised (Aggrenox)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Clopidogrel)

trmcl_fl = tsr_6.loc[:, "trmcl_fl"]
trmcl_fl[trm_id == "Y"] = "N"
trmcl_fl = trmcl_fl.fillna(trmcl_fl.mode()[0])
# print(trmcl_fl)
print(trmcl_fl.value_counts() / len(trmcl_fl))
# print(trmcl_fl.describe())

sns.countplot(trmcl_fl)
plt.title("Hospitalised (Clopidogrel) - Barplot")
plt.xlabel('Hospitalised (Clopidogrel)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Pletaal)

trmpl_fl = tsr_6.loc[:, "trmpl_fl"]
trmpl_fl[trm_id == "Y"] = "N"
trmpl_fl = trmpl_fl.fillna(trmpl_fl.mode()[0])
# print(trmpl_fl)
print(trmpl_fl.value_counts() / len(trmpl_fl))
# print(trmpl_fl.describe())

sns.countplot(trmpl_fl)
plt.title("Hospitalised (Pletaal) - Barplot")
plt.xlabel('Hospitalised (Pletaal)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (LMW Heparin)

trmlm_fl = tsr_6.loc[:, "trmlm_fl"]
trmlm_fl[trm_id == "Y"] = "N"
trmlm_fl = trmlm_fl.fillna(trmlm_fl.mode()[0])
# print(trmlm_fl)
print(trmlm_fl.value_counts() / len(trmlm_fl))
# print(trmlm_fl.describe())

sns.countplot(trmlm_fl)
plt.title("Hospitalised (LMW Heparin) - Barplot")
plt.xlabel('Hospitalised (LMW Heparin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (IV t-PA)

trmiv_fl = tsr_6.loc[:, "trmiv_fl"]
trmiv_fl[trm_id == "Y"] = "N"
trmiv_fl = trmiv_fl.fillna(trmiv_fl.mode()[0])
# print(trmiv_fl)
print(trmiv_fl.value_counts() / len(trmiv_fl))
# print(trmiv_fl.describe())

sns.countplot(trmiv_fl)
plt.title("Hospitalised (IV t-PA) - Barplot")
plt.xlabel('Hospitalised (IV t-PA)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Ventilator)

trmve_fl = tsr_6.loc[:, "trmve_fl"]
trmve_fl[trm_id == "Y"] = "N"
trmve_fl = trmve_fl.fillna(trmve_fl.mode()[0])
# print(trmve_fl)
print(trmve_fl.value_counts() / len(trmve_fl))
# print(trmve_fl.describe())

sns.countplot(trmve_fl)
plt.title("Hospitalised (Ventilator) - Barplot")
plt.xlabel('Hospitalised (Ventilator)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Nasogastric Tube)

trmng_fl = tsr_6.loc[:, "trmng_fl"]
trmng_fl[trm_id == "Y"] = "N"
trmng_fl = trmng_fl.fillna(trmng_fl.mode()[0])
# print(trmng_fl)
print(trmng_fl.value_counts() / len(trmng_fl))
# print(trmng_fl.describe())

sns.countplot(trmng_fl)
plt.title("Hospitalised (Nasogastric Tube) - Barplot")
plt.xlabel('Hospitalised (Nasogastric Tube)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Dysphagia Screen)

trmdy_fl = tsr_6.loc[:, "trmdy_fl"]
trmdy_fl[trm_id == "Y"] = "N"
trmdy_fl = trmdy_fl.fillna(trmdy_fl.mode()[0])
# print(trmdy_fl)
print(trmdy_fl.value_counts() / len(trmdy_fl))
# print(trmdy_fl.describe())

sns.countplot(trmdy_fl)
plt.title("Hospitalised (Dysphagia Screen) - Barplot")
plt.xlabel('Hospitalised (Dysphagia Screen)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Admission to ICU)

trmicu_fl = tsr_6.loc[:, "trmicu_fl"]
trmicu_fl[trm_id == "Y"] = "N"
trmicu_fl = trmicu_fl.fillna(trmicu_fl.mode()[0])
# print(trmicu_fl)
print(trmicu_fl.value_counts() / len(trmicu_fl))
# print(trmicu_fl.describe())

sns.countplot(trmicu_fl)
plt.title("Hospitalised (Admission to ICU) - Barplot")
plt.xlabel('Hospitalised (Admission to ICU)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Smoking Cessation Counseling)

trmsm_fl = tsr_6.loc[:, "trmsm_fl"]
trmsm_fl[trm_id == "Y"] = "N"
trmsm_fl = trmsm_fl.fillna(trmsm_fl.mode()[0])
# print(trmsm_fl)
print(trmsm_fl.value_counts() / len(trmsm_fl))
# print(trmsm_fl.describe())

sns.countplot(trmsm_fl)
plt.title("Hospitalised (Smoking Cessation Counseling) - Barplot")
plt.xlabel('Hospitalised (Smoking Cessation Counseling)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Education About Stroke)

trmed_fl = tsr_6.loc[:, "trmed_fl"]
trmed_fl[trm_id == "Y"] = "N"
trmed_fl = trmed_fl.fillna(trmed_fl.mode()[0])
# print(trmed_fl)
print(trmed_fl.value_counts() / len(trmed_fl))
# print(trmed_fl.describe())

sns.countplot(trmed_fl)
plt.title("Hospitalised (Education About Stroke) - Barplot")
plt.xlabel('Hospitalised (Education About Stroke)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Operation for)

trmop_fl = tsr_6.loc[:, "trmop_fl"]
trmop_fl[(trmop_fl != "N") & (trmop_fl != "Y")] = np.nan
trmop_fl[trm_id == "Y"] = "N"
trmop_fl = trmop_fl.fillna(trmop_fl.mode()[0])
# print(trmop_fl)
print(trmop_fl.value_counts() / len(trmop_fl))
# print(trmop_fl.describe())

sns.countplot(trmop_fl)
plt.title("Hospitalised (Operation for) - Barplot")
plt.xlabel('Hospitalised (Operation for)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hospitalised (Operation for Options)

trmop_id = tsr_6.loc[:, "trmop_id"]
trmop_id = pd.to_numeric(trmop_id, errors="coerce")
trmop_id = trmop_id.fillna(trmop_id.mode()[0])
trmop_id[trmop_fl == "N"] = np.nan
# print(trmop_id)
print(trmop_id.value_counts() / len(trmop_id))
# print(trmop_id.describe())

# trmop_id_labels = ["Infracion","ICH","Carotid Stenosis \n (eg:Endarterectomy)","Aneurysm","AVM"]
trmop_id_labels = ["Infracion", "ICH", "Aneurysm"]
sns.countplot(trmop_id).set_xticklabels(trmop_id_labels)
plt.title("Hospitalised (Operation for Options) - Barplot")
plt.xlabel('Hospitalised (Operation for Options)')
plt.ylabel('Number', rotation=0)
plt.show()
trmop_id = trmop_id.fillna(999)

# Hospitalised (Others)

trmot_fl = tsr_6.loc[:, "trmot_fl"]
trmot_fl[(trmot_fl != "N") & (trmot_fl != "Y")] = np.nan
trmot_fl[trm_id == "Y"] = "N"
trmot_fl = trmot_fl.fillna(trmot_fl.mode()[0])
# print(trmot_fl)
print(trmot_fl.value_counts() / len(trmot_fl))
# print(trmot_fl.describe())

sns.countplot(trmot_fl)
plt.title("Hospitalised (Others) - Barplot")
plt.xlabel('Hospitalised (Others)')
plt.ylabel('Number', rotation=0)
plt.show()

# TRMOT_TX

# Discharged (None)

om_fl = tsr_6.loc[:, "om_fl"]
om_fl = om_fl.fillna(om_fl.mode()[0])
# print(om_fl)
print(om_fl.value_counts() / len(om_fl))
# print(om_fl.describe())

sns.countplot(om_fl)
plt.title("Discharged (Others) - Barplot")
plt.xlabel('Discharged (Others)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Aspirin)

omas_fl = tsr_6.loc[:, "omas_fl"]
omas_fl[om_fl == "Y"] = "N"
omas_fl = omas_fl.fillna(omas_fl.mode()[0])
# print(omas_fl)
print(omas_fl.value_counts() / len(omas_fl))
# print(omas_fl.describe())

sns.countplot(omas_fl)
plt.title("Discharged (Aspirin) - Barplot")
plt.xlabel('Discharged (Aspirin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Aggrenox)

omag_fl = tsr_6.loc[:, "omag_fl"]
omag_fl[om_fl == "Y"] = "N"
omag_fl = omag_fl.fillna(omag_fl.mode()[0])
# print(omag_fl)
print(omag_fl.value_counts() / len(omag_fl))
# print(omag_fl.describe())

sns.countplot(omag_fl)
plt.title("Discharged (Aggrenox) - Barplot")
plt.xlabel('Discharged (Aggrenox)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Ticlopidine)

omti_fl = tsr_6.loc[:, "omti_fl"]
omti_fl[om_fl == "Y"] = "N"
omti_fl = omti_fl.fillna(omti_fl.mode()[0])
# print(omti_fl)
print(omti_fl.value_counts() / len(omti_fl))
# print(omti_fl.describe())

sns.countplot(omti_fl)
plt.title("Discharged (Ticlopidine) - Barplot")
plt.xlabel('Discharged (Ticlopidine)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Clopidogrel)

omcl_fl = tsr_6.loc[:, "omcl_fl"]
omcl_fl[(omcl_fl != "N") & (omcl_fl != "Y")] = np.nan
omcl_fl[trm_id == "Y"] = "N"
omcl_fl = omcl_fl.fillna(omcl_fl.mode()[0])
# print(omcl_fl)
print(omcl_fl.value_counts() / len(omcl_fl))
# print(omcl_fl.describe())

sns.countplot(omcl_fl)
plt.title("Discharged (Clopidogrel) - Barplot")
plt.xlabel('Discharged (Clopidogrel)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Pletaal)

ompl_fl = tsr_6.loc[:, "ompl_fl"]
ompl_fl[om_fl == "Y"] = "N"
ompl_fl = ompl_fl.fillna(ompl_fl.mode()[0])
# print(ompl_fl)
print(ompl_fl.value_counts() / len(ompl_fl))
# print(ompl_fl.describe())

sns.countplot(ompl_fl)
plt.title("Discharged (Pletaal) - Barplot")
plt.xlabel('Discharged (Pletaal)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Anti H/T Drug)

omanh_fl = tsr_6.loc[:, "omanh_fl"]
omanh_fl[(omanh_fl != "Y") & (omanh_fl != "N")] = np.nan
omanh_fl[om_fl == "Y"] = "N"
omanh_fl = omanh_fl.fillna(omanh_fl.mode()[0])
# print(omanh_fl)
print(omanh_fl.value_counts() / len(omanh_fl))
# print(omanh_fl.describe())

sns.countplot(omanh_fl)
plt.title("Discharged (Anti H/T Drug) - Barplot")
plt.xlabel('Discharged (Anti H/T Drug)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Warfarin)

omwa_fl = tsr_6.loc[:, "omwa_fl"]
omwa_fl[(omwa_fl != "N") & (omwa_fl != "Y")] = np.nan
omwa_fl[om_fl == "Y"] = "N"
omwa_fl = omwa_fl.fillna(omwa_fl.mode()[0])
# print(omwa_fl)
print(omwa_fl.value_counts() / len(omwa_fl))
# print(omwa_fl.describe())

sns.countplot(omwa_fl)
plt.title("Discharged (Warfarin) - Barplot")
plt.xlabel('Discharged (Warfarin)')
plt.ylabel('Number', rotation=0)
plt.show()

# OMWA_TX(離院時INR)

# Discharged (Anti DM Drug)

omand_fl = tsr_6.loc[:, "omand_fl"]
omand_fl[om_fl == "Y"] = "N"
omand_fl = omand_fl.fillna(omand_fl.mode()[0])
# print(omand_fl)
print(omand_fl.value_counts() / len(omand_fl))
# print(omand_fl.describe())

sns.countplot(omand_fl)
plt.title("Discharged (Anti DM Drug) - Barplot")
plt.xlabel('Discharged (Anti DM Drug)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Anti DM Drug - Oral)

omora_fl = tsr_6.loc[:, "omora_fl"]
omora_fl[(omora_fl != "Y") & (omora_fl != "N")] = np.nan
omora_fl[om_fl == "Y"] = "N"
omora_fl = omora_fl.fillna(omora_fl.mode()[0])
# print(omora_fl)
print(omora_fl.value_counts() / len(omora_fl))
# print(omora_fl.describe())

sns.countplot(omora_fl)
plt.title("Discharged (Anti DM Drug - Oral) - Barplot")
plt.xlabel('Discharged (Anti DM Drug - Oral)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Anti DM Drug - Insulin)

omins_fl = tsr_6.loc[:, "omins_fl"]
omins_fl[(omins_fl != "Y") & (omins_fl != "N")] = np.nan
omins_fl[om_fl == "Y"] = "N"
omins_fl = omins_fl.fillna(omins_fl.mode()[0])
# print(omins_fl)
print(omins_fl.value_counts() / len(omins_fl))
# print(omins_fl.describe())

sns.countplot(omins_fl)
plt.title("Discharged (Anti DM Drug - Insulin) - Barplot")
plt.xlabel('Discharged (Anti DM Drug - Insulin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Lipid Lower Drug)

omli_fl = tsr_6.loc[:, "omli_fl"]
omli_fl[(omli_fl != "Y") & (omli_fl != "N")] = np.nan
omli_fl[om_fl == "Y"] = "N"
omli_fl = omli_fl.fillna(omli_fl.mode()[0])
# print(omli_fl)
print(omli_fl.value_counts() / len(omli_fl))
# print(omli_fl.describe())

sns.countplot(omli_fl)
plt.title("Discharged (Lipid Lower Drug) - Barplot")
plt.xlabel('Discharged (Lipid Lower Drug)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Lipid Lower Drug - Statin)

omst_fl = tsr_6.loc[:, "omst_fl"]
omst_fl[om_fl == "Y"] = "N"
omst_fl = omst_fl.fillna(omst_fl.mode()[0])
# print(omst_fl)
print(omst_fl.value_counts() / len(omst_fl))
# print(omst_fl.describe())

sns.countplot(omst_fl)
plt.title("Discharged (Lipid Lower Drug - Statin) - Barplot")
plt.xlabel('Discharged (Lipid Lower Drug - Statin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Lipid Lower Drug - Non-Statin)

omns_fl = tsr_6.loc[:, "omns_fl"]
omns_fl[(omns_fl != "N") & (omns_fl != "Y")] = np.nan
omns_fl[om_fl == "Y"] = "N"
omns_fl = omns_fl.fillna(omns_fl.mode()[0])
# print(omns_fl)
print(omns_fl.value_counts() / len(omns_fl))
# print(omns_fl.describe())

sns.countplot(omns_fl)
plt.title("Discharged (Lipid Lower Drug - Non-Statin) - Barplot")
plt.xlabel('Discharged (Lipid Lower Drug - Non-Statin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (Others)

omliot_fl = tsr_6.loc[:, "omliot_fl"]
omliot_fl[(omliot_fl != "N") & (omliot_fl != "Y")] = np.nan
omliot_fl[om_fl == "Y"] = "N"
omliot_fl = omliot_fl.fillna(omliot_fl.mode()[0])
# print(omliot_fl)
print(omliot_fl.value_counts() / len(omliot_fl))
# print(omliot_fl.describe())

sns.countplot(omliot_fl)
plt.title("Discharged (Others) - Barplot")
plt.xlabel('Discharged (Others)')
plt.ylabel('Number', rotation=0)
plt.show()

# OMLIOT_TX

# Discharged (Others2)

omliot2_fl = tsr_6.loc[:, "omliot2_fl"]
omliot2_fl[(omliot2_fl != "N") & (omliot2_fl != "Y")] = np.nan
omliot2_fl[om_fl == "Y"] = "N"
omliot2_fl = omliot2_fl.fillna(omliot2_fl.mode()[0])
# print(omliot2_fl)
print(omliot2_fl.value_counts() / len(omliot2_fl))
# print(omliot2_fl.describe())

sns.countplot(omliot2_fl)
plt.title("Discharged (Others2) - Barplot")
plt.xlabel('Discharged (Others2)')
plt.ylabel('Number', rotation=0)
plt.show()

# OMLIOT2_TX

# Discharged (有相關原因未使用抗血栓藥物)

omad_fl = tsr_6.loc[:, "omad_fl"]
omad_fl[(omad_fl != "N") & (omad_fl != "Y")] = np.nan
omad_fl[om_fl == "Y"] = "N"
omad_fl = omad_fl.fillna(omad_fl.mode()[0])
# print(omad_fl)
print(omad_fl.value_counts() / len(omad_fl))
# print(omad_fl.describe())

sns.countplot(omad_fl)
plt.title("Discharged (有相關原因未使用抗血栓藥物) - Barplot")
plt.xlabel('Discharged (有相關原因未使用抗血栓藥物)')
plt.ylabel('Number', rotation=0)
plt.show()

# Discharged (有相關原因未使用抗血栓藥物 Options)

omad_id = tsr_6.loc[:, "omad_id"]
omad_id = pd.to_numeric(omad_id, errors="coerce")
omad_id[(omad_id != 1) & (omad_id != 2) & (omad_id != 3) & (omad_id != 4) & (omad_id != 5) & (omad_id != 6)] = np.nan
omad_id = omad_id.fillna(omad_id.mode()[0])
omad_id[omad_fl == "N"] = np.nan
# print(omad_id)
print(omad_id.value_counts() / len(omad_id))
# print(omad_id.describe())

omad_id[omad_id == 1] = "UGI bleeding"
omad_id[omad_id == 2] = "Hemorrhage infarct"
omad_id[omad_id == 3] = "Others bleeding"
omad_id[omad_id == 4] = "Large infarct"
omad_id[omad_id == 5] = "Patient or family refuse"
omad_id[omad_id == 6] = "Critical AAD or Expired"

sns.countplot(omad_id, hue=omad_id)
plt.title("Discharged (有相關原因未使用抗血栓藥物 Options) - Barplot")
plt.xlabel('Discharged (有相關原因未使用抗血栓藥物 Options)')
plt.ylabel('Number', rotation=0)
plt.xticks([])
plt.legend(loc=1, title="相關原因未使用抗血栓藥物")
plt.show()

omad_id[omad_id == "UGI bleeding"] = 1
omad_id[omad_id == "Hemorrhage infarct"] = 2
omad_id[omad_id == "Others bleeding"] = 3
omad_id[omad_id == "Large infarct"] = 4
omad_id[omad_id == "Patient or family refuse"] = 5
omad_id[omad_id == "Critical AAD or Expired"] = 6
omad_id = omad_id.fillna(999)

# Before Admitted (None)

am_fl = tsr_6.loc[:, "am_fl"]
am_fl = am_fl.fillna(am_fl.mode()[0])
# print(am_fl)
print(am_fl.value_counts() / len(am_fl))
# print(am_fl.describe())

sns.countplot(am_fl)
plt.title("Before Admitted (None) - Barplot")
plt.xlabel('Before Admitted (None)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Aspirin)

amas_fl = tsr_6.loc[:, "amas_fl"]
amas_fl[(amas_fl != "Y") & (amas_fl != "N")] = np.nan
amas_fl[am_fl == "Y"] = "N"
amas_fl = amas_fl.fillna(amas_fl.mode()[0])
# print(amas_fl)
print(amas_fl.value_counts() / len(amas_fl))
# print(amas_fl.describe())

sns.countplot(amas_fl)
plt.title("Before Admitted (Aspirin) - Barplot")
plt.xlabel('Before Admitted (Aspirin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Aggrenox)

amag_fl = tsr_6.loc[:, "amag_fl"]
amag_fl[amag_fl == '0'] = "N"
amag_fl[am_fl == "Y"] = "N"
amag_fl = amag_fl.fillna(amag_fl.mode()[0])
# print(amag_fl)
print(amag_fl.value_counts() / len(amag_fl))
# print(amag_fl.describe())

sns.countplot(amag_fl)
plt.title("Before Admitted (Aggrenox) - Barplot")
plt.xlabel('Before Admitted (Aggrenox)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Ticlopidine)

amti_fl = tsr_6.loc[:, "amti_fl"]
amti_fl[am_fl == "Y"] = "N"
amti_fl[(amti_fl != "Y") & (amti_fl != "N")] = np.nan
amti_fl = amti_fl.fillna(amti_fl.mode()[0])
# print(amti_fl)
print(amti_fl.value_counts() / len(amti_fl))
# print(amti_fl.describe())

sns.countplot(amti_fl)
plt.title("Before Admitted (Ticlopidine) - Barplot")
plt.xlabel('Before Admitted (Ticlopidine)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Clopidogrel)

amcl_fl = tsr_6.loc[:, "amcl_fl"]
amcl_fl[amcl_fl == "0"] = "N"
amcl_fl[am_fl == "Y"] = "N"
amcl_fl[(amcl_fl != "Y") & (amcl_fl != "N")] = np.nan
amcl_fl = amcl_fl.fillna(amcl_fl.mode()[0])
# print(amcl_fl)
print(amcl_fl.value_counts() / len(amcl_fl))
# print(amcl_fl.describe())

sns.countplot(amcl_fl)
plt.title("Before Admitted (Clopidogrel) - Barplot")
plt.xlabel('Before Admitted (Clopidogrel)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Pletaal)

ampl_fl = tsr_6.loc[:, "ampl_fl"]
ampl_fl[(ampl_fl != "Y") & (ampl_fl != "N")] = np.nan
ampl_fl[am_fl == "Y"] = "N"
ampl_fl = ampl_fl.fillna(ampl_fl.mode()[0])
# print(ampl_fl)
print(ampl_fl.value_counts() / len(ampl_fl))
# print(ampl_fl.describe())

sns.countplot(ampl_fl)
plt.title("Before Admitted (Pletaal) - Barplot")
plt.xlabel('Before Admitted (Pletaal)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Anti H/T Drug)

amanh_fl = tsr_6.loc[:, "amanh_fl"]
amanh_fl[(amanh_fl != "Y") & (amanh_fl != "N")] = np.nan
amanh_fl[am_fl == "Y"] = "N"
amanh_fl = amanh_fl.fillna(amanh_fl.mode()[0])
# print(amanh_fl)
print(amanh_fl.value_counts() / len(amanh_fl))
# print(amanh_fl.describe())

sns.countplot(amanh_fl)
plt.title("Before Admitted (Anti H/T Drug) - Barplot")
plt.xlabel('Before Admitted (Anti H/T Drug)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Wafirin)

amwa_fl = tsr_6.loc[:, "amwa_fl"]
amwa_fl[am_fl == "Y"] = "N"
amwa_fl = amwa_fl.fillna(amwa_fl.mode()[0])
# print(amwa_fl)
print(amwa_fl.value_counts() / len(amwa_fl))
# print(amwa_fl.describe())

sns.countplot(amwa_fl)
plt.title("Before Admitted (Wafirin) - Barplot")
plt.xlabel('Before Admitted (Wafirin)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Anti DM Drug)

amand_fl = tsr_6.loc[:, "amand_fl"]
amand_fl[am_fl == "Y"] = "N"
amand_fl = amand_fl.fillna(amand_fl.mode()[0])
# print(amand_fl)
print(amand_fl.value_counts() / len(amand_fl))
# print(amand_fl.describe())

sns.countplot(amand_fl)
plt.title("Before Admitted (Anti DM Drug) - Barplot")
plt.xlabel('Before Admitted (Anti DM Drug)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Lipid Lowering Drug)

amli_fl = tsr_6.loc[:, "amli_fl"]
amli_fl[am_fl == "Y"] = "N"
amli_fl = amli_fl.fillna(amli_fl.mode()[0])
# print(amli_fl)
print(amli_fl.value_counts() / len(amli_fl))
# print(amli_fl.describe())

sns.countplot(amli_fl)
plt.title("Before Admitted (Lipid Lowering Drug) - Barplot")
plt.xlabel('Before Admitted (Lipid Lowering Drug)')
plt.ylabel('Number', rotation=0)
plt.show()

# Before Admitted (Others)

amliot_fl = tsr_6.loc[:, "amliot_fl"]
amliot_fl[am_fl == "Y"] = "N"
amliot_fl = amliot_fl.fillna(amliot_fl.mode()[0])
# print(amliot_fl)
print(amliot_fl.value_counts() / len(amliot_fl))
# print(amliot_fl.describe())

sns.countplot(amliot_fl)
plt.title("Before Admitted (Others) - Barplot")
plt.xlabel('Before Admitted (Others)')
plt.ylabel('Number', rotation=0)
plt.show()

# AMLIOT_TX

# Before Admitted (Others2)

amliot2_fl = tsr_6.loc[:, "amliot2_fl"]
amliot2_fl[amliot2_fl == str(0)] = "N"
amliot2_fl[am_fl == "Y"] = "N"
amliot2_fl = amliot2_fl.fillna(amliot2_fl.mode()[0])
# print(amliot2_fl)
print(amliot2_fl.value_counts() / len(amliot2_fl))
# print(amliot2_fl.describe())

sns.countplot(amliot2_fl)
plt.title("Before Admitted (Others2) - Barplot")
plt.xlabel('Before Admitted (Others2)')
plt.ylabel('Number', rotation=0)
plt.show()

# AMLIOT2_TX

# Complication (None)

com_id = tsr_6.loc[:, "com_id"]
com_id[com_id == '0'] = "N"
com_id = com_id.fillna(com_id.mode()[0])
# print(com_id)
print(com_id.value_counts() / len(com_id))
# print(com_id.describe())

sns.countplot(com_id)
plt.title("Complication (None) - Barplot")
plt.xlabel('Complication (None)')
plt.ylabel('Number', rotation=0)
plt.show()

# Complication (Pneumonia)

compn_fl = tsr_6.loc[:, "compn_fl"]
compn_fl[compn_fl == str(0)] = "N"
compn_fl[com_id == "Y"] = "N"
compn_fl = compn_fl.fillna(compn_fl.mode()[0])
# print(compn_fl)
print(compn_fl.value_counts() / len(compn_fl))
# print(compn_fl.describe())

sns.countplot(compn_fl)
plt.title("Complication (Pneumonia) - Barplot")
plt.xlabel('Complication (Pneumonia)')
plt.ylabel('Number', rotation=0)
plt.show()

# Complication (Urinary Tract Infection)

comut_fl = tsr_6.loc[:, "comut_fl"]
comut_fl[com_id == "Y"] = "N"
comut_fl = comut_fl.fillna(comut_fl.mode()[0])
# print(comut_fl)
print(comut_fl.value_counts() / len(comut_fl))
# print(comut_fl.describe())

sns.countplot(comut_fl)
plt.title("Complication (Urinary Tract Infection) - Barplot")
plt.xlabel('Complication (Urinary Tract Infection)')
plt.ylabel('Number', rotation=0)
plt.show()

# Complication (UGI Bleeding)

comug_fl = tsr_6.loc[:, "comug_fl"]
comug_fl[com_id == "Y"] = "N"
comug_fl = comug_fl.fillna(comug_fl.mode()[0])
# print(comug_fl)
print(comug_fl.value_counts() / len(comug_fl))
# print(comug_fl.describe())

sns.countplot(comug_fl)
plt.title("Complication (UGI Bleeding) - Barplot")
plt.xlabel('Complication (UGI Bleeding)')
plt.ylabel('Number', rotation=0)
plt.show()

# Complication (Pressure Sore)

compr_fl = tsr_6.loc[:, "compr_fl"]
compr_fl[com_id == "Y"] = "N"
compr_fl = compr_fl.fillna(compr_fl.mode()[0])
# print(compr_fl)
print(compr_fl.value_counts() / len(compr_fl))
# print(compr_fl.describe())

sns.countplot(compr_fl)
plt.title("Complication (Pressure Sore) - Barplot")
plt.xlabel('Complication (Pressure Sore)')
plt.ylabel('Number', rotation=0)
plt.show()

# Complication (Pulmonary Edema)

compu_fl = tsr_6.loc[:, "compu_fl"]
compu_fl[compu_fl == str(0)] = "N"
compu_fl[com_id == "Y"] = "N"
compu_fl = compu_fl.fillna(compu_fl.mode()[0])
# print(compu_fl)
print(compu_fl.value_counts() / len(compu_fl))
# print(compu_fl.describe())

sns.countplot(compu_fl)
plt.title("Complication (Pulmonary Edema) - Barplot")
plt.xlabel('Complication (Pulmonary Edema)')
plt.ylabel('Number', rotation=0)
plt.show()

# Complication (Acute Coronary Syndrome)

comac_fl = tsr_6.loc[:, "comac_fl"]
comac_fl[com_id == "Y"] = "N"
comac_fl = comac_fl.fillna(comac_fl.mode()[0])
# print(comac_fl)
print(comac_fl.value_counts() / len(comac_fl))
# print(comac_fl.describe())

sns.countplot(comac_fl)
plt.title("Complication (Acute Coronary Syndrome) - Barplot")
plt.xlabel('Complication (Acute Coronary Syndrome)')
plt.ylabel('Number', rotation=0)
plt.show()

# Complication (Seizure)

comse_fl = tsr_6.loc[:, "comse_fl"]
comse_fl[(comse_fl != "Y") & (comse_fl != "N")] = np.nan
comse_fl[com_id == "Y"] = "N"
comse_fl = comse_fl.fillna(comse_fl.mode()[0])
# print(comse_fl)
print(comse_fl.value_counts() / len(comse_fl))
# print(comse_fl.describe())

sns.countplot(comse_fl)
plt.title("Complication (Seizure) - Barplot")
plt.xlabel('Complication (Seizure)')
plt.ylabel('Number', rotation=0)
plt.show()

# Complication (Deep Vein Thrombosis)

comde_fl = tsr_6.loc[:, "comde_fl"]
comde_fl[(comde_fl != "N") & (comde_fl != "Y")] = np.nan
comde_fl[com_id == "Y"] = "N"
comde_fl = comde_fl.fillna(comde_fl.mode()[0])
# print(comde_fl)
print(comde_fl.value_counts() / len(comde_fl))
# print(comde_fl.describe())

sns.countplot(comde_fl)
plt.title("Complication (Deep Vein Thrombosis) - Barplot")
plt.xlabel('Complication (Deep Vein Thrombosis)')
plt.ylabel('Number', rotation=0)
plt.xticks(rotation=90)
plt.show()

# Complication (Others)

como_fl = tsr_6.loc[:, "como_fl"]
como_fl[(como_fl != "N") & (como_fl != "Y")] = np.nan
como_fl[com_id == "Y"] = "N"
como_fl = como_fl.fillna(como_fl.mode()[0])
# print(como_fl)
print(como_fl.value_counts() / len(como_fl))
# print(como_fl.describe())

sns.countplot(como_fl)
plt.title("Complication (Others) - Barplot")
plt.xlabel('Complication (Others)')
plt.ylabel('Number', rotation=0)
plt.show()

# COMO_TX

# Deterioration (None)

det_id = tsr_6.loc[:, "det_id"]
det_id[det_id == '0'] = "N"
det_id[(det_id != "N") & (det_id != "Y")] = np.nan
det_id = det_id.fillna(det_id.mode()[0])
# print(det_id)
print(det_id.value_counts() / len(det_id))
# print(det_id.describe())

sns.countplot(det_id)
plt.title("Deterioration (None) - Barplot")
plt.xlabel('Deterioration (None)')
plt.ylabel('Number', rotation=0)
plt.show()

# Deterioration (Stroke-in-evolution - NIHSS≧2)

detst_fl = tsr_6.loc[:, "detst_fl"]
detst_fl[det_id == "Y"] = "N"
detst_fl = detst_fl.fillna(detst_fl.mode()[0])
# print(detst_fl)
print(detst_fl.value_counts() / len(detst_fl))
# print(detst_fl.describe())

sns.countplot(detst_fl)
plt.title("Deterioration (Stroke-in-evolution - NIHSS≧2) - Barplot")
plt.xlabel('Deterioration (Stroke-in-evolution - NIHSS≧2)')
plt.ylabel('Number', rotation=0)
plt.show()

# Deterioration (Herniation)

dethe_fl = tsr_6.loc[:, "dethe_fl"]
dethe_fl[dethe_fl == str(0)] = "N"
dethe_fl[det_id == "Y"] = "N"
dethe_fl = dethe_fl.fillna(dethe_fl.mode()[0])
# print(dethe_fl)
print(dethe_fl.value_counts() / len(dethe_fl))
# print(dethe_fl.describe())

sns.countplot(dethe_fl)
plt.title("Deterioration (Herniation) - Barplot")
plt.xlabel('Deterioration (Herniation)')
plt.ylabel('Number', rotation=0)
plt.show()

# Deterioration (Hemorrhagic Infarct)

detho_fl = tsr_6.loc[:, "detho_fl"]
detho_fl[det_id == "Y"] = "N"
detho_fl = detho_fl.fillna(detho_fl.mode()[0])
# print(detho_fl)
print(detho_fl.value_counts() / len(detho_fl))
# print(detho_fl.describe())

sns.countplot(detho_fl)
plt.title("Deterioration (Hemorrhagic Infarct) - Barplot")
plt.xlabel('Deterioration (Hemorrhagic Infarct)')
plt.ylabel('Number', rotation=0)
plt.show()

# Deterioration (Hemorrhagic Infarct 36hrs)

dethoh_fl = tsr_6.loc[:, "dethoh_fl"]
dethoh_fl[(dethoh_fl != "N") & (dethoh_fl != "Y")] = np.nan
dethoh_fl[det_id == "Y"] = "N"
dethoh_fl = dethoh_fl.fillna(dethoh_fl.mode()[0])
# print(dethoh_fl)
print(dethoh_fl.value_counts() / len(dethoh_fl))
# print(dethoh_fl.describe())

sns.countplot(dethoh_fl)
plt.title("Deterioration (Hemorrhagic Infarct 36hrs) - Barplot")
plt.xlabel('Deterioration (Hemorrhagic Infarct 36hrs)')
plt.ylabel('Number', rotation=0)
plt.show()

# Deterioration (Hematoma Enlargement - ICH)

detha_fl = tsr_6.loc[:, "detha_fl"]
detha_fl[(detha_fl != "N") & (detha_fl != "Y")] = np.nan
detha_fl[det_id == "Y"] = "N"
detha_fl = detha_fl.fillna(detha_fl.mode()[0])
# print(detha_fl)
print(detha_fl.value_counts() / len(detha_fl))
# print(detha_fl.describe())

sns.countplot(detha_fl)
plt.title("Deterioration (Hematoma Enlargement - ICH) - Barplot")
plt.xlabel('Deterioration (Hematoma Enlargement - ICH)')
plt.ylabel('Number', rotation=0)
plt.show()

# Deterioration (Vasospasm - SAH)

detva_fl = tsr_6.loc[:, "detva_fl"]
detva_fl[det_id == "Y"] = "N"
detva_fl = detva_fl.fillna(detva_fl.mode()[0])
# print(detva_fl)
print(detva_fl.value_counts() / len(detva_fl))
# print(detva_fl.describe())

sns.countplot(detva_fl)
plt.title("Deterioration (Vasospasm - SAH) - Barplot")
plt.xlabel('Deterioration (Vasospasm - SAH)')
plt.ylabel('Number', rotation=0)
plt.show()

# Deterioration (Re-bleeding - SAH)

detre_fl = tsr_6.loc[:, "detre_fl"]
detre_fl[(detre_fl != "N") & (detre_fl != "Y")] = np.nan
detre_fl[det_id == "Y"] = "N"
detre_fl = detre_fl.fillna(detre_fl.mode()[0])
# print(detre_fl)
print(detre_fl.value_counts() / len(detre_fl))
# print(detre_fl.describe())

sns.countplot(detre_fl)
plt.title("Deterioration (Re-bleeding - SAH) - Barplot")
plt.xlabel('Deterioration (Re-bleeding - SAH)')
plt.ylabel('Number', rotation=0)
plt.show()

# Deterioration (Medical Problems)

detme_fl = tsr_6.loc[:, "detme_fl"]
detme_fl[(detme_fl != "N") & (detme_fl != "Y")] = np.nan
detme_fl[det_id == "Y"] = "N"
detme_fl = detme_fl.fillna(detme_fl.mode()[0])
# print(detme_fl)
print(detme_fl.value_counts() / len(detme_fl))
# print(detme_fl.describe())

sns.countplot(detme_fl)
plt.title("Deterioration (Medical Problems) - Barplot")
plt.xlabel('Deterioration (Medical Problems)')
plt.ylabel('Number', rotation=0)
plt.show()

# Deterioration (Others)

deto_fl = tsr_6.loc[:, "deto_fl"]
deto_fl[deto_fl == str(0)] = "N"
deto_fl[deto_fl == str(1)] = "Y"
deto_fl[(deto_fl != "N") & (deto_fl != "Y")] = np.nan
deto_fl[det_id == "Y"] = "N"
deto_fl = deto_fl.fillna(deto_fl.mode()[0])
# print(deto_fl)
print(deto_fl.value_counts() / len(deto_fl))
# print(deto_fl.describe())

sns.countplot(deto_fl)
plt.title("Deterioration (Others) - Barplot")
plt.xlabel('Deterioration (Others)')
plt.ylabel('Number', rotation=0)
plt.show()

# DETO_TX

# CT日期

ct_time = tsr_6.loc[:, ["ct_dt", "cth_nm", "ctm_nm"]]
ct_time.ct_dt = pd.to_datetime(ct_time.ct_dt, errors="coerce", format="%Y-%m-%d")
ct_time.ct_dt[(ct_time.ct_dt.dt.year < 2006) | (ct_time.ct_dt.dt.year > 2021)] = np.nan

ct_time['cth_nm'] = pd.to_numeric(ct_time['cth_nm'], errors='coerce')
ct_time['cth_nm'][(ct_time['cth_nm'] < 0) | (ct_time['cth_nm'] > 24)] = np.nan
ct_time['cth_nm'][ct_time['cth_nm'] == 24] = 0
ct_time['ctm_nm'] = pd.to_numeric(ct_time['ctm_nm'], errors='coerce')
ct_time['ctm_nm'][(ct_time['ctm_nm'] < 0) | (ct_time['ctm_nm'] > 60)] = np.nan
ct_time['ctm_nm'][ct_time['ctm_nm'] == 60] = 0

ct_time['ct_dt'] = ct_time['ct_dt'].fillna(ct_time['ct_dt'].mode()[0])
ct_time['cth_nm'] = ct_time['cth_nm'].fillna(ct_time['cth_nm'].mean())
ct_time['ctm_nm'] = ct_time['ctm_nm'].fillna(ct_time['ctm_nm'].mean())

ct = ct_time['ct_dt'].astype(str) + ' ' + ct_time['cth_nm'].astype(int).map(str) + ':' + ct_time['ctm_nm'].astype(
    int).map(str)

ct_dt = pd.to_datetime(ct, format='%Y/%m/%d %H:%M', errors='coerce')
# print(ct_dt.value_counts() / len(ct_dt))
print(ct_dt.describe())

ct_dt.value_counts().plot()
plt.title("CT Date - Lineplot")
plt.xlabel('CT Date')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# CT時間 - 時

cth_nm = tsr_6.loc[:, "cth_nm"]
cth_nm = pd.to_numeric(cth_nm, errors="coerce")
cth_nm[(cth_nm < 0) | (cth_nm > 24)] = np.nan
cth_nm[cth_nm == 24] = 0
cth_nm = cth_nm.fillna(cth_nm.mean())
# print(cth_nm)
# print(cth_nm.value_counts() / len(cth_nm))
print(cth_nm.describe())

# CT時間 - 分

ctm_nm = tsr_6.loc[:, "ctm_nm"]
ctm_nm = pd.to_numeric(ctm_nm, errors="coerce")
ctm_nm[(ctm_nm < 0) | (ctm_nm > 60)] = np.nan
ctm_nm[ctm_nm == 60] = 0
ctm_nm = ctm_nm.fillna(ctm_nm.mean())
# print(ctm_nm)
# print(ctm_nm.value_counts() / len(ctm_nm))
print(ctm_nm.describe())

# CT no Findings

ct_fl = tsr_6.loc[:, "ct_fl"]
ct_fl[(ct_fl != "N") & (ct_fl != "Y")] = np.nan
ct_fl = ct_fl.fillna(ct_fl.mode()[0])
# print(ct_fl)
print(ct_fl.value_counts() / len(ct_fl))
# print(ct_fl.describe())

sns.countplot(ct_fl)
plt.title("CT no Findings - Barplot")
plt.xlabel('CT no Findings')
plt.ylabel('Number', rotation=0)
plt.show()

# CTO_TX

# MRI日期

mri_time = tsr_6.loc[:, ["mri_dt", "mrih_nm", "mrim_nm"]]
mri_time.mri_dt = pd.to_datetime(mri_time.mri_dt, errors="coerce", format="%Y-%m-%d")
mri_time.mri_dt[(mri_time.mri_dt.dt.year < 2006) | (mri_time.mri_dt.dt.year > 2021)] = np.nan

mri_time['mrih_nm'] = pd.to_numeric(mri_time['mrih_nm'], errors='coerce')
mri_time['mrih_nm'][(mri_time['mrih_nm'] < 0) | (mri_time['mrih_nm'] > 24)] = np.nan
mri_time['mrih_nm'][mri_time['mrih_nm'] == 24] = 0
mri_time['mrim_nm'] = pd.to_numeric(mri_time['mrim_nm'], errors='coerce')
mri_time['mrim_nm'][(mri_time['mrim_nm'] < 0) | (mri_time['mrim_nm'] > 60)] = np.nan
mri_time['mrim_nm'][mri_time['mrim_nm'] == 60] = 0

mri_time['mri_dt'] = mri_time['mri_dt'].fillna(mri_time['mri_dt'].mode()[0])
mri_time['mrih_nm'] = mri_time['mrih_nm'].fillna(mri_time['mrih_nm'].mean())
mri_time['mrim_nm'] = mri_time['mrim_nm'].fillna(mri_time['mrim_nm'].mean())

mri = mri_time['mri_dt'].astype(str) + ' ' + mri_time['mrih_nm'].astype(int).map(str) + ':' + mri_time[
    'mrim_nm'].astype(int).map(str)

mri_dt = pd.to_datetime(mri, format='%Y/%m/%d %H:%M', errors='coerce')
# print(mri_dt.value_counts() / len(mri_dt))
print(mri_dt.describe())

mri_dt.value_counts().plot()
plt.title("MRI Date - Lineplot")
plt.xlabel('MRI Date')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# MRI時間 - 時

mrih_nm = tsr_6.loc[:, "mrih_nm"]
mrih_nm = pd.to_numeric(mrih_nm, errors="coerce")
mrih_nm[(mrih_nm < 0) | (mrih_nm > 24)] = np.nan
mrih_nm[mrih_nm == 24] = 0
mrih_nm = mrih_nm.fillna(mrih_nm.mean())
# print(mrih_nm)
# print(mrih_nm.value_counts() / len(mrih_nm))
print(mrih_nm.describe())

# MRI時間 - 分

mrim_nm = tsr_6.loc[:, "mrim_nm"]
mrim_nm = pd.to_numeric(mrim_nm, errors="coerce")
mrim_nm[(mrim_nm < 0) | (mrim_nm > 60)] = np.nan
mrim_nm[mrim_nm == 60] = 0
mrim_nm = mrim_nm.fillna(mrim_nm.mean())
# print(mrim_nm)
# print(mrim_nm.value_counts() / len(mrim_nm))
print(mrim_nm.describe())

# MRI no Findings

mri_fl = tsr_6.loc[:, "mri_fl"]
mri_fl[(mri_fl != "N") & (mri_fl != "Y")] = np.nan
mri_fl = mri_fl.fillna(mri_fl.mode()[0])
# print(mri_fl)
print(mri_fl.value_counts() / len(mri_fl))
# print(mri_fl.describe())

sns.countplot(mri_fl)
plt.title("CT no Findings - Barplot")
plt.xlabel('CT no Findings')
plt.ylabel('Number', rotation=0)
plt.show()

# MRIO_TX

# Ultrasound/MRA Studies

# Carotid Duplex

cd_id = tsr_6.loc[:, "cd_id"]
cd_id = pd.to_numeric(cd_id, errors="coerce")
cd_id[(cd_id != 0) & (cd_id != 1) & (cd_id != 2)] = np.nan
cd_id = cd_id.fillna(cd_id.mode()[0])
# print(cd_id)
print(cd_id.value_counts() / len(cd_id))
# print(cd_id.describe())

cd_id_labels = ["Undone", "Present", "Absent"]
sns.countplot(cd_id).set_xticklabels(cd_id_labels)
plt.title("Carotid Duplex - Barplot")
plt.xlabel('Carotid Duplex')
plt.ylabel('Number', rotation=0)
plt.show()

# Carotid Atherosclerosis (R ICA)

cdr_id = tsr_6.loc[:, "cdr_id"]
cdr_id = pd.to_numeric(cdr_id, errors="coerce")
cdr_id[(cdr_id != 1) & (cdr_id != 2) & (cdr_id != 3) & (cdr_id != 4)] = np.nan
cdr_id = cdr_id.fillna(cdr_id.mode()[0])
# print(cdr_id)
print(cdr_id.value_counts() / len(cdr_id))
# print(cdr_id.describe())

cdr_id_labels = ["0-49%", "50-69%", "70-99%", "100%"]
sns.countplot(cdr_id).set_xticklabels(cdr_id_labels)
plt.title("Carotid Atherosclerosis (R ICA) - Barplot")
plt.xlabel('Carotid Atherosclerosis (R ICA)')
plt.ylabel('Number', rotation=0)
plt.show()

# Carotid Atherosclerosis (L ICA)

cdl_id = tsr_6.loc[:, "cdl_id"]
cdl_id = pd.to_numeric(cdl_id, errors="coerce")
cdl_id[(cdl_id != 1) & (cdl_id != 2) & (cdl_id != 3) & (cdl_id != 4)] = np.nan
cdl_id = cdl_id.fillna(cdl_id.mode()[0])
# print(cdl_id)
print(cdl_id.value_counts() / len(cdl_id))
# print(cdl_id.describe())

cdl_id_labels = ["0-49%", "50-69%", "70-99%", "100%"]
sns.countplot(cdl_id).set_xticklabels(cdl_id_labels)
plt.title("Carotid Atherosclerosis (L ICA) - Barplot")
plt.xlabel('Carotid Atherosclerosis (L ICA)')
plt.ylabel('Number', rotation=0)
plt.show()

# TCCS

tccs_id = tsr_6.loc[:, "tccs_id"]
tccs_id = pd.to_numeric(tccs_id, errors="coerce")
tccs_id[(tccs_id != 1) & (tccs_id != 0)] = np.nan
tccs_id = tccs_id.fillna(tccs_id.mode()[0])
# print(tccs_id)
print(tccs_id.value_counts() / len(tccs_id))
# print(tccs_id.describe())

tccs_id_labels = ["Undone", "Poor window"]
sns.countplot(tccs_id).set_xticklabels(tccs_id_labels)
plt.title("TCCS - Barplot")
plt.xlabel('TCCS')
plt.ylabel('Number', rotation=0)
plt.show()

# TCCS (R MCA)

tccsr_id = tsr_6.loc[:, "tccsr_id"]
tccsr_id = pd.to_numeric(tccsr_id, errors="coerce")
tccsr_id[(tccsr_id != 1) & (tccsr_id != 2) & (tccsr_id != 3)] = np.nan
tccsr_id = tccsr_id.fillna(tccsr_id.mode()[0])
# print(tccsr_id)
print(tccsr_id.value_counts() / len(tccsr_id))
# print(tccsr_id.describe())

tccsr_id_labels = ["0-49%", "50-99%", "100%"]
sns.countplot(tccsr_id).set_xticklabels(tccsr_id_labels)
plt.title("TCCS (R MCA) - Barplot")
plt.xlabel('TCCS (R MCA)')
plt.ylabel('Number', rotation=0)
plt.show()

# TCCS (L MCA)

tccsl_id = tsr_6.loc[:, "tccsl_id"]
tccsl_id = pd.to_numeric(tccsl_id, errors="coerce")
tccsl_id[(tccsl_id != 1) & (tccsl_id != 2) & (tccsl_id != 3)] = np.nan
# tccsl_id = tccsl_id.fillna(tccsl_id.mode()[0])
# print(tccsl_id)
print(tccsl_id.value_counts() / len(tccsl_id))
# print(tccsl_id.describe())

# tccsl_id_labels = ["0-49%","50-99%","100%"]
tccsl_id_labels = ["0-49%", "50-99%"]
sns.countplot(tccsl_id).set_xticklabels(tccsl_id_labels)
plt.title("TCCS (L MCA) - Barplot")
plt.xlabel('TCCS (L MCA)')
plt.ylabel('Number', rotation=0)
plt.show()
tccsl_id = tccsl_id.fillna(999)

# TCCS (BA)

tccsba_id = tsr_6.loc[:, "tccsba_id"]
tccsba_id = pd.to_numeric(tccsba_id, errors="coerce")
tccsba_id[(tccsba_id != 1) & (tccsba_id != 2) & (tccsba_id != 3)] = np.nan
tccsba_id = tccsba_id.fillna(tccsba_id.mode()[0])
# print(tccsba_id)
print(tccsba_id.value_counts() / len(tccsba_id))
# print(tccsba_id.describe())

tccsba_id_labels = ["0-49%", "50-99%", "100%"]
sns.countplot(tccsba_id).set_xticklabels(tccsba_id_labels)
plt.title("TCCS (BA) - Barplot")
plt.xlabel('TCCS (BA)')
plt.ylabel('Number', rotation=0)
plt.show()

# MRA

mra_fl = tsr_6.loc[:, "mra_fl"]
mra_fl[mra_fl == str(0)] = "N"
mra_fl[mra_fl == str(1)] = "Y"
mra_fl[(mra_fl != "N") & (mra_fl != "Y")] = np.nan
mra_fl = mra_fl.fillna(mra_fl.mode()[0])
# print(mra_fl)
print(mra_fl.value_counts() / len(mra_fl))
# print(mra_fl.describe())

sns.countplot(mra_fl)
plt.title("MRA - Barplot")
plt.xlabel('MRA')
plt.ylabel('Number', rotation=0)
plt.show()

# CTA

cta_fl = tsr_6.loc[:, "cta_fl"]
cta_fl[cta_fl == str(0)] = "N"
cta_fl[cta_fl == str(1)] = "Y"
cta_fl[(cta_fl != "N") & (cta_fl != "Y")] = np.nan
cta_fl = cta_fl.fillna(cta_fl.mode()[0])
# print(cta_fl)
print(cta_fl.value_counts() / len(cta_fl))
# print(cta_fl.describe())

sns.countplot(cta_fl)
plt.title("CTA - Barplot")
plt.xlabel('CTA')
plt.ylabel('Number', rotation=0)
plt.show()

# DSA

dsa_fl = tsr_6.loc[:, "dsa_fl"]
dsa_fl[dsa_fl == str(0)] = "N"
dsa_fl[dsa_fl == str(1)] = "Y"
dsa_fl[(dsa_fl != "N") & (dsa_fl != "Y")] = np.nan
dsa_fl = dsa_fl.fillna(dsa_fl.mode()[0])
# print(dsa_fl)
print(dsa_fl.value_counts() / len(dsa_fl))
# print(dsa_fl.describe())

sns.countplot(dsa_fl)
plt.title("DSA - Barplot")
plt.xlabel('DSA')
plt.ylabel('Number', rotation=0)
plt.show()

# Undone MRI, CTA and DSA

mcd_id = tsr_6.loc[:, "mcd_id"]
mcd_id[mcd_id == str(0)] = 0
mcd_id[mcd_id == str(1)] = 1
mcd_id[mcd_id == 0] = "N"
mcd_id[mcd_id == 1] = "Y"
mcd_id[(mcd_id != "N") & (mcd_id != "Y")] = np.nan
mcd_id = mcd_id.fillna(mcd_id.mode()[0])
# print(mcd_id)
print(mcd_id.value_counts() / len(mcd_id))
# print(mcd_id.describe())

sns.countplot(mcd_id)
plt.title("Undone MRI, CTA and DSA - Barplot")
plt.xlabel('Undone MRI, CTA and DSA')
plt.ylabel('Number', rotation=0)
plt.show()

# MRI, CTA and DSA (R MCA)

mcdr_id = tsr_6.loc[:, "mcdr_id"]
mcdr_id = pd.to_numeric(mcdr_id, errors="coerce")
mcdr_id[(mcdr_id != 1) & (mcdr_id != 2) & (mcdr_id != 3)] = np.nan
mcdr_id = mcdr_id.fillna(mcdr_id.mode()[0])
# print(mcdr_id)
print(mcdr_id.value_counts() / len(mcdr_id))
# print(mcdr_id.describe())

mcdr_id_labels = ["0-49%", "50-99%", "100%"]
sns.countplot(mcdr_id).set_xticklabels(mcdr_id_labels)
plt.title("MRI, CTA and DSA (R MCA) - Barplot")
plt.xlabel('MRI, CTA and DSA (R MCA)')
plt.ylabel('Number', rotation=0)
plt.show()

# MRI, CTA and DSA (L MCA)

mcdl_id = tsr_6.loc[:, "mcdl_id"]
mcdl_id = pd.to_numeric(mcdl_id, errors="coerce")
mcdl_id[(mcdl_id != 1) & (mcdl_id != 2) & (mcdl_id != 3)] = np.nan
mcdl_id = mcdl_id.fillna(mcdl_id.mode()[0])
# print(mcdl_id)
print(mcdl_id.value_counts() / len(mcdl_id))
# print(mcdl_id.describe())

mcdl_id_labels = ["0-49%", "50-99%", "100%"]
sns.countplot(mcdl_id).set_xticklabels(mcdl_id_labels)
plt.title("MRI, CTA and DSA (L MCA) - Barplot")
plt.xlabel('MRI, CTA and DSA (L MCA)')
plt.ylabel('Number', rotation=0)
plt.show()

# MRI, CTA and DSA (VA or BA)

mcdba_id = tsr_6.loc[:, "mcdba_id"]
mcdba_id = pd.to_numeric(mcdba_id, errors="coerce")
mcdba_id[(mcdba_id != 1) & (mcdba_id != 2) & (mcdba_id != 3)] = np.nan
mcdba_id = mcdba_id.fillna(mcdba_id.mode()[0])
# print(mcdba_id)
print(mcdba_id.value_counts() / len(mcdba_id))
# print(mcdba_id.describe())

mcdba_id_labels = ["0-49%", "50-99%", "100%"]
sns.countplot(mcdba_id).set_xticklabels(mcdba_id_labels)
plt.title("MRI, CTA and DSA (VA or BA) - Barplot")
plt.xlabel('MRI, CTA and DSA (VA or BA)')
plt.ylabel('Number', rotation=0)
plt.show()

# MRI, CTA and DSA (R ICA)

mcdri_id = tsr_6.loc[:, "mcdri_id"]
mcdri_id = pd.to_numeric(mcdri_id, errors="coerce")
mcdri_id[(mcdri_id != 1) & (mcdri_id != 2) & (mcdri_id != 3)] = np.nan
mcdri_id = mcdri_id.fillna(mcdri_id.mode()[0])
# print(mcdri_id)
print(mcdri_id.value_counts() / len(mcdri_id))
# print(mcdri_id.describe())

mcdri_id_labels = ["0-49%", "50-99%", "100%"]
sns.countplot(mcdri_id).set_xticklabels(mcdri_id_labels)
plt.title("MRI, CTA and DSA (R ICA) - Barplot")
plt.xlabel('MRI, CTA and DSA (R ICA)')
plt.ylabel('Number', rotation=0)
plt.show()

# MRI, CTA and DSA (L ICA)

mcdli_id = tsr_6.loc[:, "mcdli_id"]
mcdli_id = pd.to_numeric(mcdli_id, errors="coerce")
mcdli_id[(mcdli_id != 1) & (mcdli_id != 2) & (mcdli_id != 3)] = np.nan
mcdli_id = mcdli_id.fillna(mcdli_id.mode()[0])
# print(mcdli_id)
print(mcdli_id.value_counts() / len(mcdli_id))
# print(mcdli_id.describe())

mcdli_id_labels = ["0-49%", "50-99%", "100%"]
sns.countplot(mcdli_id).set_xticklabels(mcdli_id_labels)
plt.title("MRI, CTA and DSA (L ICA) - Barplot")
plt.xlabel('MRI, CTA and DSA (L ICA)')
plt.ylabel('Number', rotation=0)
plt.show()

# NIHSS 最初進院日期

nihsin_time = tsr_6.loc[:, ["nihsin_dt", "nihsinh_nm", "nihsinm_nm"]]
nihsin_time.nihsin_dt = pd.to_datetime(nihsin_time.nihsin_dt, errors="coerce", format="%Y-%m-%d")
nihsin_time.nihsin_dt[(nihsin_time.nihsin_dt.dt.year < 2006) | (nihsin_time.nihsin_dt.dt.year > 2021)] = np.nan

nihsin_time['nihsinh_nm'] = pd.to_numeric(nihsin_time['nihsinh_nm'], errors='coerce')
nihsin_time['nihsinh_nm'][(nihsin_time['nihsinh_nm'] < 0) | (nihsin_time['nihsinh_nm'] > 24)] = np.nan
nihsin_time['nihsinh_nm'][nihsin_time['nihsinh_nm'] == 24] = 0
nihsin_time['nihsinm_nm'] = pd.to_numeric(nihsin_time['nihsinm_nm'], errors='coerce')
nihsin_time['nihsinm_nm'][(nihsin_time['nihsinm_nm'] < 0) | (nihsin_time['nihsinm_nm'] > 24)] = np.nan
nihsin_time['nihsinm_nm'][nihsin_time['nihsinm_nm'] == 24] = 0

nihsin_time['nihsin_dt'] = nihsin_time['nihsin_dt'].fillna(nihsin_time['nihsin_dt'].mode()[0])
nihsin_time['nihsinh_nm'] = nihsin_time['nihsinh_nm'].fillna(nihsin_time['nihsinh_nm'].mean())
nihsin_time['nihsinm_nm'] = nihsin_time['nihsinm_nm'].fillna(nihsin_time['nihsinm_nm'].mean())

nihsin = nihsin_time['nihsin_dt'].astype(str) + ' ' + nihsin_time['nihsinh_nm'].astype(int).map(str) + ':' + \
         nihsin_time['nihsinm_nm'].astype(int).map(str)

nihsin_dt = pd.to_datetime(nihsin, format='%Y/%m/%d %H:%M', errors='coerce')
# print(nihsin_dt.value_counts() / len(nihsin_dt))
print(nihsin_dt.describe())

nihsin_dt.value_counts().plot()
plt.title("NIHSS 最初進院日期 - Lineplot")
plt.xlabel('NIHSS 最初進院日期')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# NIHSS 最初進院時間 - 時

nihsinh_nm = tsr_6.loc[:, "nihsinh_nm"]
nihsinh_nm = pd.to_numeric(nihsinh_nm, errors='coerce')
nihsinh_nm[(nihsinh_nm < 0) | (nihsinh_nm > 24)] = np.nan
nihsinh_nm[nihsinh_nm == 24] = 0
nihsinh_nm = nihsinh_nm.fillna(nihsinh_nm.mean())
# print(nihsinh_nm)
# print(nihsinh_nm.value_counts() / len(nihsinh_nm))
print(nihsinh_nm.describe())

# NIHSS 最初進院時間 - 分

nihsinm_nm = tsr_6.loc[:, "nihsinm_nm"]
nihsinm_nm = pd.to_numeric(nihsinm_nm, errors='coerce')
nihsinm_nm[(nihsinm_nm < 0) | (nihsinm_nm > 60)] = np.nan
nihsinm_nm[nihsinm_nm == 60] = 0
nihsinm_nm = nihsinm_nm.fillna(nihsinm_nm.mean())
# print(nihsinm_nm)
# print(nihsinm_nm.value_counts() / len(nihsinm_nm))
print(nihsinm_nm.describe())

# NIHSS 離院日期

nihsot_dt = tsr_6.loc[:, "nihsot_dt"]
nihsot_dt = pd.to_datetime(nihsot_dt, errors='coerce')
nihsot_dt[(nihsot_dt.dt.year < 2005) | (nihsot_dt.dt.year > 2021) | (nihsot_dt < nihsin_time.nihsin_dt)] = np.nan
# print(nihsot_dt)
# print(nihsot_dt.value_counts() / len(nihsot_dt))
# print(nihsot_dt.describe())

nihss_hospitalised_time = nihsot_dt - nihsin_dt
nihss_hospitalised_time = nihss_hospitalised_time.dt.days

q1 = nihss_hospitalised_time.quantile(0.25)
q3 = nihss_hospitalised_time.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
nihss_hospitalised_time[
    (nihss_hospitalised_time < inner_fence_low) | (nihss_hospitalised_time > inner_fence_upp)] = np.nan
nihss_hospitalised_time[nihss_hospitalised_time < 0] = np.nan

nihss_hospitalised_time = nihss_hospitalised_time.fillna(nihss_hospitalised_time.mean())

# print(nihss_hospitalised_time.value_counts().sort_values(ascending= True))
print(nihss_hospitalised_time.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihss_hospitalised_time.plot.box(ax=ax1)
ax1.set_title("NIHSS 住院天數 - Boxplot")
ax1.set_xlabel('NIHSS 住院天數')
ax1.set_ylabel('Days', rotation=0)
ax1.set_xticks([])

# nihss_hospitalised_time.plot.hist(ax = ax2, bins=100)
# plt.show()
nihss_hospitalised_time.plot.hist(ax=ax2, bins=100)
ax2.set_title("NIHSS 住院天數 - Histogram")
ax2.set_xlabel('NIHSS 住院天數')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHSS 離院時間 - 時

nihsoth_nm = tsr_6.loc[:, "nihsoth_nm"]
nihsoth_nm = pd.to_numeric(nihsoth_nm, errors='coerce')
nihsoth_nm[(nihsoth_nm < 0) | (nihsoth_nm > 24)] = np.nan
nihsoth_nm[nihsoth_nm == 24] = 0
nihsoth_nm = nihsoth_nm.fillna(nihsoth_nm.mean())
# print(nihsoth_nm)
# print(nihsoth_nm.value_counts() / len(nihsoth_nm))
print(nihsoth_nm.describe())

# NIHSS 離院時間 - 分

nihsotm_nm = tsr_6.loc[:, "nihsotm_nm"]
nihsotm_nm = pd.to_numeric(nihsotm_nm, errors='coerce')
nihsotm_nm[(nihsotm_nm < 0) | (nihsotm_nm > 60)] = np.nan
nihsotm_nm[nihsotm_nm == 60] = 0
nihsotm_nm = nihsotm_nm.fillna(nihsotm_nm.mean())
# print(nihsotm_nm)
# print(nihsotm_nm.value_counts() / len(nihsotm_nm))
print(nihsotm_nm.describe())

# ECG

ecg_id = tsr_6.loc[:, "ecg_id"]
ecg_id[ecg_id == str(1)] = 1
ecg_id[ecg_id == str(0)] = 0
ecg_id[ecg_id == 1] = "Y"
ecg_id[ecg_id == 0] = "N"
ecg_id[(ecg_id != "N") & (ecg_id != "Y")] = np.nan
ecg_id = ecg_id.fillna(ecg_id.mode()[0])
# print(ecg_id)
print(ecg_id.value_counts() / len(ecg_id))
# print(ecg_id.describe())

ecg_id_labels = ["Normal", "Undone"]
sns.countplot(ecg_id).set_xticklabels(ecg_id_labels)
plt.title("ECG - Barplot")
plt.xlabel('ECG')
plt.ylabel('Number', rotation=0)

plt.show()

# ECG (LVH)

ecgl_fl = tsr_6.loc[:, "ecgl_fl"]
ecgl_fl[ecgl_fl == str(1)] = 1
ecgl_fl[ecgl_fl == str(0)] = 0
ecgl_fl[ecgl_fl == 1] = "Y"
ecgl_fl[ecgl_fl == 0] = "N"
ecgl_fl[(ecgl_fl != "N") & (ecgl_fl != "Y")] = np.nan
ecgl_fl[ecg_id == "N"] = "N"
ecgl_fl = ecgl_fl.fillna(ecgl_fl.mode()[0])
# print(ecgl_fl)
print(ecgl_fl.value_counts() / len(ecgl_fl))
# print(ecgl_fl.describe())

sns.countplot(ecgl_fl)
plt.title("ECG (LVH) - Barplot")
plt.xlabel('ECG (LVH)')
plt.ylabel('Number', rotation=0)
plt.show()

# ECG (AF)

ecga_fl = tsr_6.loc[:, "ecga_fl"]
ecga_fl[ecga_fl == str(1)] = 1
ecga_fl[ecga_fl == str(0)] = 0
ecga_fl[ecga_fl == 1] = "Y"
ecga_fl[ecga_fl == 0] = "N"
ecga_fl[(ecga_fl != "N") & (ecga_fl != "Y")] = np.nan
ecga_fl[ecg_id == "N"] = "N"
ecga_fl = ecga_fl.fillna(ecga_fl.mode()[0])
# print(ecga_fl)
print(ecga_fl.value_counts() / len(ecga_fl))
# print(ecga_fl.describe())

sns.countplot(ecga_fl)
plt.title("ECG (AF) - Barplot")
plt.xlabel('ECG (AF)')
plt.ylabel('Number', rotation=0)
plt.show()

# ECG (Q wave)

ecgq_fl = tsr_6.loc[:, "ecgq_fl"]
ecgq_fl[ecgq_fl == str(1)] = 1
ecgq_fl[ecgq_fl == str(0)] = 0
ecgq_fl[ecgq_fl == 1] = "Y"
ecgq_fl[ecgq_fl == 0] = "N"
ecgq_fl[(ecgq_fl != "N") & (ecgq_fl != "Y")] = np.nan
ecgq_fl[ecg_id == "N"] = "N"
ecgq_fl = ecgq_fl.fillna(ecgq_fl.mode()[0])
# print(ecgq_fl)
print(ecgq_fl.value_counts() / len(ecgq_fl))
# print(ecgq_fl.describe())

sns.countplot(ecgq_fl)
plt.title("ECG (Q wave) - Barplot")
plt.xlabel('ECG (Q wave)')
plt.ylabel('Number', rotation=0)
plt.show()

# ECG (Others)

ecgo_fl = tsr_6.loc[:, "ecgo_fl"]
ecgo_fl[ecgo_fl == str(1)] = 1
ecgo_fl[ecgo_fl == str(0)] = 0
ecgo_fl[ecgo_fl == 1] = "Y"
ecgo_fl[ecgo_fl == 0] = "N"
ecgo_fl[(ecgo_fl != "N") & (ecgo_fl != "Y")] = np.nan
ecgo_fl[ecg_id == "N"] = "N"
ecgo_fl = ecgo_fl.fillna(ecgo_fl.mode()[0])
# print(ecgo_fl)
print(ecgo_fl.value_counts() / len(ecgo_fl))
# print(ecgo_fl.describe())

sns.countplot(ecgo_fl)
plt.title("ECG (Others) - Barplot")
plt.xlabel('ECG (Others)')
plt.ylabel('Number', rotation=0)
plt.show()

# ECGO_TX

# 抽血結果 - 急診抽血或第一次抽血

# HB

hb_nm = tsr_6.loc[:, "hb_nm"]

q1 = hb_nm.quantile(0.25)
q3 = hb_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
hb_nm[(hb_nm < inner_fence_low) | (hb_nm > inner_fence_upp)] = np.nan

hb_nm = hb_nm.fillna(round(hb_nm.mean(), 3))

# print(hb_nm)
# print(hb_nm.value_counts() / len(hb_nm))
print(hb_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

hb_nm.plot.box(ax=ax1)
ax1.set_title("HB - Boxplot")
ax1.set_xlabel('HB')
ax1.set_ylabel('g/dL', rotation=0)
ax1.set_xticks([])

# hb_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
hb_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("HB - Histogram")
ax2.set_xlabel('HB(g/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# HCT

hct_nm = tsr_6.loc[:, "hct_nm"]

q1 = hct_nm.quantile(0.25)
q3 = hct_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
hct_nm[(hct_nm < inner_fence_low) | (hct_nm > inner_fence_upp)] = np.nan

hct_nm = hct_nm.fillna(round(hct_nm.mean(), 3))

# print(hct_nm)
# print(hct_nm.value_counts() / len(hct_nm))
print(hct_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

hct_nm.plot.box(ax=ax1)
ax1.set_title("HCT - Boxplot")
ax1.set_xlabel('HCT')
ax1.set_ylabel('%', rotation=0)
ax1.set_xticks([])

# hct_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
hct_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("HCT - Histogram")
ax2.set_xlabel('HCT(%)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# PLATELET

platelet_nm = tsr_6.loc[:, "platelet_nm"]

q1 = platelet_nm.quantile(0.25)
q3 = platelet_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
platelet_nm[(platelet_nm < inner_fence_low) | (platelet_nm > inner_fence_upp)] = np.nan

platelet_nm = platelet_nm.fillna(round(platelet_nm.mean(), 3))

# print(platelet_nm)
# print(platelet_nm.value_counts() / len(platelet_nm))
print(platelet_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

platelet_nm.plot.box(ax=ax1)
ax1.set_title("PLATELET - Boxplot")
ax1.set_xlabel('PLATELET')
ax1.set_ylabel('1000 cells/μL', rotation=0)
ax1.set_xticks([])

# platelet_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
platelet_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("PLATELET - Histogram")
ax2.set_xlabel('PLATELET(1000 cells/μL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# WBC

wbc_nm = tsr_6.loc[:, "wbc_nm"]

q1 = wbc_nm.quantile(0.25)
q3 = wbc_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
wbc_nm[(wbc_nm < inner_fence_low) | (wbc_nm > inner_fence_upp)] = np.nan

wbc_nm = wbc_nm.fillna(round(wbc_nm.mean(), 3))

# print(wbc_nm)
# print(wbc_nm.value_counts() / len(wbc_nm))
print(wbc_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

wbc_nm.plot.box(ax=ax1)
ax1.set_title("WBC - Boxplot")
ax1.set_xlabel('WBC')
ax1.set_ylabel('1000 cells/μL', rotation=0)
ax1.set_xticks([])

# wbc_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
wbc_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("WBC - Histogram")
ax2.set_xlabel('WBC(1000 cells/μL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# PTT1

ptt1_nm = tsr_6.loc[:, "ptt1_nm"]

q1 = ptt1_nm.quantile(0.25)
q3 = ptt1_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
ptt1_nm[(ptt1_nm < inner_fence_low) | (ptt1_nm > inner_fence_upp)] = np.nan

ptt1_nm = ptt1_nm.fillna(round(ptt1_nm.mean(), 3))

# print(ptt1_nm)
# print(ptt1_nm.value_counts() / len(ptt1_nm))
print(ptt1_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ptt1_nm.plot.box(ax=ax1)
ax1.set_title("PTT1 - Boxplot")
ax1.set_xlabel('PTT1')
ax1.set_ylabel('Second', rotation=0)
ax1.set_xticks([])

# ptt1_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
ptt1_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("PTT1 - Histogram")
ax2.set_xlabel('PTT1(Second)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# PTT2

ptt2_nm = tsr_6.loc[:, "ptt2_nm"]

q1 = ptt2_nm.quantile(0.25)
q3 = ptt2_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
ptt2_nm[(ptt2_nm < inner_fence_low) | (ptt2_nm > inner_fence_upp)] = np.nan

ptt2_nm = ptt2_nm.fillna(round(ptt2_nm.mean(), 3))

# print(ptt2_nm)
# print(ptt2_nm.value_counts() / len(ptt2_nm))
print(ptt2_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ptt2_nm.plot.box(ax=ax1)
ax1.set_title("PTT2 - Boxplot")
ax1.set_xlabel('PTT2')
ax1.set_ylabel('Second', rotation=0)
ax1.set_xticks([])

# ptt2_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
ptt2_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("PTT2 - Histogram")
ax2.set_xlabel('PTT2(Second)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# PT (INR)

ptinr_nm = tsr_6.loc[:, "ptinr_nm"]

q1 = ptinr_nm.quantile(0.25)
q3 = ptinr_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
ptinr_nm[(ptinr_nm < inner_fence_low) | (ptinr_nm > inner_fence_upp)] = np.nan

ptinr_nm = ptinr_nm.fillna(round(ptinr_nm.mean(), 3))

# print(ptinr_nm)
# print(ptinr_nmptinr_nm.value_counts() / len(ptinr_nm))
print(ptinr_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ptinr_nm.plot.box(ax=ax1)
ax1.set_title("PT(INR) - Boxplot")
ax1.set_xlabel('PT(INR)')
ax1.set_ylabel('Second', rotation=0)
ax1.set_xticks([])

# ptinr_nmptinr_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
ptinr_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("PT(INR) - Histogram")
ax2.set_xlabel('PT(INR)(Second)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# GLUCOSE (ER)

er_nm = tsr_6.loc[:, "er_nm"]

q1 = er_nm.quantile(0.25)
q3 = er_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
er_nm[(er_nm < inner_fence_low) | (er_nm > inner_fence_upp)] = np.nan

er_nm = er_nm.fillna(round(er_nm.mean(), 3))

# print(er_nm)
# print(er_nm.value_counts() / len(er_nm))
print(er_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

er_nm.plot.box(ax=ax1)
ax1.set_title("GLUCOSE (ER) - Boxplot")
ax1.set_xlabel('GLUCOSE (ER)')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# er_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
er_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("GLUCOSE (ER) - Histogram")
ax2.set_xlabel('GLUCOSE (ER)(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# BUN

bun_nm = tsr_6.loc[:, "bun_nm"]

q1 = bun_nm.quantile(0.25)
q3 = bun_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
bun_nm[(bun_nm < inner_fence_low) | (bun_nm > inner_fence_upp)] = np.nan

bun_nm = bun_nm.fillna(round(bun_nm.mean(), 3))

# print(bun_nm)
# print(bun_nm.value_counts() / len(bun_nm))
print(bun_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

bun_nm.plot.box(ax=ax1)
ax1.set_title("BUN - Boxplot")
ax1.set_xlabel('BUN')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# bun_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
bun_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("BUN - Histogram")
ax2.set_xlabel('BUN(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Creatinine

cre_nm = tsr_6.loc[:, "cre_nm"]

q1 = cre_nm.quantile(0.25)
q3 = cre_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
cre_nm[(cre_nm < inner_fence_low) | (cre_nm > inner_fence_upp)] = np.nan

cre_nm = cre_nm.fillna(round(cre_nm.mean(), 3))

# print(cre_nm)
# print(cre_nm.value_counts() / len(cre_nm))
print(cre_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

cre_nm.plot.box(ax=ax1)
ax1.set_title("Creatinine - Boxplot")
ax1.set_xlabel('Creatinine')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# cre_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
cre_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("Creatinine - Histogram")
ax2.set_xlabel('Creatinine(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Albumin ???

alb_nm = tsr_6.loc[:, "alb_nm"]
alb_nm[alb_nm == 999.9] = np.nan
alb_nm[alb_nm == 999] = np.nan
alb_nm[alb_nm == 99.9] = np.nan
alb_nm = alb_nm.fillna(round(alb_nm.mean(), 3))
# print(alb_nm)
# print(alb_nm.value_counts() / len(alb_nm))
print(alb_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

alb_nm.plot.box(ax=ax1)
ax1.set_title("Albumin - Boxplot")
ax1.set_xlabel('Albumin')
ax1.set_ylabel('g/dL', rotation=0)
ax1.set_xticks([])

# alb_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
alb_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("Albumin - Histogram")
ax2.set_xlabel('Albumin(g/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

alb_nm_1 = tsr_6.loc[:, "alb_nm"]
alb_nm_1[alb_nm_1 == 999.9] = np.nan

q1 = alb_nm_1.quantile(0.25)
q3 = alb_nm_1.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
alb_nm_1[(alb_nm_1 < inner_fence_low) | (alb_nm_1 > inner_fence_upp)] = np.nan

alb_nm_1 = alb_nm_1.fillna(round(alb_nm_1.mean(), 3))

# print(alb_nm_1)
# print(alb_nm_1.value_counts() / len(alb_nm_1))
print(alb_nm_1.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

alb_nm_1.plot.box(ax=ax1)
ax1.set_title("Albumin - Boxplot")
ax1.set_xlabel('Albumin')
ax1.set_ylabel('g/dL', rotation=0)
ax1.set_xticks([])

# alb_nm_1.plot.hist(ax = ax2, bins=100)
# plt.show()
alb_nm_1.plot.hist(ax=ax2, bins=100)
ax2.set_title("Albumin - Histogram")
ax2.set_xlabel('Albumin(g/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# CRP

crp_nm = tsr_6.loc[:, "crp_nm"]
crp_nm[crp_nm == 999.9] = np.nan
crp_nm[crp_nm == 999] = np.nan
crp_nm[crp_nm == 99.9] = np.nan
crp_nm = crp_nm.fillna(round(crp_nm.mean(), 3))
# print(crp_nm)
# print(crp_nm.value_counts() / len(crp_nm))
print(crp_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

crp_nm.plot.box(ax=ax1)
ax1.set_title("CRP - Boxplot")
ax1.set_xlabel('CRP')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# crp_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
crp_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("CRP - Histogram")
ax2.set_xlabel('CRP(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

crp_nm_1 = tsr_6.loc[:, "crp_nm"]
crp_nm_1[crp_nm_1 == 999.9] = np.nan

q1 = crp_nm_1.quantile(0.25)
q3 = crp_nm_1.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
crp_nm_1[(crp_nm_1 < inner_fence_low) | (crp_nm_1 > inner_fence_upp)] = np.nan

crp_nm_1 = crp_nm_1.fillna(round(crp_nm_1.mean(), 3))

# print(crp_nm_1)
# print(crp_nm_1.value_counts() / len(crp_nm_1))
print(crp_nm_1.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

crp_nm_1.plot.box(ax=ax1)
ax1.set_title("CRP - Boxplot")
ax1.set_xlabel('CRP')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# crp_nm_1.plot.hist(ax = ax2, bins=100)
# plt.show()
crp_nm_1.plot.hist(ax=ax2, bins=100)
ax2.set_title("CRP - Histogram")
ax2.set_xlabel('CRP(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# HbA1c

hbac_nm = tsr_6.loc[:, "hbac_nm"]
hbac_nm[hbac_nm == 999.9] = np.nan
hbac_nm = hbac_nm.fillna(round(hbac_nm.mean(), 3))
# print(hbac_nm)
# print(hbac_nm.value_counts() / len(hbac_nm))
print(hbac_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

hbac_nm.plot.box(ax=ax1)
ax1.set_title("HbA1c - Boxplot")
ax1.set_xlabel('HbA1c')
ax1.set_ylabel('%', rotation=0)
ax1.set_xticks([])

# hbac_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
hbac_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("HbA1c - Histogram")
ax2.set_xlabel('HbA1c(%)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

hbac_nm_1 = tsr_6.loc[:, "hbac_nm"]
hbac_nm_1[hbac_nm_1 == 999.9] = np.nan

q1 = hbac_nm_1.quantile(0.25)
q3 = hbac_nm_1.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
hbac_nm_1[(hbac_nm_1 < inner_fence_low) | (hbac_nm_1 > inner_fence_upp)] = np.nan

hbac_nm_1 = hbac_nm_1.fillna(round(hbac_nm_1.mean(), 3))

# print(hbac_nm_1)
# print(hbac_nm_1.value_counts() / len(hbac_nm_1))
print(hbac_nm_1.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

hbac_nm_1.plot.box(ax=ax1)
ax1.set_title("HbA1c - Boxplot")
ax1.set_xlabel('HbA1c')
ax1.set_ylabel('%', rotation=0)
ax1.set_xticks([])

# hbac_nm_1.plot.hist(ax = ax2, bins=100)
# plt.show()
hbac_nm_1.plot.hist(ax=ax2, bins=100)
ax2.set_title("HbA1c - Histogram")
ax2.set_xlabel('HbA1c(%)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# 抽血結果 - 第一次空腹抽血

# Glu (AC)

ac_nm = tsr_6.loc[:, "ac_nm"]

q1 = ac_nm.quantile(0.25)
q3 = ac_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
ac_nm[(ac_nm < inner_fence_low) | (ac_nm > inner_fence_upp)] = np.nan

ac_nm = ac_nm.fillna(round(ac_nm.mean(), 3))

# print(ac_nm)
# print(ac_nm.value_counts() / len(ac_nm))
print(ac_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ac_nm.plot.box(ax=ax1)
ax1.set_title("Glu (AC) - Boxplot")
ax1.set_xlabel('Glu (AC)')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# ac_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
ac_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("Glu (AC) - Histogram")
ax2.set_xlabel('Glu (AC)(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# UA

ua_nm = tsr_6.loc[:, "ua_nm"]
ua_nm[ua_nm == 999.9] = np.nan

q1 = ua_nm.quantile(0.25)
q3 = ua_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
ua_nm[(ua_nm < inner_fence_low) | (ua_nm > inner_fence_upp)] = np.nan

ua_nm = ua_nm.fillna(round(ua_nm.mean(), 3))

# print(ua_nm)
# print(ua_nm.value_counts() / len(ua_nm))
print(ua_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ua_nm.plot.box(ax=ax1)
ax1.set_title("UA - Boxplot")
ax1.set_xlabel('UA')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# ua_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
ua_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("UA - Histogram")
ax2.set_xlabel('UA(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# T-CHO

tcho_nm = tsr_6.loc[:, "tcho_nm"]
tcho_nm[tcho_nm == 999.9] = np.nan

q1 = tcho_nm.quantile(0.25)
q3 = tcho_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
tcho_nm[(tcho_nm < inner_fence_low) | (tcho_nm > inner_fence_upp)] = np.nan

tcho_nm = tcho_nm.fillna(round(tcho_nm.mean(), 3))

# print(tcho_nm)
# print(tcho_nm.value_counts() / len(tcho_nm))
print(tcho_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

tcho_nm.plot.box(ax=ax1)
ax1.set_title("T-CHO - Boxplot")
ax1.set_xlabel('T-CHO')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# tcho_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
tcho_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("T-CHO - Histogram")
ax2.set_xlabel('T-CHO(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# TG

tg_nm = tsr_6.loc[:, "tg_nm"]
tg_nm[tg_nm == 999.9] = np.nan
tg_nm[tg_nm == 2000] = np.nan

q1 = tg_nm.quantile(0.25)
q3 = tg_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
tg_nm[(tg_nm < inner_fence_low) | (tg_nm > inner_fence_upp)] = np.nan

tg_nm = tg_nm.fillna(round(tg_nm.mean(), 3))

# print(tg_nm)
# print(tg_nm.value_counts() / len(tg_nm))
print(tg_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

tg_nm.plot.box(ax=ax1)
ax1.set_title("TG - Boxplot")
ax1.set_xlabel('TG')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# tg_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
tg_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("TG - Histogram")
ax2.set_xlabel('TG(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# HDL

hdl_nm = tsr_6.loc[:, "hdl_nm"]
hdl_nm[hdl_nm == 999.9] = np.nan

q1 = hdl_nm.quantile(0.25)
q3 = hdl_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
hdl_nm[(hdl_nm < inner_fence_low) | (hdl_nm > inner_fence_upp)] = np.nan

hdl_nm = hdl_nm.fillna(round(hdl_nm.mean(), 3))

# print(hdl_nm)
# print(hdl_nm.value_counts() / len(hdl_nm))
print(hdl_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

hdl_nm.plot.box(ax=ax1)
ax1.set_title("HDL - Boxplot")
ax1.set_xlabel('HDL')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# hdl_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
hdl_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("HDL - Histogram")
ax2.set_xlabel('HDL(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# LDL

ldl_nm = tsr_6.loc[:, "ldl_nm"]
ldl_nm[ldl_nm == 999.9] = np.nan

q1 = ldl_nm.quantile(0.25)
q3 = ldl_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
ldl_nm[(ldl_nm < inner_fence_low) | (ldl_nm > inner_fence_upp)] = np.nan

ldl_nm = ldl_nm.fillna(round(ldl_nm.mean(), 3))

# print(ldl_nm)
# print(ldl_nm.value_counts() / len(ldl_nm))
print(ldl_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

ldl_nm.plot.box(ax=ax1)
ax1.set_title("LDL - Boxplot")
ax1.set_xlabel('LDL')
ax1.set_ylabel('mg/dL', rotation=0)
ax1.set_xticks([])

# ldl_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
ldl_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("LDL - Histogram")
ax2.set_xlabel('LDL(mg/dL)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# GOT

got_nm = tsr_6.loc[:, "got_nm"]
got_nm[got_nm == 999.9] = np.nan
got_nm = pd.to_numeric(got_nm, errors="coerce")

q1 = got_nm.quantile(0.25)
q3 = got_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
got_nm[(got_nm < inner_fence_low) | (got_nm > inner_fence_upp)] = np.nan

got_nm = got_nm.fillna(round(got_nm.mean(), 3))

got_nm = pd.to_numeric(got_nm, errors='coerce')
# print(got_nm)
# print(got_nm.value_counts() / len(got_nm))
print(got_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

got_nm.plot.box(ax=ax1)
ax1.set_title("GOT - Boxplot")
ax1.set_xlabel('GOT')
ax1.set_ylabel('U/L', rotation=0)
ax1.set_xticks([])

# got_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
got_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("GOT - Histogram")
ax2.set_xlabel('GOT(U/L)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# GPT

gpt_nm = tsr_6.loc[:, "gpt_nm"]
gpt_nm = pd.to_numeric(gpt_nm, errors="coerce")
gpt_nm[gpt_nm == 999.9] = np.nan

q1 = gpt_nm.quantile(0.25)
q3 = gpt_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
gpt_nm[(gpt_nm < inner_fence_low) | (gpt_nm > inner_fence_upp)] = np.nan

gpt_nm = gpt_nm.fillna(round(gpt_nm.mean(), 3))

# print(gpt_nm)
# print(gpt_nm.value_counts() / len(gpt_nm))
print(gpt_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

gpt_nm.plot.box(ax=ax1)
ax1.set_title("GPT - Boxplot")
ax1.set_xlabel('GPT')
ax1.set_ylabel('U/L', rotation=0)
ax1.set_xticks([])

# gpt_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
gpt_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("GPT - Histogram")
ax2.set_xlabel('GPT(U/L)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# 離院情形

off_id = tsr_6.loc[:, "off_id"]
off_id[(off_id != 1) & (off_id != 2) & (off_id != 3)] = np.nan
off_id = pd.to_numeric(off_id, errors='coerce')
off_id = off_id.fillna(off_id.mode()[0])
# print(off_id)
print(off_id.value_counts() / len(off_id))
# print(off_id.describe())

off_id_labels = ["病危自動離院", "死亡", "離院"]
sns.countplot(off_id).set_xticklabels(off_id_labels)
plt.title("離院情形 - Barplot")
plt.xlabel('離院情形')
plt.ylabel('Number', rotation=0)
plt.show()

# 死亡日期

offd_dt = tsr_6.loc[:, "offd_dt"]
offd_dt = pd.to_datetime(offd_dt, errors='coerce')
offd_dt[(offd_dt.dt.year < 2006) | (offd_dt.dt.year > 2021)] = np.nan
offd_dt = offd_dt.fillna(offd_dt.mode()[0])
offd_dt[off_id != 2] = np.nan
# print(offd_dt)
# print(offd_dt.value_counts() / len(offd_dt))
print(offd_dt.describe())

offd_dt.value_counts().plot()
plt.title("死亡日期 - Lineplot")
plt.xlabel('死亡日期')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# 死因

offd_id = tsr_6.loc[:, "offd_id"]
offd_id = pd.to_numeric(offd_id, errors="coerce")
offd_id[(offd_id != 1) & (offd_id != 2) & (offd_id != 99)] = np.nan
offd_id = offd_id.fillna(offd_id.mode()[0])
offd_id[off_id != 2] = np.nan
# print(offd_id)
print(offd_id.value_counts() / len(offd_id))
# print(offd_id.describe())

# offd_id_labels = ["中風直接致死","併發症致死","其他"]
offd_id_labels = ["其他"]
sns.countplot(offd_id)  # .set_xticklabels(offd_id_labels)
plt.title("死因 - Barplot")
plt.xlabel('死因')
plt.ylabel('Number', rotation=0)
plt.show()
offd_id = offd_id.fillna(999)

# OFFD_TX

# 離院，目的地

offdt_id = tsr_6.loc[:, "offdt_id"]
offdt_id = pd.to_numeric(offdt_id, errors="coerce")
offdt_id[(offdt_id != 1) & (offdt_id != 2) & (offdt_id != 3) & (offdt_id != 4) & (offdt_id != 5)] = np.nan
offdt_id = offdt_id.fillna(offdt_id.mode()[0])
# print(offdt_id)
print(offdt_id.value_counts() / len(offdt_id))
# print(offdt_id.describe())

offdt_id_labels = ["回家", "護理之家", "轉院", "呼吸病房(RCW)", "轉復健"]
sns.countplot(offdt_id).set_xticklabels(offdt_id_labels)
plt.title("離院，目的地 - Barplot")
plt.xlabel('離院，目的地')
plt.ylabel('Number', rotation=0)
plt.show()

# 離院，轉院醫院

offdtorg_id = tsr_6.loc[:, "offdtorg_id"]
# print(offdtorg_id)
print(offdtorg_id.value_counts() / len(offdtorg_id))
# print(offdtorg_id.describe())

# OFFDTORG_TX

# 轉復健日期

offre_dt = tsr_6.loc[:, "offre_dt"]
offre_dt = pd.to_datetime(offre_dt, errors='coerce')
offre_dt[(offre_dt.dt.year < 2006) | (offre_dt.dt.year > 2021)] = np.nan
offre_dt = offre_dt.fillna(offre_dt.mode()[0])
offre_dt[off_id != 3] = np.nan
# print(offre_dt)
# print(offre_dt.value_counts() / len(offre_dt))
print(offre_dt.describe())

offre_dt.value_counts().plot()
plt.title("轉復健日期 - Lineplot")
plt.xlabel('轉復健日期')
plt.ylabel('Number', rotation=0)
plt.xticks()
plt.show()

# Feeding

feeding = tsr_6.loc[:, "feeding"]
feeding = pd.to_numeric(feeding, errors="coerce")
feeding[(feeding < 0) | (feeding > 10)] = np.nan
feeding = feeding.fillna(feeding.mode()[0])
# print(feeding)
# print(feeding.value_counts() / len(feeding))
print(feeding.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

feeding.plot.box(ax=ax1)
ax1.set_title("Feeding - Boxplot")
ax1.set_xlabel('Feeding')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

feeding.plot.hist(ax=ax2)
ax2.set_title("Feeding - Histogram")
ax2.set_xlabel('Feeding(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Transfers

transfers = tsr_6.loc[:, "transfers"]
transfers = pd.to_numeric(transfers, errors="coerce")
transfers[(transfers < 0) | (transfers > 15)] = np.nan
transfers = transfers.fillna(transfers.mode()[0])
# print(transfers)
# print(transfers.value_counts() / len(transfers))
print(transfers.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

transfers.plot.box(ax=ax1)
ax1.set_title("Transfers - Boxplot")
ax1.set_xlabel('Transfers')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

transfers.plot.hist(ax=ax2)
ax2.set_title("Transfers - Histogram")
ax2.set_xlabel('Transfers(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Bathing

bathing = tsr_6.loc[:, "bathing"]
bathing = pd.to_numeric(bathing, errors="coerce")
bathing[(bathing < 0) | (bathing > 5)] = np.nan
bathing = bathing.fillna(bathing.mode()[0])
# print(bathing)
# print(bathing.value_counts() / len(bathing))
print(bathing.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

bathing.plot.box(ax=ax1)
ax1.set_title("Bathing - Boxplot")
ax1.set_xlabel('Bathing')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

bathing.plot.hist(ax=ax2)
ax2.set_title("Bathing - Histogram")
ax2.set_xlabel('Bathing(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Toilet

toilet_use = tsr_6.loc[:, "toilet_use"]
toilet_use = pd.to_numeric(toilet_use, errors="coerce")
toilet_use[(toilet_use < 0) | (toilet_use > 10)] = np.nan
toilet_use = toilet_use.fillna(toilet_use.mode()[0])
# print(toilet_use)
# print(toilet_use.value_counts() / len(toilet_use))
print(toilet_use.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

toilet_use.plot.box(ax=ax1)
ax1.set_title("toilet_use - Boxplot")
ax1.set_xlabel('toilet_use')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

toilet_use.plot.hist(ax=ax2)
ax2.set_title("toilet_use - Histogram")
ax2.set_xlabel('toilet_use(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Grooming

grooming = tsr_6.loc[:, "grooming"]
grooming = pd.to_numeric(grooming, errors="coerce")
grooming[(grooming < 0) | (grooming > 5)] = np.nan
grooming = grooming.fillna(grooming.mode()[0])
# print(grooming)
# print(grooming.value_counts() / len(grooming))
print(grooming.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

grooming.plot.box(ax=ax1)
ax1.set_title("Grooming - Boxplot")
ax1.set_xlabel('Grooming')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

grooming.plot.hist(ax=ax2)
ax2.set_title("Grooming - Histogram")
ax2.set_xlabel('Grooming(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Mobility

mobility = tsr_6.loc[:, "mobility"]
mobility = pd.to_numeric(mobility, errors="coerce")
mobility[(mobility < 0) | (mobility > 15)] = np.nan
mobility = mobility.fillna(mobility.mode()[0])
# print(mobility)
# print(mobility.value_counts() / len(mobility))
print(mobility.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

mobility.plot.box(ax=ax1)
ax1.set_title("Mobility - Boxplot")
ax1.set_xlabel('Mobility')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

mobility.plot.hist(ax=ax2)
ax2.set_title("Mobility - Histogram")
ax2.set_xlabel('Mobility(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Stairs

stairs = tsr_6.loc[:, "stairs"]
stairs = pd.to_numeric(stairs, errors="coerce")
stairs[(stairs < 0) | (stairs > 10)] = np.nan
stairs = stairs.fillna(stairs.mode()[0])
# print(stairs)
# print(stairs.value_counts() / len(stairs))
print(stairs.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

stairs.plot.box(ax=ax1)
ax1.set_title("Stairs - Boxplot")
ax1.set_xlabel('Stairs')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

stairs.plot.hist(ax=ax2)
ax2.set_title("Stairs - Histogram")
ax2.set_xlabel('Stairs(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Dressing

dressing = tsr_6.loc[:, "dressing"]
dressing = pd.to_numeric(dressing, errors="coerce")
dressing[(dressing < 0) | (dressing > 10)] = np.nan
dressing = dressing.fillna(dressing.mode()[0])
# print(dressing)
# print(dressing.value_counts() / len(dressing))
print(dressing.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

dressing.plot.box(ax=ax1)
ax1.set_title("Dressing - Boxplot")
ax1.set_xlabel('Dressing')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

dressing.plot.hist(ax=ax2)
ax2.set_title("Dressing - Histogram")
ax2.set_xlabel('Dressing(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Bowel Control

bowel_control = tsr_6.loc[:, "bowel_control"]
bowel_control = pd.to_numeric(bowel_control, errors="coerce")
bowel_control[(bowel_control < 0) | (bowel_control > 10)] = np.nan
bowel_control = bowel_control.fillna(bowel_control.mode()[0])
# print(bowel_control)
# print(bowel_control.value_counts() / len(bowel_control))
print(bowel_control.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

bowel_control.plot.box(ax=ax1)
ax1.set_title("Bowel Control - Boxplot")
ax1.set_xlabel('Bowel Control')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

bowel_control.plot.hist(ax=ax2)
ax2.set_title("Bowel Control - Histogram")
ax2.set_xlabel('Bowel Control(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Bladder Control

bladder_control = tsr_6.loc[:, "bladder_control"]
bladder_control = pd.to_numeric(bladder_control, errors="coerce")
bladder_control[(bladder_control < 0) | (bladder_control > 10)] = np.nan
bladder_control = bladder_control.fillna(bladder_control.mode()[0])
# print(bladder_control)
# print(bladder_control.value_counts() / len(bladder_control))
print(bladder_control.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

bladder_control.plot.box(ax=ax1)
ax1.set_title("Bladder Control - Boxplot")
ax1.set_xlabel('Bladder Control')
ax1.set_ylabel('Times', rotation=0)
ax1.set_xticks([])

bladder_control.plot.hist(ax=ax2)
ax2.set_title("Bladder Control - Histogram")
ax2.set_xlabel('Bladder Control(Times)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Total

total = feeding + transfers + bathing + toilet_use + grooming + mobility + dressing + stairs + bowel_control + bladder_control
total = total.fillna(total.mean())
# print(total)
# print(total.value_counts() / len(total))
print(total.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

total.plot.box(ax=ax1)
ax1.set_title("Total Barthel Index - Boxplot")
ax1.set_xlabel('Total Barthel Index')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

total.plot.hist(ax=ax2)
ax2.set_title("Total Barthel Index - Histogram")
ax2.set_xlabel('Total Barthel Index(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Discharged mRS

discharged_mrs = tsr_6.loc[:, "discharged_mrs"]
discharged_mrs = pd.to_numeric(discharged_mrs, errors="coerce")
discharged_mrs[(discharged_mrs != 0) & (discharged_mrs != 1) & (discharged_mrs != 2) & (discharged_mrs != 3) & (
            discharged_mrs != 4) & (discharged_mrs != 5) & (discharged_mrs != 6)] = np.nan
discharged_mrs = discharged_mrs.fillna(discharged_mrs.mode()[0])
# print(discharged_mrs)
print(discharged_mrs.value_counts() / len(discharged_mrs))
print(discharged_mrs.describe())

sns.countplot(discharged_mrs)
plt.title("Discharged mRS - Histogram")
plt.xlabel('Discharged mRS(Score)')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Cortical ACA_CT_Right

cortical_aca_ctr = tsr_6.loc[:, "cortical_aca_ctr"]
cortical_aca_ctr = cortical_aca_ctr.fillna(cortical_aca_ctr.mode()[0])
# print(cortical_aca_ctr)
print(cortical_aca_ctr.value_counts() / len(cortical_aca_ctr))
# print(cortical_aca_ctr.describe())

sns.countplot(cortical_aca_ctr)
plt.title("Cortical_ACA_ctr - Barplot")
plt.xlabel('Cortical_ACA_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Cortical MCA_CT_Right

cortical_mca_ctr = tsr_6.loc[:, "cortical_mca_ctr"]
cortical_mca_ctr[cortical_mca_ctr == '0'] = "N"
cortical_mca_ctr = cortical_mca_ctr.fillna(cortical_mca_ctr.mode()[0])
# print(cortical_mca_ctr)
print(cortical_mca_ctr.value_counts() / len(cortical_mca_ctr))
# print(cortical_mca_ctr.describe())

sns.countplot(cortical_mca_ctr)
plt.title("Cortical_MCA_ctr - Barplot")
plt.xlabel('Cortical_MCA_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Subcortical ACA_CT_Right

subcortical_aca_ctr = tsr_6.loc[:, "subcortical_aca_ctr"]
subcortical_aca_ctr = subcortical_aca_ctr.fillna(subcortical_aca_ctr.mode()[0])
# print(subcortical_aca_ctr)
print(subcortical_aca_ctr.value_counts() / len(subcortical_aca_ctr))
# print(subcortical_aca_ctr.describe())

sns.countplot(subcortical_aca_ctr)
plt.title("Subcortical_ACA_ctr - Barplot")
plt.xlabel('Subcortical_ACA_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Subcortical MCA_CT_Right

subcortical_mca_ctr = tsr_6.loc[:, "subcortical_mca_ctr"]
subcortical_mca_ctr[subcortical_mca_ctr == '0'] = "N"
subcortical_mca_ctr = subcortical_mca_ctr.fillna(subcortical_mca_ctr.mode()[0])
# print(subcortical_mca_ctr)
print(subcortical_mca_ctr.value_counts() / len(subcortical_mca_ctr))
# print(subcortical_mca_ctr.describe())

sns.countplot(subcortical_mca_ctr)
plt.title("Subcortical_MCA_ctr - Barplot")
plt.xlabel('Subcortical_MCA_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation PCA_cortex_CT_Right

pca_cortex_ctr = tsr_6.loc[:, "pca_cortex_ctr"]
pca_cortex_ctr[pca_cortex_ctr == '0'] = "N"
pca_cortex_ctr = pca_cortex_ctr.fillna(pca_cortex_ctr.mode()[0])
# print(pca_cortex_ctr)
print(pca_cortex_ctr.value_counts() / len(pca_cortex_ctr))
# print(pca_cortex_ctr.describe())

sns.countplot(pca_cortex_ctr)
plt.title("PCA_cortex_ctr - Barplot")
plt.xlabel('PCA_cortex_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Thalamus_CT_Right

thalamus_ctr = tsr_6.loc[:, "thalamus_ctr"]
thalamus_ctr[thalamus_ctr == '0'] = "N"
thalamus_ctr = thalamus_ctr.fillna(thalamus_ctr.mode()[0])
# print(thalamus_ctr)
print(thalamus_ctr.value_counts() / len(thalamus_ctr))
# print(thalamus_ctr.describe())

sns.countplot(thalamus_ctr)
plt.title("Thalamus_ctr - Barplot")
plt.xlabel('Thalamus_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Brainstem_CT_Right

brainstem_ctr = tsr_6.loc[:, "brainstem_ctr"]
brainstem_ctr = brainstem_ctr.fillna(brainstem_ctr.mode()[0])
# print(brainstem_ctr)
print(brainstem_ctr.value_counts() / len(brainstem_ctr))
# print(brainstem_ctr.describe())

sns.countplot(brainstem_ctr)
plt.title("Brainstem_ctr - Barplot")
plt.xlabel('Brainstem_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Cerebellum_CT_Right

cerebellum_ctr = tsr_6.loc[:, "cerebellum_ctr"]
cerebellum_ctr = cerebellum_ctr.fillna(cerebellum_ctr.mode()[0])
# print(cerebellum_ctr)
print(cerebellum_ctr.value_counts() / len(cerebellum_ctr))
# print(cerebellum_ctr.describe())

sns.countplot(cerebellum_ctr)
plt.title("Cerebellum_ctr - Barplot")
plt.xlabel('Cerebellum_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Watershed_CT_Right

watershed_ctr = tsr_6.loc[:, "watershed_ctr"]
watershed_ctr[watershed_ctr == '0'] = "N"
watershed_ctr = watershed_ctr.fillna(watershed_ctr.mode()[0])
# print(watershed_ctr)
print(watershed_ctr.value_counts() / len(watershed_ctr))
# print(watershed_ctr.describe())

sns.countplot(watershed_ctr)
plt.title("Watershed_ctr - Barplot")
plt.xlabel('Watershed_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Hemorrhagic_infarct_CT_Right

hemorrhagic_infarct_ctr = tsr_6.loc[:, "hemorrhagic_infarct_ctr"]
hemorrhagic_infarct_ctr[hemorrhagic_infarct_ctr == '0'] = "N"
hemorrhagic_infarct_ctr = hemorrhagic_infarct_ctr.fillna(hemorrhagic_infarct_ctr.mode()[0])
# print(hemorrhagic_infarct_ctr)
print(hemorrhagic_infarct_ctr.value_counts() / len(hemorrhagic_infarct_ctr))
# print(hemorrhagic_infarct_ctr.describe())

sns.countplot(hemorrhagic_infarct_ctr)
plt.title("Hemorrhagic_infarct_ctr - Barplot")
plt.xlabel('Hemorrhagic_infarct_ctr')
plt.ylabel('Number', rotation=0)
plt.show()

# Old_stroke_CTci

old_stroke_ctci = tsr_6.loc[:, "old_stroke_ctci"]
old_stroke_ctci[old_stroke_ctci == '0'] = 'N'
old_stroke_ctci = old_stroke_ctci.fillna(old_stroke_ctci.mode()[0])
# print(old_stroke_ctci)
print(old_stroke_ctci.value_counts() / len(old_stroke_ctci))
# print(old_stroke_ctci.describe())

sns.countplot(old_stroke_ctci)
plt.title("Old_stroke_ctci - Barplot")
plt.xlabel('Old_stroke_ctci')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Cortical ACA_CT_Left

cortical_aca_ctl = tsr_6.loc[:, "cortical_aca_ctl"]
cortical_aca_ctl[cortical_aca_ctl == '0'] = 'N'
cortical_aca_ctl = cortical_aca_ctl.fillna(cortical_aca_ctl.mode()[0])
# print(cortical_aca_ctl)
print(cortical_aca_ctl.value_counts() / len(cortical_aca_ctl))
# print(cortical_aca_ctl.describe())

sns.countplot(cortical_aca_ctl)
plt.title("Cortical_ACA_ctl - Barplot")
plt.xlabel('Cortical_ACA_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Cortical MCA_CT_Left

cortical_mca_ctl = tsr_6.loc[:, "cortical_mca_ctl"]
cortical_mca_ctl = cortical_mca_ctl.fillna(cortical_mca_ctl.mode()[0])
# print(cortical_mca_ctl)
print(cortical_mca_ctl.value_counts() / len(cortical_mca_ctl))
# print(cortical_mca_ctl.describe())

sns.countplot(cortical_mca_ctl)
plt.title("Cortical_MCA_ctl - Barplot")
plt.xlabel('Cortical_MCA_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Subcortical ACA_CT_Left

subcortical_aca_ctl = tsr_6.loc[:, "subcortical_aca_ctl"]
subcortical_aca_ctl = subcortical_aca_ctl.fillna(subcortical_aca_ctl.mode()[0])
# print(subcortical_aca_ctl)
print(subcortical_aca_ctl.value_counts() / len(subcortical_aca_ctl))
# print(subcortical_aca_ctl.describe())

sns.countplot(subcortical_aca_ctl)
plt.title("Subcortical_ACA_ctl - Barplot")
plt.xlabel('Subcortical_ACA_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Subcortical MCA_CT_Left

subcortical_mca_ctl = tsr_6.loc[:, "subcortical_mca_ctl"]
subcortical_mca_ctl = subcortical_mca_ctl.fillna(subcortical_mca_ctl.mode()[0])
# print(subcortical_mca_ctl)
print(subcortical_mca_ctl.value_counts() / len(subcortical_mca_ctl))
# print(subcortical_mca_ctl.describe())

sns.countplot(subcortical_mca_ctl)
plt.title("Subcortical_MCA_ctl - Barplot")
plt.xlabel('Subcortical_MCA_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation PCA_cortex_CT_Left

pca_cortex_ctl = tsr_6.loc[:, "pca_cortex_ctl"]
pca_cortex_ctl[pca_cortex_ctl == '0'] = 'N'
pca_cortex_ctl = pca_cortex_ctl.fillna(pca_cortex_ctl.mode()[0])
# print(pca_cortex_ctl)
print(pca_cortex_ctl.value_counts() / len(pca_cortex_ctl))
# print(pca_cortex_ctl.describe())

sns.countplot(pca_cortex_ctl)
plt.title("PCA_cortex_ctl - Barplot")
plt.xlabel('PCA_cortex_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Thalamus_CT_Left

thalamus_ctl = tsr_6.loc[:, "thalamus_ctl"]
thalamus_ctl = thalamus_ctl.fillna(thalamus_ctl.mode()[0])
# print(thalamus_ctl)
print(thalamus_ctl.value_counts() / len(thalamus_ctl))
# print(thalamus_ctl.describe())

sns.countplot(thalamus_ctl)
plt.title("Thalamus_ctl - Barplot")
plt.xlabel('Thalamus_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Brainstem_CT_Left

brainstem_ctl = tsr_6.loc[:, "brainstem_ctl"]
brainstem_ctl[brainstem_ctl == '0'] = 'N'
brainstem_ctl = brainstem_ctl.fillna(brainstem_ctl.mode()[0])
# print(brainstem_ctl)
print(brainstem_ctl.value_counts() / len(brainstem_ctl))
# print(brainstem_ctl.describe())

sns.countplot(brainstem_ctl)
plt.title("Brainstem_ctl - Barplot")
plt.xlabel('Brainstem_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Cerebellum_CT_Left

cerebellum_ctl = tsr_6.loc[:, "cerebellum_ctl"]
cerebellum_ctl = cerebellum_ctl.fillna(cerebellum_ctl.mode()[0])
# print(cerebellum_ctl)
print(cerebellum_ctl.value_counts() / len(cerebellum_ctl))
# print(cerebellum_ctl.describe())

sns.countplot(cerebellum_ctl)
plt.title("Cerebellum_ctl - Barplot")
plt.xlabel('Cerebellum_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Watershed_CT_Left

watershed_ctl = tsr_6.loc[:, "watershed_ctl"]
watershed_ctl = watershed_ctl.fillna(watershed_ctl.mode()[0])
# print(watershed_ctl)
print(watershed_ctl.value_counts() / len(watershed_ctl))
# print(watershed_ctl.describe())

sns.countplot(watershed_ctl)
plt.title("Watershed_ctl - Barplot")
plt.xlabel('Watershed_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Hemorrhagic_infarct_CT_Left

hemorrhagic_infarct_ctl = tsr_6.loc[:, "hemorrhagic_infarct_ctl"]
hemorrhagic_infarct_ctl[hemorrhagic_infarct_ctl == '0'] = 'N'
hemorrhagic_infarct_ctl = hemorrhagic_infarct_ctl.fillna(hemorrhagic_infarct_ctl.mode()[0])
# print(hemorrhagic_infarct_ctl)
print(hemorrhagic_infarct_ctl.value_counts() / len(hemorrhagic_infarct_ctl))
# print(hemorrhagic_infarct_ctl.describe())

sns.countplot(hemorrhagic_infarct_ctl)
plt.title("Hemorrhagic_infarct_ctl - Barplot")
plt.xlabel('Hemorrhagic_infarct_ctl')
plt.ylabel('Number', rotation=0)
plt.show()

# Old_stroke_CTch

old_stroke_ctch = tsr_6.loc[:, "old_stroke_ctch"]
old_stroke_ctch[old_stroke_ctch == '0'] = 'N'
old_stroke_ctch = old_stroke_ctch.fillna(old_stroke_ctch.mode()[0])
# print(old_stroke_ctch)
print(old_stroke_ctch.value_counts() / len(old_stroke_ctch))
# print(old_stroke_ctch.describe())

sns.countplot(old_stroke_ctch)
plt.title("Old_stroke_ctch - Barplot")
plt.xlabel('Old_stroke_ctch')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Cortical ACA_MRI_Right

cortical_aca_mrir = tsr_6.loc[:, "cortical_aca_mrir"]
cortical_aca_mrir[cortical_aca_mrir == '0'] = 'N'
cortical_aca_mrir = cortical_aca_mrir.fillna(cortical_aca_mrir.mode()[0])
# print(cortical_aca_mrir)
print(cortical_aca_mrir.value_counts() / len(cortical_aca_mrir))
# print(cortical_aca_mrir.describe())

sns.countplot(cortical_aca_mrir)
plt.title("Cortical_ACA_mrir - Barplot")
plt.xlabel('Cortical_ACA_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Cortical MCA_MRI_Right

cortical_mca_mrir = tsr_6.loc[:, "cortical_mca_mrir"]
cortical_mca_mrir = cortical_mca_mrir.fillna(cortical_mca_mrir.mode()[0])
# print(cortical_mca_mrir)
print(cortical_mca_mrir.value_counts() / len(cortical_mca_mrir))
# print(cortical_mca_mrir.describe())

sns.countplot(cortical_mca_mrir)
plt.title("Cortical_MCA_mrir - Barplot")
plt.xlabel('Cortical_MCA_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Subcortical ACA_MRI_Right

subcortical_aca_mrir = tsr_6.loc[:, "subcortical_aca_mrir"]
subcortical_aca_mrir = subcortical_aca_mrir.fillna(subcortical_aca_mrir.mode()[0])
# print(subcortical_aca_mrir)
print(subcortical_aca_mrir.value_counts() / len(subcortical_aca_mrir))
# print(subcortical_aca_mrir.describe())

sns.countplot(subcortical_aca_mrir)
plt.title("Subcortical_ACA_mrir - Barplot")
plt.xlabel('Subcortical_ACA_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Subcortical MCA_MRI_Right

subcortical_mca_mrir = tsr_6.loc[:, "subcortical_mca_mrir"]
subcortical_mca_mrir[subcortical_mca_mrir == '1'] = "Y"
subcortical_mca_mrir = subcortical_mca_mrir.fillna(subcortical_mca_mrir.mode()[0])
# print(subcortical_mca_mrir)
print(subcortical_mca_mrir.value_counts() / len(subcortical_mca_mrir))
# print(subcortical_mca_mrir.describe())

sns.countplot(subcortical_mca_mrir)
plt.title("Subcortical_MCA_mrir - Barplot")
plt.xlabel('Subcortical_MCA_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation PCA_cortex_MRI_Right

pca_cortex_mrir = tsr_6.loc[:, "pca_cortex_mrir"]
pca_cortex_mrir = pca_cortex_mrir.fillna(pca_cortex_mrir.mode()[0])
# print(pca_cortex_mrir)
print(pca_cortex_mrir.value_counts() / len(pca_cortex_mrir))
# print(pca_cortex_mrir.describe())

sns.countplot(pca_cortex_mrir)
plt.title("PCA_cortex_mrir - Barplot")
plt.xlabel('PCA_cortex_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Thalamus_MRI_Right

thalamus_mrir = tsr_6.loc[:, "thalamus_mrir"]
thalamus_mrir = thalamus_mrir.fillna(thalamus_mrir.mode()[0])
# print(thalamus_mrir)
print(thalamus_mrir.value_counts() / len(thalamus_mrir))
# print(thalamus_mrir.describe())

sns.countplot(thalamus_mrir)
plt.title("Thalamus_mrir - Barplot")
plt.xlabel('Thalamus_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Brainstem_MRI_Right

brainstem_mrir = tsr_6.loc[:, "brainstem_mrir"]
brainstem_mrir[brainstem_mrir == '0'] = "N"
brainstem_mrir = brainstem_mrir.fillna(brainstem_mrir.mode()[0])
# print(brainstem_mrir)
print(brainstem_mrir.value_counts() / len(brainstem_mrir))
# print(brainstem_mrir.describe())

sns.countplot(brainstem_mrir)
plt.title("Brainstem_mrir - Barplot")
plt.xlabel('Brainstem_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Cerebellum_MRI_Right

cerebellum_mrir = tsr_6.loc[:, "cerebellum_mrir"]
cerebellum_mrir = cerebellum_mrir.fillna(cerebellum_mrir.mode()[0])
# print(cerebellum_mrir)
print(cerebellum_mrir.value_counts() / len(cerebellum_mrir))
# print(cerebellum_mrir.describe())

sns.countplot(cerebellum_mrir)
plt.title("Cerebellum_mrir - Barplot")
plt.xlabel('Cerebellum_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Watershed_MRI_Right

watershed_mrir = tsr_6.loc[:, "watershed_mrir"]
watershed_mrir[watershed_mrir == '0'] = "N"
watershed_mrir = watershed_mrir.fillna(watershed_mrir.mode()[0])
# print(watershed_mrir)
print(watershed_mrir.value_counts() / len(watershed_mrir))
# print(watershed_mrir.describe())

sns.countplot(watershed_mrir)
plt.title("Watershed_mrir - Barplot")
plt.xlabel('Watershed_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Hemorrhagic_infarct_MRI_Right

hemorrhagic_infarct_mrir = tsr_6.loc[:, "hemorrhagic_infarct_mrir"]
hemorrhagic_infarct_mrir[hemorrhagic_infarct_mrir == '0'] = 'N'
hemorrhagic_infarct_mrir = hemorrhagic_infarct_mrir.fillna(hemorrhagic_infarct_mrir.mode()[0])
# print(hemorrhagic_infarct_mrir)
print(hemorrhagic_infarct_mrir.value_counts() / len(hemorrhagic_infarct_mrir))
# print(hemorrhagic_infarct_mrir.describe())

sns.countplot(hemorrhagic_infarct_mrir)
plt.title("Hemorrhagic_infarct_mrir - Barplot")
plt.xlabel('Hemorrhagic_infarct_mrir')
plt.ylabel('Number', rotation=0)
plt.show()

# Old_stroke_MRIci

old_stroke_mrici = tsr_6.loc[:, "old_stroke_mrici"]
old_stroke_mrici = old_stroke_mrici.fillna(old_stroke_mrici.mode()[0])
# print(old_stroke_mrici)
print(old_stroke_mrici.value_counts() / len(old_stroke_mrici))
# print(old_stroke_mrici.describe())

sns.countplot(old_stroke_mrici)
plt.title("Old_stroke_mrici - Barplot")
plt.xlabel('Old_stroke_mrici')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Cortical ACA_MRI_Left

cortical_aca_mril = tsr_6.loc[:, "cortical_aca_mril"]
cortical_aca_mril[cortical_aca_mril == '0'] = 'N'
cortical_aca_mril = cortical_aca_mril.fillna(cortical_aca_mril.mode()[0])
# print(cortical_aca_mril)
print(cortical_aca_mril.value_counts() / len(cortical_aca_mril))
# print(cortical_aca_mril.describe()))

sns.countplot(cortical_aca_mril)
plt.title("Cortical_ACA_mril - Barplot")
plt.xlabel('Cortical_ACA_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Cortical MCA_MRI_Left

cortical_mca_mril = tsr_6.loc[:, "cortical_mca_mril"]
cortical_mca_mril = cortical_mca_mril.fillna(cortical_mca_mril.mode()[0])
# print(cortical_mca_mril)
print(cortical_mca_mril.value_counts() / len(cortical_mca_mril))
# print(cortical_mca_mril.describe())

sns.countplot(cortical_mca_mril)
plt.title("Cortical_MCA_mril - Barplot")
plt.xlabel('Cortical_MCA_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Subcortical ACA_MRI_Left

subcortical_aca_mril = tsr_6.loc[:, "subcortical_aca_mril"]
subcortical_aca_mril = subcortical_aca_mril.fillna(subcortical_aca_mril.mode()[0])
# print(subcortical_aca_mril)
print(subcortical_aca_mril.value_counts() / len(subcortical_aca_mril))
# print(subcortical_aca_mril.describe())

sns.countplot(subcortical_aca_mril)
plt.title("Subcortical_ACA_mril - Barplot")
plt.xlabel('Subcortical_ACA_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Ant.Circulation: Subcortical MCA_MRI_Left

subcortical_mca_mril = tsr_6.loc[:, "subcortical_mca_mril"]
subcortical_mca_mril = subcortical_mca_mril.fillna(subcortical_mca_mril.mode()[0])
# print(subcortical_mca_mril)
print(subcortical_mca_mril.value_counts() / len(subcortical_mca_mril))
# print(subcortical_mca_mril.describe())

sns.countplot(subcortical_mca_mril)
plt.title("Subcortical_MCA_mril - Barplot")
plt.xlabel('Subcortical_MCA_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation PCA_cortex_MRI_Left

pca_cortex_mril = tsr_6.loc[:, "pca_cortex_mril"]
pca_cortex_mril = pca_cortex_mril.fillna(pca_cortex_mril.mode()[0])
# print(pca_cortex_mril)
print(pca_cortex_mril.value_counts() / len(pca_cortex_mril))
# print(pca_cortex_mril.describe())

sns.countplot(pca_cortex_mril)
plt.title("PCA_cortex_mril - Barplot")
plt.xlabel('PCA_cortex_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Thalamus_MRI_Left

thalamus_mril = tsr_6.loc[:, "thalamus_mril"]
thalamus_mril[(thalamus_mril != "N") & (thalamus_mril != "Y")] = np.nan
thalamus_mril = thalamus_mril.fillna(thalamus_mril.mode()[0])
# print(thalamus_mril)
print(thalamus_mril.value_counts() / len(thalamus_mril))
# print(thalamus_mril.describe())

sns.countplot(thalamus_mril)
plt.title("Thalamus_mril - Barplot")
plt.xlabel('Thalamus_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Brainstem_MRI_Left

brainstem_mril = tsr_6.loc[:, "brainstem_mril"]
brainstem_mril[(brainstem_mril != "N") & (brainstem_mril != "Y")] = np.nan
brainstem_mril = brainstem_mril.fillna(brainstem_mril.mode()[0])
# print(brainstem_mril)
print(brainstem_mril.value_counts() / len(brainstem_mril))
# print(brainstem_mril.describe())

sns.countplot(brainstem_mril)
plt.title("Brainstem_mril - Barplot")
plt.xlabel('Brainstem_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Posterior Circulation Cerebellum_MRI_Left

cerebellum_mril = tsr_6.loc[:, "cerebellum_mril"]
cerebellum_mril[(cerebellum_mril != "N") & (cerebellum_mril != "Y")] = np.nan
cerebellum_mril = cerebellum_mril.fillna(cerebellum_mril.mode()[0])
# print(cerebellum_mril)
print(cerebellum_mril.value_counts() / len(cerebellum_mril))
# print(cerebellum_mril.describe())

sns.countplot(cerebellum_mril)
plt.title("Cerebellum_mril - Barplot")
plt.xlabel('Cerebellum_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Watershed_MRI_Left

watershed_mril = tsr_6.loc[:, "watershed_mril"]
watershed_mril[(watershed_mril != "N") & (watershed_mril != "Y")] = np.nan
watershed_mril = watershed_mril.fillna(watershed_mril.mode()[0])
# print(watershed_mril)
print(watershed_mril.value_counts() / len(watershed_mril))
# print(watershed_mril.describe())

sns.countplot(watershed_mril)
plt.title("Watershed_mril - Barplot")
plt.xlabel('Watershed_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Hemorrhagic_infarct_MRI_Left

hemorrhagic_infarct_mril = tsr_6.loc[:, "hemorrhagic_infarct_mril"]
hemorrhagic_infarct_mril[(hemorrhagic_infarct_mril != "N") & (hemorrhagic_infarct_mril != "Y")] = np.nan
hemorrhagic_infarct_mril = hemorrhagic_infarct_mril.fillna(hemorrhagic_infarct_mril.mode()[0])
# print(hemorrhagic_infarct_mril)
print(hemorrhagic_infarct_mril.value_counts() / len(hemorrhagic_infarct_mril))
# print(hemorrhagic_infarct_mril.describe())

sns.countplot(hemorrhagic_infarct_mril)
plt.title("Hemorrhagic_infarct_mril - Barplot")
plt.xlabel('Hemorrhagic_infarct_mril')
plt.ylabel('Number', rotation=0)
plt.show()

# Old_stroke_MRIch

old_stroke_mrich = tsr_6.loc[:, "old_stroke_mrich"]
old_stroke_mrich[(old_stroke_mrich != "N") & (old_stroke_mrich != "Y")] = np.nan
old_stroke_mrich = old_stroke_mrich.fillna(old_stroke_mrich.mode()[0])
# print(old_stroke_mrich)
print(old_stroke_mrich.value_counts() / len(old_stroke_mrich))
# print(old_stroke_mrich.describe())

sns.countplot(old_stroke_mrich)
plt.title("Old_stroke_mrich - Barplot")
plt.xlabel('Old_stroke_mrich')
plt.ylabel('Number', rotation=0)
plt.show()

# Risk Factors

# Heart Disease

hd_id = tsr_6.loc[:, "hd_id"]
hd_id[(hd_id != 0) & (hd_id != 1) & (hd_id != 2)] = np.nan
hd_id = hd_id.fillna(hd_id.mode()[0])
# print(hd_id)
print(hd_id.value_counts() / len(hd_id))
# print(hd_id.describe())

hd_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(hd_id).set_xticklabels(hd_id_labels)
plt.title("Heart Disease - Barplot")
plt.xlabel('Heart Disease')
plt.ylabel('Number', rotation=0)
plt.show()

# Previous CVA

pcva_id = tsr_6.loc[:, "pcva_id"]
pcva_id = pd.to_numeric(pcva_id, errors = 'coerce')
pcva_id[(pcva_id != 0) & (pcva_id != 1) & (pcva_id != 2)] = np.nan
pcva_id = pcva_id.fillna(pcva_id.mode()[0])
# print(pcva_id)
print(pcva_id.value_counts() / len(pcva_id))
# print(pcva_id.describe())

pcva_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(pcva_id).set_xticklabels(pcva_id_labels)
plt.title("Previous CVA - Barplot")
plt.xlabel('Previous CVA')
plt.ylabel('Number', rotation=0)
plt.show()

# Previous CVA (Cerebral Infraction)

pcvaci_id = tsr_6.loc[:, "pcvaci_id"]
pcvaci_id = pcvaci_id.fillna(pcvaci_id.mode()[0])
# print(pcvaci_id)
print(pcvaci_id.value_counts() / len(pcvaci_id))
# print(pcvaci_id.describe())

pcvaci_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(pcvaci_id).set_xticklabels(pcvaci_id_labels)
plt.title("Previous CVA (Cerebral Infraction) - Barplot")
plt.xlabel('Previous CVA (Cerebral Infraction)')
plt.ylabel('Number', rotation=0)
plt.show()

# Previous CVA (Cerebral Hemorrhage)

pcvach_id = tsr_6.loc[:, "pcvach_id"]
pcvach_id = pcvach_id.fillna(pcvach_id.mode()[0])
# print(pcvach_id)
print(pcvach_id.value_counts() / len(pcvach_id))
# print(pcvach_id.describe())

pcvach_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(pcvach_id).set_xticklabels(pcvach_id_labels)
plt.title("Previous CVA (Cerebral Hemorrhage) - Barplot")
plt.xlabel('Previous CVA (Cerebral Hemorrhage)')
plt.ylabel('Number', rotation=0)
plt.show()

# Polycythemia

po_id = tsr_6.loc[:, "po_id"]
po_id = po_id.fillna(po_id.mode()[0])
# print(po_id)
print(po_id.value_counts() / len(po_id))
# print(po_id.describe())

po_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(po_id).set_xticklabels(po_id_labels)
plt.title("Polycythemia - Barplot")
plt.xlabel('Polycythemia')
plt.ylabel('Number', rotation=0)
plt.show()

# Uremia

ur_id = tsr_6.loc[:, "ur_id"]
ur_id[(ur_id != 0) & (ur_id != 1) & (ur_id != 2)] = np.nan
ur_id = ur_id.fillna(ur_id.mode()[0])
# print(ur_id)
print(ur_id.value_counts() / len(ur_id))
# print(ur_id.describe())

ur_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(ur_id).set_xticklabels(ur_id_labels)
plt.title("Uremia - Barplot")
plt.xlabel('Uremia')
plt.ylabel('Number', rotation=0)
plt.show()

# Smoking

sm_id = tsr_6.loc[:, "sm_id"]
sm_id[(sm_id != 0) & (sm_id != 1) & (sm_id != 2)] = np.nan
sm_id = sm_id.fillna(sm_id.mode()[0])
# print(sm_id)
print(sm_id.value_counts() / len(sm_id))
# print(sm_id.describe())

sm_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(sm_id).set_xticklabels(sm_id_labels)
plt.title("Smoking - Barplot")
plt.xlabel('Smoking')
plt.ylabel('Number', rotation=0)
plt.show()

# Smoking (支/天)

smc_nm = tsr_6.loc[:, "smc_nm"]

q1 = smc_nm.quantile(0.25)
q3 = smc_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
smc_nm[(smc_nm < inner_fence_low) | (smc_nm > inner_fence_upp)] = np.nan

smc_nm = smc_nm.fillna(round(smc_nm.mean(), 3))

# print(smc_nm)
# print(smc_nm.value_counts() / len(smc_nm))
print(smc_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

smc_nm.plot.box(ax=ax1)
ax1.set_title("Smoking (cigarette/ per day) - Boxplot")
ax1.set_xlabel('Smoking')
ax1.set_ylabel('cigarette/ per day', rotation=0)
ax1.set_xticks([])

# smc_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
smc_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("Smoking (cigarette/ per day) - Histogram")
ax2.set_xlabel('Smoking(cigarette/ per day)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Smoking (年)

smy_nm = tsr_6.loc[:, "smy_nm"]

q1 = smy_nm.quantile(0.25)
q3 = smy_nm.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
smy_nm[(smy_nm < inner_fence_low) | (smy_nm > inner_fence_upp)] = np.nan

smy_nm = smy_nm.fillna(round(smy_nm.mean(), 3))

# print(smy_nm)
# print(smy_nm.value_counts() / len(smy_nm))
print(smy_nm.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

smy_nm.plot.box(ax=ax1)
ax1.set_title("Smoking (year) - Boxplot")
ax1.set_xlabel('Smoking')
ax1.set_ylabel('year', rotation=0)
ax1.set_xticks([])

# smy_nm.plot.hist(ax = ax2, bins=100)
# plt.show()
smy_nm.plot.hist(ax=ax2, bins=100)
ax2.set_title("Smoking (year) - Histogram")
ax2.set_xlabel('Smoking(year)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Smoking Status

smcp_id = tsr_6.loc[:, "smcp_id"]
smcp_id[(smcp_id != 1) & (smcp_id != 2)] = np.nan
smcp_id = smcp_id.fillna(smcp_id.mode()[0])
# print(smcp_id)
print(smcp_id.value_counts() / len(smcp_id))
# print(smcp_id.describe())

smcp_id_labels = ["Current", "Past (more than 2 years)"]
sns.countplot(smcp_id).set_xticklabels(smcp_id_labels)
plt.title("Smoking Status - Barplot")
plt.xlabel('Smoking Status')
plt.ylabel('Number', rotation=0)
plt.show()

# Previous TIA

ptia_id = tsr_6.loc[:, "ptia_id"]
ptia_id[(ptia_id != 0) & (ptia_id != 1) & (ptia_id != 2)] = np.nan
ptia_id = ptia_id.fillna(ptia_id.mode()[0])
# print(ptia_id)
print(ptia_id.value_counts() / len(ptia_id))
# print(ptia_id.describe())

ptia_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(ptia_id).set_xticklabels(ptia_id_labels)
plt.title("Smoking Status - Barplot")
plt.xlabel('Smoking Status')
plt.ylabel('Number', rotation=0)
plt.show()

# Dyslipidemia

hc_id = tsr_6.loc[:, "hc_id"]
hc_id[(hc_id != 0) & (hc_id != 1) & (hc_id != 2)] = np.nan
hc_id = hc_id.fillna(hc_id.mode()[0])
# print(hc_id)
print(hc_id.value_counts() / len(hc_id))
# print(hc_id.describe())

hc_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(hc_id).set_xticklabels(hc_id_labels)
plt.title("Dyslipidemia - Barplot")
plt.xlabel('Dyslipidemia')
plt.ylabel('Number', rotation=0)
plt.show()

# Dyslipidemia (Hypertriglyceridemia)

hcht_id = tsr_6.loc[:, "hcht_id"]
hcht_id = hcht_id.fillna(hcht_id.mode()[0])
# print(hcht_id)
print(hcht_id.value_counts() / len(hcht_id))
# print(hcht_id.describe())

hcht_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(hcht_id).set_xticklabels(hcht_id_labels)
plt.title("Dyslipidemia (Hypertriglyceridemia) - Barplot")
plt.xlabel('Dyslipidemia (Hypertriglyceridemia)')
plt.ylabel('Number', rotation=0)
plt.show()

# Dyslipidemia (Hypercholesterolemia)

hchc_id = tsr_6.loc[:, "hchc_id"]
hchc_id = hchc_id.fillna(hchc_id.mode()[0])
# print(hchc_id)
print(hchc_id.value_counts() / len(hchc_id))
# print(hchc_id.describe())

hchc_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(hchc_id).set_xticklabels(hchc_id_labels)
plt.title("Dyslipidemia (Hypercholesterolemia) - Barplot")
plt.xlabel('Dyslipidemia (Hypercholesterolemia)')
plt.ylabel('Number', rotation=0)
plt.show()

# Hypertension

ht_id = tsr_6.loc[:, "ht_id"]
ht_id[(ht_id != 0) & (ht_id != 1) & (ht_id != 2)] = np.nan
ht_id = ht_id.fillna(ht_id.mode()[0])
# print(ht_id)
print(ht_id.value_counts() / len(ht_id))
# print(ht_id.describe())

ht_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(ht_id).set_xticklabels(ht_id_labels)
plt.title("Hypertension - Barplot")
plt.xlabel('Hypertension')
plt.ylabel('Number', rotation=0)
plt.show()

# DM

dm_id = tsr_6.loc[:, "dm_id"]
dm_id[(dm_id != 0) & (dm_id != 1) & (dm_id != 2)] = np.nan
dm_id = dm_id.fillna(dm_id.mode()[0])
# print(dm_id)
print(dm_id.value_counts() / len(dm_id))
# print(dm_id.describe())

dm_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(dm_id).set_xticklabels(dm_id_labels)
plt.title("DM - Barplot")
plt.xlabel('DM')
plt.ylabel('Number', rotation=0)
plt.show()

# PAD

pad_id = tsr_6.loc[:, "pad_id"]
pad_id = pad_id.fillna(pad_id.mode()[0])
# print(pad_id)
print(pad_id.value_counts() / len(pad_id))
# print(pad_id.describe())

pad_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(pad_id).set_xticklabels(pad_id_labels)
plt.title("PAD - Barplot")
plt.xlabel('PAD')
plt.ylabel('Number', rotation=0)
plt.show()

# Alcohol

al_id = tsr_6.loc[:, "al_id"]
al_id = pd.to_numeric(al_id, errors='coerce')
al_id[(al_id != 0) & (al_id != 1) & (al_id != 2)] = np.nan
# al_id = al_id.fillna(al_id.mode()[0])
# print(al_id)
print(al_id.value_counts() / len(al_id))
# print(al_id.describe())

al_id_labels = ["No", "Yes", "Unknown"]
sns.countplot(al_id).set_xticklabels(al_id_labels)
plt.title("Alcohol - Barplot")
plt.xlabel('Alcohol')
plt.ylabel('Number', rotation=0)
plt.show()
al_id = al_id.fillna(999)

# Cancer

ca_id = tsr_6.loc[:, "ca_id"]
ca_id = pd.to_numeric(ca_id, errors='coerce')
ca_id[(ca_id != 0) & (ca_id != 1) & (ca_id != 2)] = np.nan
# ca_id = ca_id.fillna(ca_id.mode()[0])
# print(ca_id)
print(ca_id.value_counts() / len(ca_id))
# print(ca_id.describe())

# ca_id_labels = ["No","Yes", "Unknown"]
ca_id_labels = ["No", "Yes"]
sns.countplot(ca_id).set_xticklabels(ca_id_labels)
plt.title("Cancer - Barplot")
plt.xlabel('Cancer')
plt.ylabel('Number', rotation=0)
plt.show()
ca_id = ca_id.fillna(999)

# CA_TX

# Others

ot_id = tsr_6.loc[:, "ot_id"]
ot_id = pd.to_numeric(ot_id, errors='coerce')
ot_id[(ot_id != 0) & (ot_id != 1) & (ot_id != 2)] = np.nan
ot_id = ot_id.fillna(ot_id.mode()[0])
# print(ot_id)
print(ot_id.value_counts() / len(ot_id))
# print(ot_id.describe())

ot_id_labels = ["No", "Yes"]
sns.countplot(ot_id).set_xticklabels(ot_id_labels)
plt.title("Others - Barplot")
plt.xlabel('Others')
plt.ylabel('Number', rotation=0)
plt.show()

# OT_TX

# Hypertension Was Diagnosed at This Visit

thishy_id = tsr_6.loc[:, "thishy_id"]
thishy_id[thishy_id == "1"] = "Y"
thishy_id[(thishy_id != "N") & (thishy_id != "Y")] = np.nan
thishy_id = thishy_id.fillna(thishy_id.mode()[0])
# print(thishy_id)
print(thishy_id.value_counts() / len(thishy_id))
# print(thishy_id.describe())

sns.countplot(thishy_id)
plt.title("Hypertension Was Diagnosed at This Visit - Barplot")
plt.xlabel('Hypertension Was Diagnosed at This Visit')
plt.ylabel('Number', rotation=0)
plt.show()

# DM Was Diagnosed at This Visit

thisdi_id = tsr_6.loc[:, "thisdi_id"]
thisdi_id[thisdi_id == "1"] = 1
thisdi_id[thisdi_id == "0"] = 0
thisdi_id[thisdi_id == 1] = "Y"
thisdi_id[thisdi_id == 0] = "N"
thisdi_id[(thisdi_id != "N") & (thisdi_id != "Y")] = np.nan
thisdi_id = thisdi_id.fillna(thisdi_id.mode()[0])
# print(thisdi_id)
print(thisdi_id.value_counts() / len(thisdi_id))
# print(thisdi_id.describe())

sns.countplot(thisdi_id)
plt.title("DM Was Diagnosed at This Visit - Barplot")
plt.xlabel('DM Was Diagnosed at This Visit')
plt.ylabel('Number', rotation=0)
plt.show()

# FAHIID_PARENTS_1

fahiid_parents_1 = tsr_6.loc[:, "fahiid_parents_1"]
fahiid_parents_1 = fahiid_parents_1.fillna(fahiid_parents_1.mode()[0])
# print(fahiid_parents_1)
print(fahiid_parents_1.value_counts() / len(fahiid_parents_1))
# print(fahiid_parents_1.describe())

fahiid_parents_1_labels = ["沒有", "有", "不知道"]
sns.countplot(fahiid_parents_1).set_xticklabels(fahiid_parents_1_labels)
plt.title("FAHIID_PARENTS_1 - Barplot")
plt.xlabel('FAHIID_PARENTS_1')
plt.ylabel('Number', rotation=0)
plt.show()

# FAHIID_PARENTS_2

fahiid_parents_2 = tsr_6.loc[:, "fahiid_parents_2"]
fahiid_parents_2 = fahiid_parents_2.fillna(fahiid_parents_2.mode()[0])
# print(fahiid_parents_2)
print(fahiid_parents_2.value_counts() / len(fahiid_parents_2))
# print(fahiid_parents_2.describe())

fahiid_parents_2_labels = ["沒有", "有", "不知道"]
sns.countplot(fahiid_parents_2).set_xticklabels(fahiid_parents_2_labels)
plt.title("FAHIID_PARENTS_2 - Barplot")
plt.xlabel('FAHIID_PARENTS_2')
plt.ylabel('Number', rotation=0)
plt.show()

# FAHIID_PARENTS_3

fahiid_parents_3 = tsr_6.loc[:, "fahiid_parents_3"]
fahiid_parents_3[(fahiid_parents_3 != 0) & (fahiid_parents_3 != 1) & (fahiid_parents_3 != 2)] = np.nan
fahiid_parents_3 = fahiid_parents_3.fillna(fahiid_parents_3.mode()[0])
# print(fahiid_parents_3)
print(fahiid_parents_3.value_counts() / len(fahiid_parents_3))
# print(fahiid_parents_3.describe())

fahiid_parents_3_labels = ["沒有", "有", "不知道"]
sns.countplot(fahiid_parents_3).set_xticklabels(fahiid_parents_3_labels)
plt.title("FAHIID_PARENTS_3 - Barplot")
plt.xlabel('FAHIID_PARENTS_3')
plt.ylabel('Number', rotation=0)
plt.show()

# FAHIID_PARENTS_4

fahiid_parents_4 = tsr_6.loc[:, "fahiid_parents_4"]
fahiid_parents_4[(fahiid_parents_4 != 0) & (fahiid_parents_4 != 1) & (fahiid_parents_4 != 2)] = np.nan
fahiid_parents_4 = fahiid_parents_4.fillna(fahiid_parents_4.mode()[0])
# print(fahiid_parents_4)
print(fahiid_parents_4.value_counts() / len(fahiid_parents_4))
# print(fahiid_parents_4.describe())

fahiid_parents_4_labels = ["沒有", "有", "不知道"]
sns.countplot(fahiid_parents_4).set_xticklabels(fahiid_parents_4_labels)
plt.title("FAHIID_PARENTS_4 - Barplot")
plt.xlabel('FAHIID_PARENTS_4')
plt.ylabel('Number', rotation=0)
plt.show()

# FAHIID_BRSI_1

fahiid_brsi_1 = tsr_6.loc[:, "fahiid_brsi_1"]
fahiid_brsi_1 = pd.to_numeric(fahiid_brsi_1, errors = "coerce")
fahiid_brsi_1[(fahiid_brsi_1 != 0) & (fahiid_brsi_1 != 1) & (fahiid_brsi_1 != 2)] = np.nan
fahiid_brsi_1 = fahiid_brsi_1.fillna(fahiid_brsi_1.mode()[0])
# print(fahiid_brsi_1)
print(fahiid_brsi_1.value_counts() / len(fahiid_brsi_1))
# print(fahiid_brsi_1.describe())

fahiid_brsi_1_labels = ["沒有", "有", "不知道"]
sns.countplot(fahiid_brsi_1).set_xticklabels(fahiid_brsi_1_labels)
plt.title("FAHIID_BRSI_1 - Barplot")
plt.xlabel('FAHIID_BRSI_1')
plt.ylabel('Number', rotation=0)
plt.show()

# FAHIID_BRSI_2

fahiid_brsi_2 = tsr_6.loc[:, "fahiid_brsi_2"]
fahiid_brsi_2 = pd.to_numeric(fahiid_brsi_2, errors = "coerce")
fahiid_brsi_2[(fahiid_brsi_2 != 0) & (fahiid_brsi_2 != 1) & (fahiid_brsi_2 != 2)] = np.nan
fahiid_brsi_2 = fahiid_brsi_2.fillna(fahiid_brsi_2.mode()[0])
# print(fahiid_brsi_2)
print(fahiid_brsi_2.value_counts() / len(fahiid_brsi_2))
# print(fahiid_brsi_2.describe())

fahiid_brsi_2_labels = ["沒有", "有", "不知道"]
sns.countplot(fahiid_brsi_2).set_xticklabels(fahiid_brsi_2_labels)
plt.title("FAHIID_BRSI_2 - Barplot")
plt.xlabel('FAHIID_BRSI_2')
plt.ylabel('Number', rotation=0)
plt.show()

# FAHIID_BRSI_3

fahiid_brsi_3 = tsr_6.loc[:, "fahiid_brsi_3"]
fahiid_brsi_3[(fahiid_brsi_3 != '0') & (fahiid_brsi_3 != '1') & (fahiid_brsi_3 != '2')] = np.nan
fahiid_brsi_3 = fahiid_brsi_3.fillna(fahiid_brsi_3.mode()[0])
# print(fahiid_brsi_3)
print(fahiid_brsi_3.value_counts() / len(fahiid_brsi_3))
# print(fahiid_brsi_3.describe())

fahiid_brsi_3_labels = ["沒有", "有", "不知道"]
sns.countplot(fahiid_brsi_3).set_xticklabels(fahiid_brsi_3_labels)
plt.title("FAHIID_BRSI_3 - Barplot")
plt.xlabel('FAHIID_BRSI_3')
plt.ylabel('Number', rotation=0)
plt.show()

# FAHIID_BRSI_4

fahiid_brsi_4 = tsr_6.loc[:, "fahiid_brsi_4"]
fahiid_brsi_4[(fahiid_brsi_4 != '0') & (fahiid_brsi_4 != '1') & (fahiid_brsi_4 != '2')] = np.nan
fahiid_brsi_4 = fahiid_brsi_4.fillna(fahiid_brsi_4.mode()[0])
# print(fahiid_brsi_4)
print(fahiid_brsi_4.value_counts() / len(fahiid_brsi_4))
# print(fahiid_brsi_4.describe())

fahiid_brsi_4_labels = ["沒有", "有", "不知道"]
sns.countplot(fahiid_brsi_4).set_xticklabels(fahiid_brsi_4_labels)
plt.title("FAHIID_BRSI_4 - Barplot")
plt.xlabel('FAHIID_BRSI_4')
plt.ylabel('Number', rotation=0)
plt.show()

# NIHS_1a_in

nihs_1a_in = tsr_6.loc[:, "nihs_1a_in"]
nihs_1a_in[(nihs_1a_in < 0) | (nihs_1a_in > 3)] = np.nan
nihs_1a_in = nihs_1a_in.fillna(nihs_1a_in.mode()[0])
# print(nihs_1a_in)
print(nihs_1a_in.value_counts() / len(nihs_1a_in))
# print(nihs_1a_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_1a_in.plot.box(ax=ax1)
ax1.set_title("NIHS_1a_in - Boxplot")
ax1.set_xlabel('NIHS_1a_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_1a_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_1a_in - Histogram")
ax2.set_xlabel('NIHS_1a_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_1b_in

nihs_1b_in = tsr_6.loc[:, "nihs_1b_in"]
nihs_1b_in[(nihs_1b_in < 0) | (nihs_1b_in > 2)] = np.nan
nihs_1b_in = nihs_1b_in.fillna(nihs_1b_in.mode()[0])
# print(nihs_1b_in)
print(nihs_1b_in.value_counts() / len(nihs_1b_in))
# print(nihs_1b_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_1b_in.plot.box(ax=ax1)
ax1.set_title("NIHS_1b_in - Boxplot")
ax1.set_xlabel('NIHS_1b_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_1b_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_1b_in - Histogram")
ax2.set_xlabel('NIHS_1b_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_1c_in

nihs_1c_in = tsr_6.loc[:, "nihs_1c_in"]
nihs_1c_in[(nihs_1c_in < 0) | (nihs_1c_in > 2)] = np.nan
nihs_1c_in = nihs_1c_in.fillna(nihs_1c_in.mode()[0])
# print(nihs_1c_in)
print(nihs_1c_in.value_counts() / len(nihs_1c_in))
# print(nihs_1c_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_1c_in.plot.box(ax=ax1)
ax1.set_title("NIHS_1c_in - Boxplot")
ax1.set_xlabel('NIHS_1c_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_1c_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_1c_in - Histogram")
ax2.set_xlabel('NIHS_1c_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_2_in

nihs_2_in = tsr_6.loc[:, "nihs_2_in"]
nihs_2_in[(nihs_2_in < 0) | (nihs_2_in > 2)] = np.nan
nihs_2_in = nihs_2_in.fillna(nihs_2_in.mode()[0])
# print(nihs_2_in)
print(nihs_2_in.value_counts() / len(nihs_2_in))
# print(nihs_2_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_2_in.plot.box(ax=ax1)
ax1.set_title("NIHS_2_in - Boxplot")
ax1.set_xlabel('NIHS_2_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_2_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_2_in - Histogram")
ax2.set_xlabel('NIHS_2_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_3_in

nihs_3_in = tsr_6.loc[:, "nihs_3_in"]
nihs_3_in[(nihs_3_in < 0) | (nihs_3_in > 3)] = np.nan
nihs_3_in = nihs_3_in.fillna(nihs_3_in.mode()[0])
# print(nihs_3_in)
print(nihs_3_in.value_counts() / len(nihs_3_in))
# print(nihs_3_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_3_in.plot.box(ax=ax1)
ax1.set_title("NIHS_3_in - Boxplot")
ax1.set_xlabel('NIHS_3_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_3_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_3_in - Histogram")
ax2.set_xlabel('NIHS_3_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_4_in

nihs_4_in = tsr_6.loc[:, "nihs_4_in"]
nihs_4_in[(nihs_4_in < 0) | (nihs_4_in > 3)] = np.nan
nihs_4_in = nihs_4_in.fillna(nihs_4_in.mode()[0])
# print(nihs_4_in)
print(nihs_4_in.value_counts() / len(nihs_4_in))
# print(nihs_4_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_4_in.plot.box(ax=ax1)
ax1.set_title("NIHS_4_in - Boxplot")
ax1.set_xlabel('NIHS_4_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_4_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_4_in - Histogram")
ax2.set_xlabel('NIHS_4_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_5aL_in

nihs_5al_in = tsr_6.loc[:, "nihs_5al_in"]
nihs_5al_in[(nihs_5al_in < 0) | (nihs_5al_in > 4)] = np.nan
nihs_5al_in = nihs_5al_in.fillna(nihs_5al_in.mode()[0])
# print(nihs_5al_in)
print(nihs_5al_in.value_counts() / len(nihs_5al_in))
# print(nihs_5al_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_5al_in.plot.box(ax=ax1)
ax1.set_title("NIHS_5aL_in - Boxplot")
ax1.set_xlabel('NIHS_5aL_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_5al_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_5aL_in - Histogram")
ax2.set_xlabel('NIHS_5aL_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_5bR_in

nihs_5br_in = tsr_6.loc[:, "nihs_5br_in"]
nihs_5br_in[(nihs_5br_in < 0) | (nihs_5br_in > 4)] = np.nan
nihs_5br_in = nihs_5br_in.fillna(nihs_5br_in.mode()[0])
# print(nihs_5br_in)
print(nihs_5br_in.value_counts() / len(nihs_5br_in))
# print(nihs_5br_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_5br_in.plot.box(ax=ax1)
ax1.set_title("NIHS_5bR_in - Boxplot")
ax1.set_xlabel('NIHS_5bR_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_5br_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_5bR_in - Histogram")
ax2.set_xlabel('NIHS_5bR_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_6aL_in

nihs_6al_in = tsr_6.loc[:, "nihs_6al_in"]
nihs_6al_in[(nihs_6al_in < 0) | (nihs_6al_in > 4)] = np.nan
nihs_6al_in = nihs_6al_in.fillna(nihs_6al_in.mode()[0])
# print(nihs_6al_in)
print(nihs_6al_in.value_counts() / len(nihs_6al_in))
# print(nihs_6al_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_6al_in.plot.box(ax=ax1)
ax1.set_title("NIHS_6aL_in - Boxplot")
ax1.set_xlabel('NIHS_6aL_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_6al_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_6aL_in - Histogram")
ax2.set_xlabel('NIHS_6aL_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_6bR_in

nihs_6br_in = tsr_6.loc[:, "nihs_6br_in"]
nihs_6br_in[(nihs_6br_in < 0) | (nihs_6br_in > 4)] = np.nan
nihs_6br_in = nihs_6br_in.fillna(nihs_6br_in.mode()[0])
# print(nihs_6br_in)
print(nihs_6br_in.value_counts() / len(nihs_6br_in))
# print(nihs_6br_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_6br_in.plot.box(ax=ax1)
ax1.set_title("NIHS_6bR_in - Boxplot")
ax1.set_xlabel('NIHS_6bR_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_6br_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_6bR_in - Histogram")
ax2.set_xlabel('NIHS_6bR_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_7_in

nihs_7_in = tsr_6.loc[:, "nihs_7_in"]
nihs_7_in[(nihs_7_in < 0) | (nihs_7_in > 2)] = np.nan
nihs_7_in = nihs_7_in.fillna(nihs_7_in.mode()[0])
# print(nihs_7_in)
print(nihs_7_in.value_counts() / len(nihs_7_in))
# print(nihs_7_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_7_in.plot.box(ax=ax1)
ax1.set_title("NIHS_7_in - Boxplot")
ax1.set_xlabel('NIHS_7_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_7_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_7_in - Histogram")
ax2.set_xlabel('NIHS_7_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_8_in

nihs_8_in = tsr_6.loc[:, "nihs_8_in"]
nihs_8_in[(nihs_8_in < 0) | (nihs_8_in > 2)] = np.nan
nihs_8_in = nihs_8_in.fillna(nihs_8_in.mode()[0])
# print(nihs_8_in)
print(nihs_8_in.value_counts() / len(nihs_8_in))
# print(nihs_8_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_8_in.plot.box(ax=ax1)
ax1.set_title("NIHS_8_in - Boxplot")
ax1.set_xlabel('NIHS_8_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_8_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_8_in - Histogram")
ax2.set_xlabel('NIHS_8_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_9_in

nihs_9_in = tsr_6.loc[:, "nihs_9_in"]
nihs_9_in[(nihs_9_in < 0) | (nihs_9_in > 3)] = np.nan
nihs_9_in = nihs_9_in.fillna(nihs_9_in.mode()[0])
# print(nihs_9_in)
print(nihs_9_in.value_counts() / len(nihs_9_in))
# print(nihs_9_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_9_in.plot.box(ax=ax1)
ax1.set_title("NIHS_9_in - Boxplot")
ax1.set_xlabel('NIHS_9_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_9_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_9_in - Histogram")
ax2.set_xlabel('NIHS_9_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_10_in

nihs_10_in = tsr_6.loc[:, "nihs_10_in"]
nihs_10_in[(nihs_10_in < 0) | (nihs_10_in > 2)] = np.nan
nihs_10_in = nihs_10_in.fillna(nihs_10_in.mode()[0])
# print(nihs_10_in)
print(nihs_10_in.value_counts() / len(nihs_10_in))
# print(nihs_10_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_10_in.plot.box(ax=ax1)
ax1.set_title("NIHS_10_in - Boxplot")
ax1.set_xlabel('NIHS_10_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_10_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_10_in - Histogram")
ax2.set_xlabel('NIHS_10_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_11_in

nihs_11_in = tsr_6.loc[:, "nihs_11_in"]
nihs_11_in[(nihs_11_in < 0) | (nihs_11_in > 2)] = np.nan
nihs_11_in = nihs_11_in.fillna(nihs_11_in.mode()[0])
# print(nihs_11_in)
print(nihs_11_in.value_counts() / len(nihs_11_in))
# print(nihs_11_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_11_in.plot.box(ax=ax1)
ax1.set_title("NIHS_11_in - Boxplot")
ax1.set_xlabel('NIHS_11_in')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_11_in.plot.hist(ax=ax2)
ax2.set_title("NIHS_11_in - Histogram")
ax2.set_xlabel('NIHS_11_in(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Total_in

total_in = nihs_1a_in + nihs_1b_in + nihs_1c_in + nihs_2_in + nihs_3_in + nihs_4_in + nihs_5al_in + nihs_5br_in + nihs_6al_in + nihs_6br_in + nihs_7_in + nihs_8_in + nihs_9_in + nihs_10_in + nihs_11_in
# print(total_in)
# print(total_in.value_counts() / len(total_in))
print(total_in.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

total_in.plot.box(ax=ax1)
ax1.set_title("NIHSS Score IN - Boxplot")
ax1.set_xlabel('NIHSS Score IN')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

total_in.plot.hist(ax=ax2)
ax2.set_title("NIHSS Score IN - Histogram")
ax2.set_xlabel('NIHSS Score IN(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_1a_out

nihs_1a_out = tsr_6.loc[:, "nihs_1a_out"]
nihs_1a_out[(nihs_1a_out < 0) | (nihs_1a_out > 3)] = np.nan
nihs_1a_out = nihs_1a_out.fillna(nihs_1a_out.mode()[0])
# print(nihs_1a_out)
print(nihs_1a_out.value_counts() / len(nihs_1a_out))
# print(nihs_1a_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_1a_out.plot.box(ax=ax1)
ax1.set_title("NIHS_1a_out - Boxplot")
ax1.set_xlabel('NIHS_1a_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_1a_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_1a_out - Histogram")
ax2.set_xlabel('NIHS_1a_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_1b_out

nihs_1b_out = tsr_6.loc[:, "nihs_1b_out"]
nihs_1b_out[(nihs_1b_out < 0) | (nihs_1b_out > 2)] = np.nan
nihs_1b_out = nihs_1b_out.fillna(nihs_1b_out.mode()[0])
# print(nihs_1b_out)
print(nihs_1b_out.value_counts() / len(nihs_1b_out))
# print(nihs_1b_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_1b_out.plot.box(ax=ax1)
ax1.set_title("NIHS_1b_out - Boxplot")
ax1.set_xlabel('NIHS_1b_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_1b_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_1b_out - Histogram")
ax2.set_xlabel('NIHS_1b_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_1c_out

nihs_1c_out = tsr_6.loc[:, "nihs_1c_out"]
nihs_1c_out[(nihs_1c_out < 0) | (nihs_1c_out > 2)] = np.nan
nihs_1c_out = nihs_1c_out.fillna(nihs_1c_out.mode()[0])
# print(nihs_1c_out)
print(nihs_1c_out.value_counts() / len(nihs_1c_out))
# print(nihs_1c_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_1c_out.plot.box(ax=ax1)
ax1.set_title("NIHS_1c_out - Boxplot")
ax1.set_xlabel('NIHS_1c_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_1c_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_1c_out - Histogram")
ax2.set_xlabel('NIHS_1c_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_2_out

nihs_2_out = tsr_6.loc[:, "nihs_2_out"]
nihs_2_out[(nihs_2_out < 0) | (nihs_2_out > 2)] = np.nan
nihs_2_out = nihs_2_out.fillna(nihs_2_out.mode()[0])
# print(nihs_2_out)
print(nihs_2_out.value_counts() / len(nihs_2_out))
# print(nihs_2_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_2_out.plot.box(ax=ax1)
ax1.set_title("NIHS_2_out - Boxplot")
ax1.set_xlabel('NIHS_2_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_2_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_2_out - Histogram")
ax2.set_xlabel('NIHS_2_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_3_out

nihs_3_out = tsr_6.loc[:, "nihs_3_out"]
nihs_3_out[(nihs_3_out < 0) | (nihs_3_out > 3)] = np.nan
nihs_3_out = nihs_3_out.fillna(nihs_3_out.mode()[0])
# print(nihs_3_out)
print(nihs_3_out.value_counts() / len(nihs_3_out))
# print(nihs_3_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_3_out.plot.box(ax=ax1)
ax1.set_title("NIHS_3_out - Boxplot")
ax1.set_xlabel('NIHS_3_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_3_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_3_out - Histogram")
ax2.set_xlabel('NIHS_3_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_4_out

nihs_4_out = tsr_6.loc[:, "nihs_4_out"]
nihs_4_out[(nihs_4_out < 0) | (nihs_4_out > 3)] = np.nan
nihs_4_out = nihs_4_out.fillna(nihs_4_out.mode()[0])
# print(nihs_4_out)
print(nihs_4_out.value_counts() / len(nihs_4_out))
# print(nihs_4_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_4_out.plot.box(ax=ax1)
ax1.set_title("NIHS_4_out - Boxplot")
ax1.set_xlabel('NIHS_4_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_4_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_4_out - Histogram")
ax2.set_xlabel('NIHS_4_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_5aL_out

nihs_5al_out = tsr_6.loc[:, "nihs_5al_out"]
nihs_5al_out[(nihs_5al_out < 0) | (nihs_5al_out > 4)] = np.nan
nihs_5al_out = nihs_5al_out.fillna(nihs_5al_out.mode()[0])
# print(nihs_5al_out)
print(nihs_5al_out.value_counts() / len(nihs_5al_out))
# print(nihs_5al_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_5al_out.plot.box(ax=ax1)
ax1.set_title("NIHS_5aL_out - Boxplot")
ax1.set_xlabel('NIHS_5aL_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_5al_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_5aL_out - Histogram")
ax2.set_xlabel('NIHS_5aL_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_5bR_out

nihs_5br_out = tsr_6.loc[:, "nihs_5br_out"]
nihs_5br_out[(nihs_5br_out < 0) | (nihs_5br_out > 4)] = np.nan
nihs_5br_out = nihs_5br_out.fillna(nihs_5br_out.mode()[0])
# print(nihs_5br_out)
print(nihs_5br_out.value_counts() / len(nihs_5br_out))
# print(nihs_5br_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_5br_out.plot.box(ax=ax1)
ax1.set_title("NIHS_5bR_out - Boxplot")
ax1.set_xlabel('NIHS_5bR_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_5br_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_5bR_out - Histogram")
ax2.set_xlabel('NIHS_5bR_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_6aL_out

nihs_6al_out = tsr_6.loc[:, "nihs_6al_out"]
nihs_6al_out[(nihs_6al_out < 0) | (nihs_6al_out > 4)] = np.nan
nihs_6al_out = nihs_6al_out.fillna(nihs_6al_out.mode()[0])
# print(nihs_6al_out)
print(nihs_6al_out.value_counts() / len(nihs_6al_out))
# print(nihs_6al_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_6al_out.plot.box(ax=ax1)
ax1.set_title("NIHS_6aL_out - Boxplot")
ax1.set_xlabel('NIHS_6aL_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_6al_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_6aL_out - Histogram")
ax2.set_xlabel('NIHS_6aL_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_6bR_out

nihs_6br_out = tsr_6.loc[:, "nihs_6br_out"]
nihs_6br_out[(nihs_6br_out < 0) | (nihs_6br_out > 4)] = np.nan
nihs_6br_out = nihs_6br_out.fillna(nihs_6br_out.mode()[0])
# print(nihs_6br_out)
print(nihs_6br_out.value_counts() / len(nihs_6br_out))
# print(nihs_6br_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_6br_out.plot.box(ax=ax1)
ax1.set_title("NIHS_6bR_out - Boxplot")
ax1.set_xlabel('NIHS_6bR_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_6br_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_6bR_out - Histogram")
ax2.set_xlabel('NIHS_6bR_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_7_out

nihs_7_out = tsr_6.loc[:, "nihs_7_out"]
nihs_7_out[(nihs_7_out < 0) | (nihs_7_out > 2)] = np.nan
nihs_7_out = nihs_7_out.fillna(nihs_7_out.mode()[0])
# print(nihs_7_out)
print(nihs_7_out.value_counts() / len(nihs_7_out))
# print(nihs_7_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_7_out.plot.box(ax=ax1)
ax1.set_title("NIHS_7_out - Boxplot")
ax1.set_xlabel('NIHS_7_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_7_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_7_out - Histogram")
ax2.set_xlabel('NIHS_7_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_8_out

nihs_8_out = tsr_6.loc[:, "nihs_8_out"]
nihs_8_out[(nihs_8_out < 0) | (nihs_8_out > 2)] = np.nan
nihs_8_out = nihs_8_out.fillna(nihs_8_out.mode()[0])
# print(nihs_8_out)
print(nihs_8_out.value_counts() / len(nihs_8_out))
# print(nihs_8_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_8_out.plot.box(ax=ax1)
ax1.set_title("NIHS_8_out - Boxplot")
ax1.set_xlabel('NIHS_8_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_8_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_8_out - Histogram")
ax2.set_xlabel('NIHS_8_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_9_out

nihs_9_out = tsr_6.loc[:, "nihs_9_out"]
nihs_9_out[(nihs_9_out < 0) | (nihs_9_out > 3)] = np.nan
nihs_9_out = nihs_9_out.fillna(nihs_9_out.mode()[0])
# print(nihs_9_out)
print(nihs_9_out.value_counts() / len(nihs_9_out))
# print(nihs_9_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_9_out.plot.box(ax=ax1)
ax1.set_title("NIHS_9_out - Boxplot")
ax1.set_xlabel('NIHS_9_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_9_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_9_out - Histogram")
ax2.set_xlabel('NIHS_9_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_10_out

nihs_10_out = tsr_6.loc[:, "nihs_10_out"]
nihs_10_out[(nihs_10_out < 0) | (nihs_10_out > 2)] = np.nan
nihs_10_out = nihs_10_out.fillna(nihs_10_out.mode()[0])
# print(nihs_10_out)
print(nihs_10_out.value_counts() / len(nihs_10_out))
# print(nihs_10_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_10_out.plot.box(ax=ax1)
ax1.set_title("NIHS_10_out - Boxplot")
ax1.set_xlabel('NIHS_10_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_10_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_10_out - Histogram")
ax2.set_xlabel('NIHS_10_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# NIHS_11_out

nihs_11_out = tsr_6.loc[:, "nihs_11_out"]
nihs_11_out[(nihs_11_out < 0) | (nihs_11_out > 2)] = np.nan
nihs_11_out = nihs_11_out.fillna(nihs_11_out.mode()[0])
# print(nihs_11_out)
print(nihs_11_out.value_counts() / len(nihs_11_out))
# print(nihs_11_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

nihs_11_out.plot.box(ax=ax1)
ax1.set_title("NIHS_11_out - Boxplot")
ax1.set_xlabel('NIHS_11_out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

nihs_11_out.plot.hist(ax=ax2)
ax2.set_title("NIHS_11_out - Histogram")
ax2.set_xlabel('NIHS_11_out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# Total_out

total_out = nihs_1a_out + nihs_1b_out + nihs_1c_out + nihs_2_out + nihs_3_out + nihs_4_out + nihs_5al_out + nihs_5br_out + nihs_6al_out + nihs_6br_out + nihs_7_out + nihs_8_out + nihs_9_out + nihs_10_out + nihs_11_out
# print(total_out)
# print(total_out.value_counts() / len(total_out))
print(total_out.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

total_out.plot.box(ax=ax1)
ax1.set_title("NIHSS Score out - Boxplot")
ax1.set_xlabel('NIHSS Score out')
ax1.set_ylabel('Score', rotation=0)
ax1.set_xticks([])

total_out.plot.hist(ax=ax2)
ax2.set_title("NIHSS Score out - Histogram")
ax2.set_xlabel('NIHSS Score out(Score)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

# SEX

SexName = tsr_6.loc[:, "SexName"]
# print(SexName)
print(SexName.value_counts() / len(SexName))
# print(SexName.describe())

SexName_labels = ["Male", "Female"]
sns.countplot(SexName).set_xticklabels(SexName_labels)
plt.title("SEX - Barplot")
plt.xlabel('SEX')
plt.ylabel('Number', rotation=0)
plt.show()

# AGE

Age = tsr_6.loc[:, "Age"]

q1 = Age.quantile(0.25)
q3 = Age.quantile(0.75)
iqr = q3 - q1
inner_fence = 1.5 * iqr

inner_fence_low = q1 - inner_fence
inner_fence_upp = q3 + inner_fence
Age[(Age < inner_fence_low) | (Age > inner_fence_upp)] = np.nan

Age = Age.fillna(round(Age.mean(), 3))

# print(Age)
# print(Age.value_counts() / len(Age))
print(Age.describe())

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

Age.plot.box(ax=ax1)
ax1.set_title("AGE - Boxplot")
ax1.set_xlabel('AGE')
ax1.set_ylabel('Year', rotation=0)
ax1.set_xticks([])

Age.plot.hist(ax=ax2)
ax2.set_title("AGE - Histogram")
ax2.set_xlabel('AGE(Year)')
ax2.set_ylabel('Number', rotation=0)
plt.show()

TSR_6_CLEANED = pd.DataFrame([height_nm ,  weight_nm ,  edu_id ,  pro_id ,  opc_id ,  ih_fl ,  ivtpamg_nm , hospitalised_time, 
                          nivtpa_id, nivtpa1_fl ,  nivtpa2_fl , nivtpa3_fl ,  nivtpa4_fl ,  nivtpa5_fl ,  nivtpa6_fl , 
                          nivtpa7_fl , nivtpa8_fl ,  nivtpa9_fl ,  nivtpa10_fl , nivtpa11_fl ,  nivtpa99_fl ,  gcse_nm , 
                          gcsv_nm ,  gcsm_nm ,  sbp_nm  , dbp_nm ,  bt_nm ,  hr_nm ,  rr_nm ,  icd_id ,  icdtia_id ,  
                          toast_id , toastle_fl ,  toastli_fl ,  toastsce_fl ,  toastsmo_fl ,  toastsra_fl ,  toastsdi_fl , 
                          toastsmi_fl ,  toastsantip_fl ,  toastsau_fl ,  toastshy_fl ,  toastspr_fl , toastsantit_fl , 
                          toastsho_fl ,  toastshys_fl ,  toastsca_fl ,  toastso_fl ,  toastu_id ,  cich_id ,  csah_id ,
                          thd_id ,  thda_fl ,  thdh_fl ,  thdi_fl ,  thdam_fl ,  thdv_fl ,  thde_fl , thdm_fl ,
                          thdr_fl ,  thdp_fl ,  thdoo_fl ,  hb_nm ,  hct_nm ,  platelet_nm ,  wbc_nm ,  ptt1_nm , ptt2_nm ,
                          ptinr_nm ,  er_nm ,  bun_nm ,  cre_nm ,  alb_nm ,  crp_nm ,  hbac_nm , ac_nm ,  ua_nm ,
                          tcho_nm ,  tg_nm ,  hdl_nm ,  ldl_nm ,  got_nm ,  gpt_nm ,  trm_id ,  trman_fl , trmas_fl , 
                          trmti_fl ,  trmhe_fl ,  trmwa_fl , trmia_fl ,  trmfo_fl ,  trmta_fl ,  trmsd_fl ,  trmre_fl , 
                          trmen_fl , trmen_id,  trmag_fl ,  trmcl_fl ,  trmpl_fl ,  trmlm_fl ,  trmiv_fl ,  trmve_fl , 
                          trmng_fl ,  trmdy_fl ,  trmicu_fl ,  trmsm_fl ,  trmed_fl ,  trmop_fl ,  trmop_id ,  trmot_fl ,
                          om_fl ,  omas_fl ,  omag_fl ,  omti_fl ,  omcl_fl ,  omwa_fl ,  ompl_fl ,  omanh_fl ,
                          omand_fl ,  omora_fl ,  omins_fl ,  omli_fl ,  omst_fl ,  omns_fl , 
                          omliot_fl ,  omliot2_fl ,  am_fl ,  amas_fl ,  amag_fl ,  amti_fl ,  amcl_fl ,  amwa_fl , 
                          ampl_fl ,  amanh_fl ,  amand_fl ,  amli_fl ,  amliot_fl ,  amliot2_fl ,  com_id ,  compn_fl , 
                          comut_fl ,  comug_fl ,  compr_fl ,  compu_fl ,  comac_fl ,  comse_fl , comde_fl ,  como_fl ,
                          det_id ,  detst_fl ,  dethe_fl ,  detho_fl ,  detha_fl ,  detva_fl , detre_fl ,  detme_fl ,
                          deto_fl ,  off_id ,  offd_id ,  offdt_id ,  ct_fl , mri_fl ,  ecg_id , 
                          ecgl_fl ,  ecga_fl ,  ecgq_fl ,  ecgo_fl ,  cd_id ,  cdr_id , cdl_id ,  tccs_id ,  tccsr_id , 
                          tccsl_id ,  tccsba_id ,  mra_fl ,  cta_fl ,  dsa_fl , mcd_id ,  mcdr_id ,  mcdl_id , 
                          mcdba_id ,  mcdri_id ,  mcdli_id ,  omad_fl ,  omad_id , dethoh_fl ,  feeding ,  transfers , 
                          bathing ,  toilet_use ,  grooming ,  mobility ,  stairs , dressing ,  bowel_control , 
                          bladder_control ,  total ,  discharged_mrs ,  cortical_aca_ctr ,  cortical_mca_ctr , 
                          subcortical_aca_ctr ,  subcortical_mca_ctr ,  pca_cortex_ctr ,  thalamus_ctr ,  brainstem_ctr ,
                          cerebellum_ctr ,  watershed_ctr ,  hemorrhagic_infarct_ctr ,  old_stroke_ctci ,
                          cortical_aca_ctl ,  cortical_mca_ctl ,  subcortical_aca_ctl ,  subcortical_mca_ctl ,
                          pca_cortex_ctl ,  thalamus_ctl ,  brainstem_ctl ,  cerebellum_ctl , watershed_ctl , 
                          hemorrhagic_infarct_ctl ,  old_stroke_ctch ,  cortical_aca_mrir , cortical_mca_mrir ,
                          subcortical_aca_mrir ,  subcortical_mca_mrir ,  pca_cortex_mrir , thalamus_mrir , 
                          brainstem_mrir ,  cerebellum_mrir ,  watershed_mrir , hemorrhagic_infarct_mrir ,
                          old_stroke_mrici ,  cortical_aca_mril ,  cortical_mca_mril , subcortical_aca_mril ,
                          subcortical_mca_mril ,  pca_cortex_mril ,  thalamus_mril , brainstem_mril ,  cerebellum_mril , 
                          watershed_mril ,  hemorrhagic_infarct_mril , old_stroke_mrich ,  hd_id ,  pcva_id , 
                          pcvaci_id ,  pcvach_id ,  po_id , ur_id , sm_id ,  smc_nm ,  smy_nm ,  smcp_id ,  ptia_id ,
                          hc_id ,  hcht_id ,  hchc_id , ht_id ,  dm_id , pad_id , al_id ,  ca_id ,  ot_id ,  thishy_id ,  
                          thisdi_id , fahiid_parents_1 ,  fahiid_parents_2 ,  fahiid_parents_3 ,  fahiid_parents_4 ,  
                          fahiid_brsi_1 , fahiid_brsi_2 , fahiid_brsi_3 ,  fahiid_brsi_4 ,  nihs_1a_in ,  nihs_1b_in ,  
                          nihs_1c_in , nihs_2_in ,  nihs_3_in ,  nihs_4_in , nihs_5al_in ,  nihs_5br_in ,  nihs_6al_in , 
                          nihs_6br_in ,  nihs_7_in ,  nihs_8_in ,  nihs_9_in ,  nihs_10_in , nihs_11_in ,  total_in , 
                          nihs_1a_out ,  nihs_1b_out ,  nihs_1c_out ,  nihs_2_out ,  nihs_3_out ,  nihs_4_out ,
                          nihs_5al_out ,  nihs_5br_out ,  nihs_6al_out ,  nihs_6br_out ,  nihs_7_out ,  nihs_8_out ,
                          nihs_9_out ,  nihs_10_out ,  nihs_11_out ,  total_out ,  SexName ,  Age, mrs_tx_1, mrs_tx_3,
                          mrs_tx_6]).T
TSR_6_CLEANED[TSR_6_CLEANED == "N"] = 0
TSR_6_CLEANED[TSR_6_CLEANED == "Y"] = 1

csv_save = os.path.join("..", "data", "LINKED_DATA", "TSR_EHR", "TSR_6_CLEANED.csv")
TSR_6_CLEANED.to_csv(csv_save, index=False)
