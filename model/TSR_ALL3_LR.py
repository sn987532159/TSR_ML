import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import TomekLinks
import joblib

#GOOD when discharged
csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_X_TRAIN.csv")
G_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3G_y_TRAIN.csv")
G_y_train = pd.read_csv(csv_path)
G_y_train = np.ravel(G_y_train)

#BAD when discharged
csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_X_TRAIN.csv")
B_X_train = pd.read_csv(csv_path)

csv_path = os.path.join("..", "data", "LINKED_DATA", "TSR_ALL", "TSR_ALL3", "TSR_ALL3B_y_TRAIN.csv")
B_y_train = pd.read_csv(csv_path)
B_y_train = np.ravel(B_y_train)

# class weight
G_y_train_1 = sum(G_y_train)
G_y_train_0 = len(G_y_train) - sum(G_y_train)

B_y_train_1 = sum(B_y_train)
B_y_train_0 = len(B_y_train) - sum(B_y_train)

# base lr
lr = LogisticRegression(random_state=19, max_iter=10000)

#tune lr
hyperparameters_lrG = {"logisticregression__class_weight": ("None", 'balanced', {0: G_y_train_1, 1: G_y_train_0}, {0: G_y_train_0, 1: G_y_train_1})}

hyperparameters_lrB = {"logisticregression__class_weight": ("None", 'balanced', {0: B_y_train_1, 1: B_y_train_0}, {0: B_y_train_0, 1: B_y_train_1})}

pipeline = make_pipeline(TomekLinks(), LogisticRegression(random_state=19, max_iter=10000))

lrG_gscv = GridSearchCV(estimator=pipeline,
                        param_grid=hyperparameters_lrG,
                        n_jobs=-1,
                        scoring='roc_auc',
                        verbose=5,
                        cv=5)

lrB_gscv = GridSearchCV(estimator=pipeline,
                        param_grid=hyperparameters_lrB,
                        n_jobs=-1,
                        scoring='roc_auc',
                        verbose=5,
                        cv=5)
#GOOD when discharged
### BASED LR
lrG = lr.fit(G_X_train, G_y_train)
joblib.dump(lrG, "model_pickle/MICE1/TSR_ALL3G_LR_BASED.pkl")

### TUNED LR
lrG_gsCV = lrG_gscv.fit(G_X_train, G_y_train)
joblib.dump(lrG_gsCV, "model_pickle/MICE1/TSR_ALL3G_LR_TUNED.pkl")

### CALIBRATED LR
lrG_cccv = CalibratedClassifierCV(base_estimator=lrG_gsCV.best_estimator_, cv=5)
lrG_ccCV = lrG_cccv.fit(G_X_train, G_y_train)
joblib.dump(lrG_ccCV, "model_pickle/MICE1/TSR_ALL3G_LR_CALIBRATED.pkl")

#BAD when discharged
### BASED LR
lrB = lr.fit(B_X_train, B_y_train)
joblib.dump(lrB, "model_pickle/MICE1/TSR_ALL3B_LR_BASED.pkl")

### TUNED LR
lrB_gsCV = lrB_gscv.fit(B_X_train, B_y_train)
joblib.dump(lrB_gsCV, "model_pickle/MICE1/TSR_ALL3B_LR_TUNED.pkl")

### CALIBRATED LR
lrB_cccv = CalibratedClassifierCV(base_estimator=lrB_gsCV.best_estimator_, cv=5)
lrB_ccCV = lrB_cccv.fit(B_X_train, B_y_train)
joblib.dump(lrB_ccCV, "model_pickle/MICE1/TSR_ALL3B_LR_CALIBRATED.pkl")