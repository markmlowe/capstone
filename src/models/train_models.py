import pandas as pd

# create the logistic regression  model
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def train_logr(x_bal_train,y_bal_train):
    # create the base instance of the model for exploration
    log_r = LogisticRegression(max_iter=100000)

    # fit the data on the balanced training data
    log_r.fit(x_bal_train, y_bal_train)

    return log_r


def train_xgboost(x_bal_train,y_bal_train):
    # create the xgboost model for comparison with logsitic regression
    xgb_cl = xgb.XGBClassifier()

    # Fit the xgboost classifier on the balanced dataset after SMOTE
    xgb_cl.fit(x_bal_train, y_bal_train)

    return xgb_cl