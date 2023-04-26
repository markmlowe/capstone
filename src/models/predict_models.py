import pandas as pd

# create the logistic regression  model
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def predict_logr(log_r, x_test, y_test):

    # generate predicitons for logistic regression model
    logr_preds = log_r.predict(x_test)
    # generate the accuracy score for the model
    logr_score = log_r.score(x_test, y_test)

    return logr_preds, logr_score


def predict_xgb_cl(xgb_cl, x_test, y_test):

    # generate predictions for the xgb_classifier model
    xgb_preds = xgb_cl.predict(x_test)
    # generate the accuracy score for the model
    xgb_score = xgb_cl.score(x_test, y_test)
    
    return xgb_preds, xgb_score