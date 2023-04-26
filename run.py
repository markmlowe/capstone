import pandas as pd

from src.data.make_dataset import create_dataset
from src.models.train_models import train_logr, train_xgboost
from src.models.predict_models import predict_logr, predict_xgb_cl
from src.visualization.visualize import conf_matr

def run_models():

    x_bal_train, y_bal_train, x_test, y_test = create_dataset()

    #print(x_bal_train.head(2))

    log_r = train_logr(x_bal_train, y_bal_train)
    xgb_cl = train_xgboost(x_bal_train, y_bal_train)

    logr_preds, logr_score = predict_logr(log_r, x_test, y_test)
    xgb_preds, xgb_score = predict_xgb_cl(xgb_cl, x_test, y_test)

    # confusion matrix for logistic regression
    conf_matr(y_test, logr_preds, logr_score)

    # confusion matric for xgboost_classifier
    conf_matr(y_test, xgb_preds, xgb_score)



    return 0


run_models()