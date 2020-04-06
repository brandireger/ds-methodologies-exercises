import pandas as pd
import numpy as np
import seaborn as sns

from statsmodels.formula.api import ols
from math import sqrt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from pydataset import data

def plot_residuals(y, yhat):
    resids = yhat - y
    sns.scatterplot(y, resids)

def regression_errors(y, yhat):
    """
    Takes in two columns: y is the variable being predicted, yhat is the predictions
    Returns 5 values: sum of squared errors (SSE), explained sum of squares (ESS),
    total sum of squares (TSS), mean squared error (MSE), root mean squared error (RMSE)
    """
    SSE = mean_squared_error(y, yhat)*len(y)
    ESS = sum((yhat - y.mean()) ** 2)
    TSS = ESS + SSE
    MSE = mean_squared_error(y, yhat)
    RMSE = sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    """
    Takes in one column: the variable being predicted, and the dataframe containing the column
    Returns 3 baseline values: sum of squared errors (SSE), mean squared error (MSE),
    root mean squared error (RMSE)
    """
    ydf = pd.DataFrame(y)
    ydf['yhat_bl'] = y.mean()
    SSE_bl = mean_squared_error(y, ydf.yhat_bl)*len(y)
    MSE_bl = mean_squared_error(y, ydf.yhat_bl)
    RMSE_bl = sqrt(MSE_bl)
    return SSE_bl, MSE_bl, RMSE_bl

def better_than_baseline(y, yhat):
    """
    Takes in two columns and their dataframe: y is the variable being predicted, yhat is the predictions
    Returns True if the model performs better than the baseline model (baseline model is the mean of y)
    Returns False if not
    """
    ydf = pd.DataFrame(y, yhat)
    ydf['yhat_bl'] = y.mean()
    SSE = mean_squared_error(y, yhat)*len(y)
    SSE_bl = mean_squared_error(y, ydf.yhat_bl)*len(y)
    if SSE < SSE_bl:
        return True
    else:
        return False
    
def model_significance(ols_model):
    """
    Takes in the ols_model created by statsmodels.formula.api
    Returns R^2 and p-value
    """
    r2 = ols_model.rsquared
    f_pval = ols_model.f_pvalue
    return "p-value:", round(f_pval,4), 'R-squared:', round(r2,3)

   