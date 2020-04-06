from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE

def select_kbest(X, y, k):
    """
    Takes in 3 variables:
        X is the predictor variables, y is the variable to be predicted, k is the number of features to select
    Returns the names of the k best features
    """
    f_selector = SelectKBest(f_regression, k=k).fit(X, y)
    f_feature = X.loc[:,f_selector.get_support()].columns.tolist()
    return print(f_feature)

def select_rfe(X, y, k):
    """
    Takes in 3 variables:
        X is the predictor variables, y is the variable to be predicted, k is the number of features to select
    Returns the names of the k best features
    """
    lm = LinearRegression()
    rfe = RFE(lm, k)
    X_rfe = rfe.fit_transform(X, y)
    rfe_features = X.loc[:,rfe.support_].columns.tolist()
    print(rfe_features)