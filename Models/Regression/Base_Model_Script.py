from Models.Regression import Helper_Script

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


def Linear_Regression(X, y, score, n_split):
    model = LinearRegression()
    return Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def ridgeRegressor(X, y, score, n_split):
    model_lasso = Lasso()
    return Helper_Script.manual_cross_validate(model_lasso, X, y, score, n_split), model_lasso


def lassoRegressor(X, y, score, n_split):
    model = Ridge()
    return Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def Random_Forest(X, y, score, n_split):
    model = RandomForestRegressor()
    return Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def Gradient_Boosting(X, y, score, n_split):
    model = GradientBoostingRegressor()
    return Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def XGBoosting(X, y, score, n_split):
    model = XGBRegressor()
    return Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def lightBGM(X, y, score, n_split):
    model = LGBMRegressor()
    return Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def catBoost(X, y, score, n_split):
    model = CatBoostRegressor()
    return Helper_Script.manual_cross_validate(model, X, y, score, n_split), model
