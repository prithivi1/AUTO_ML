from Models.Classification import Classification_Helper_Script

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier


def Logistic_Regression(X, y, score, n_split):
    model = LogisticRegression()
    return Classification_Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def knn_Classifier(X, y, score, n_split):
    model = KNeighborsClassifier(n_neighbors=5)
    return Classification_Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def Decision_Tree(X, y, score, n_split):
    model = DecisionTreeClassifier()
    return Classification_Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def Random_Forest(X, y, score, n_split):
    model = RandomForestClassifier()
    return Classification_Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def Ada_Boost(X, y, score, n_split):
    dt_model = DecisionTreeClassifier()
    model = AdaBoostClassifier(base_estimator=dt_model)
    return Classification_Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def Gradient_Boosting(X, y, score, n_split):
    model = GradientBoostingClassifier()
    return Classification_Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def XGBoosting(X, y, score, n_split):
    model = XGBClassifier()
    return Classification_Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def lightBGM(X, y, score, n_split):
    model = LGBMClassifier()
    return Classification_Helper_Script.manual_cross_validate(model, X, y, score, n_split), model


def catBoost(X, y, score, n_split):
    model = CatBoostClassifier()
    return Classification_Helper_Script.manual_cross_validate(model, X, y, score, n_split), model
