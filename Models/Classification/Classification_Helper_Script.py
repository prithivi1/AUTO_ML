from Models.Classification.Base_Model_Script import *
from Models.Cross_Validation_Script import *

import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm
from catboost import CatBoostClassifier


def train_base_model(X, y, score, n_split):
    base_models = {}

    error_df, final_model = Logistic_Regression(X, y, score, n_split)
    base_models[LogisticRegression().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = knn_Classifier(X, y, score, n_split)
    base_models[KNeighborsClassifier().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = Decision_Tree(X, y, score, n_split)
    base_models[DecisionTreeClassifier().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = Random_Forest(X, y, score, n_split)
    base_models[RandomForestClassifier().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = Ada_Boost(X, y, score, n_split)
    base_models[AdaBoostClassifier().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = Gradient_Boosting(X, y, score, n_split)
    base_models[GradientBoostingClassifier().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = XGBoosting(X, y, score, n_split)
    base_models[XGBClassifier().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = lightBGM(X, y, score, n_split)
    base_models[lightgbm.LGBMModel().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    error_df, final_model = catBoost(X, y, score, n_split)
    base_models[CatBoostClassifier().__class__.__name__] = {'Error': error_df, 'Final Model': final_model}

    return base_models


def fitted_model(model, df, y, score):
    X_train, X_test, y_train, y_test = train_test(df, y)
    model.fit(X_train, y_train)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    training_error, testing_error = classification_error_function(y_train, y_test, train_predict, test_predict, score)

    return model, training_error, testing_error


def manual_cross_validate(model, X, y, score, n_split):
    error_df = pd.DataFrame(columns=['Trail', 'Training Accuracy', 'Testing Accuracy'])

    k_fold = KFold(n_splits=n_split)
    i = 1
    for train_index, test_index in k_fold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)

        training_accuracy, testing_accuracy = classification_error_function(y_train, y_test, train_prediction,
                                                                            test_prediction, score)
        error_df.loc[len(error_df.index)] = ['CV_' + str(i), training_accuracy, testing_accuracy]
        i = i + 1

    return error_df


def classification_error_function(y_train, y_test, train_prediction, test_prediction, function_name):
    if function_name == 'accuracy':
        training_error = metrics.accuracy_score(y_train, train_prediction)
        testing_error = metrics.accuracy_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'balanced_accuracy':
        training_error = metrics.balanced_accuracy_score(y_train, train_prediction)
        testing_error = metrics.balanced_accuracy_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'top_k_accuracy':
        training_error = metrics.top_k_accuracy_score(y_train, train_prediction)
        testing_error = metrics.top_k_accuracy_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'average_precision':
        training_error = metrics.average_precision_score(y_train, train_prediction)
        testing_error = metrics.average_precision_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_brier_score':
        training_error = metrics.neg_brier_score(y_train, train_prediction)
        testing_error = metrics.neg_brier_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'f1':
        training_error = metrics.f1_score(y_train, train_prediction)
        testing_error = metrics.f1_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'neg_log_loss':
        training_error = metrics.neg_log_loss(y_train, train_prediction)
        testing_error = metrics.neg_log_loss(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'precision':
        training_error = metrics.precision_score(y_train, train_prediction)
        testing_error = metrics.precision_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'recall':
        training_error = metrics.recall_score(y_train, train_prediction)
        testing_error = metrics.recall_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'roc_auc':
        training_error = metrics.roc_auc_score(y_train, train_prediction)
        testing_error = metrics.roc_auc_score(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'roc_auc_ovr':
        training_error = metrics.roc_auc_ovr(y_train, train_prediction)
        testing_error = metrics.roc_auc_ovr(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'roc_auc_ovo':
        training_error = metrics.roc_auc_ovo(y_train, train_prediction)
        testing_error = metrics.roc_auc_ovo(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'roc_auc_ovr_weighted':
        training_error = metrics.roc_auc_ovr_weighted(y_train, train_prediction)
        testing_error = metrics.roc_auc_ovr_weighted(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'roc_auc_ovo_weighted':
        training_error = metrics.roc_auc_ovr(y_train, train_prediction)
        testing_error = metrics.roc_auc_ovr(y_test, test_prediction)
        return training_error, testing_error
    elif function_name == 'jaccard':
        training_error = metrics.jaccard_score(y_train, train_prediction)
        testing_error = metrics.jaccard_score(y_test, test_prediction)
        return training_error, testing_error
