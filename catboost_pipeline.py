from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    make_scorer,
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd, numpy as np
from catboost import CatBoostClassifier


from collections import defaultdict
from functools import partial
from sklearn.metrics._base import _average_binary_score
from numpy.random import RandomState
import operator

import hyperopt

class CatBoostPipeline:
    def __init__(self):
        self.fit_params = None
        self.init_hyperparams = {
            "random_seed": 7,
            "verbose": 0,
            "iterations": 50,
        }
        self.X_train = None
        self.y_train = None
        self.output_features = None
    
    def select_features_greedy(self, cat_features, text_features, scoring=None):
        params = self.init_hyperparams.copy()
        params.update(
            cat_features=cat_features,
            text_features=text_features
        )
        if not scoring:
            scoring = self.scoring
        clf = CatBoostClassifier(**params)
        best = np.mean(cross_val_score(
            clf, self.X_train, self.y_train, scoring=scoring, cv=3, n_jobs=-2,
        ))
        print('All features: ', best)
        features = self.output_features.copy()
        
        while True:
            visited_features = []
            for f in features:
                new_features = features.copy()
                new_cat_features = cat_features.copy()
                new_text_features = text_features.copy()
                new_features.remove(f)
                if f in cat_features:
                    new_cat_features.remove(f)
                if f in text_features:
                    new_text_features.remove(f)
                new_params = params.copy()
                new_params.update(
                    cat_features=new_cat_features, text_features=new_text_features
                )
                clf = CatBoostClassifier(**new_params)
                temp = np.mean(cross_val_score(
                    clf, self.X_train[new_features], self.y_train, scoring=scoring,
                    cv=3, n_jobs=-2,
                ))
                print(f, temp)
                if temp >= best - 0.0001:
                    best = temp
                    features = new_features.copy()
                    cat_features = new_cat_features.copy()
                    text_features = new_text_features.copy()
                    print(f, "deleted", temp)
                    break
                else:
                    visited_features.append(f)
            if len(visited_features) == len(features):
                break
        self.output_features = features
        self.fit_params = dict(cat_features=cat_features, text_features=text_features)
        self.X_train = self.X_train[self.output_features]
        
    def select_features_exhaustive(self, cat_features, text_features, scoring=None):
        params = self.init_hyperparams.copy()
        params.update(
            cat_features=cat_features,
            text_features=text_features
        )
        if not scoring:
            scoring = self.scoring
        clf = CatBoostClassifier(**params)
        best = np.mean(cross_val_score(
            clf, self.X_train, self.y_train, scoring=scoring, cv=5, n_jobs=-2,
        ))
        print('All features: ', best)
        features = self.output_features.copy()
        while True:
            feature_scores = {}
            for f in features:
                new_features = features.copy()
                new_cat_features = cat_features.copy()
                new_text_features = text_features.copy()
                new_features.remove(f)
                if f in cat_features:
                    new_cat_features.remove(f)
                if f in text_features:
                    new_text_features.remove(f)
                new_params = params.copy()
                new_params.update(
                    cat_features=new_cat_features, text_features=new_text_features
                )
                clf = CatBoostClassifier(**new_params,)
                temp = np.mean(cross_val_score(
                    clf, self.X_train[new_features], self.y_train, scoring=scoring, cv=5, n_jobs=-2,
                ))
                feature_scores[f] = temp
            if max(feature_scores.values()) < best - 0.0001:
                break
            delete_f = max(feature_scores.items(), key=operator.itemgetter(1))[0]
            best = feature_scores[delete_f]
            features = [x for x in features if x != delete_f]
            cat_features = [x for x in cat_features if x != delete_f]
            text_features = [x for x in text_features if x != delete_f]
            print('delete', delete_f, best)

        self.output_features = features
        self.fit_params = dict(cat_features=cat_features, text_features=text_features)
        self.X_train = self.X_train[self.output_features]

    def hyperopt_objective(self, params):
        all_params = self.init_hyperparams.copy()
        all_params.update(self.fit_params)
        all_params.update(params)
        model = CatBoostClassifier(**all_params)
        score = np.mean(cross_val_score(
            model, self.X_train, self.y_train, scoring=self.scoring, cv=5, n_jobs=-2,
        ))

        return 1 - score  # as hyperopt minimises

    def hyperparam_tuning(self, params_space, max_evals=30):
        self.trials = hyperopt.Trials()

        best = hyperopt.fmin(
            self.hyperopt_objective,
            space=params_space,
            algo=hyperopt.tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            rstate=RandomState(123),
        )
        self.params_space = params_space
        self.best_params = best