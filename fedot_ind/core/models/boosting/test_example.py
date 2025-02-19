import timeit
from functools import partial

import numpy as np
import pandas as pd
from py_boost import GradientBoosting
from py_boost.multioutput.sketching import RandomProjectionSketch
from sklearn.model_selection import train_test_split

from fedot_ind.core.models.boosting.sketch_boosting import SketchBoostModel

N_TREES = 10000
VERBOSE = 100


def use_default_pyboost(X, X_test, y, y_test, use_random: bool = False):
    # sketch = RandomSamplingSketch(10)
    if use_random:
        sketch = RandomProjectionSketch(1)
        model = GradientBoosting(
            'crossentropy',
            ntrees=N_TREES, lr=0.03, verbose=VERBOSE, es=300, lambda_l2=1, gd_steps=1,
            subsample=1, colsample=1, min_data_in_leaf=10,
            max_bin=256, max_depth=6,
            multioutput_sketch=sketch
        )
    else:
        model = GradientBoosting(
            'crossentropy',
            ntrees=N_TREES, lr=0.03, verbose=VERBOSE, es=300, lambda_l2=1, gd_steps=1,
            subsample=1, colsample=1, min_data_in_leaf=10,
            max_bin=256, max_depth=6)

    model.fit(X, y, eval_sets=[{'X': X_test, 'y': y_test}])
    return model


def use_indutstrial_pyboost(X, X_test, y, y_test):
    model = SketchBoostModel(
        loss='crossentropy',
        ntrees=N_TREES, lr=0.03, verbose=VERBOSE, es=300, lambda_l2=1, gd_steps=1,
        subsample=1, colsample=1, min_data_in_leaf=10,
        max_bin=256, max_depth=6,
    )

    model.fit(X, y, eval_sets=[{'X': X_test, 'y': y_test}])
    return model


def load_test_data(use_subsample=None):
    if use_subsample is not None:
        data = pd.read_csv('./helena.csv').iloc[:use_subsample, :]
    else:
        data = pd.read_csv('./helena.csv')
    X = data.drop('class', axis=1).values.astype('float32')
    y = data['class'].values.astype('int32')

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return X, X_test, y, y_test


def eval_boosting_perfomance(eval_func, X, X_test, y, y_test):
    start = timeit.default_timer()
    model = eval_func(X, X_test, y, y_test)
    end = timeit.default_timer()
    iter_per_sec = len(model.history) / (end - start)
    inference_list = []
    for i in range(10):
        start = timeit.default_timer()
        model.predict(X_test)
        end = timeit.default_timer()
        inference = end - start
        inference_list.append(inference)
    return dict(model=model, inference_time=np.mean(inference_list), learning_time=iter_per_sec)


X, X_test, y, y_test = load_test_data(10000)
ind = eval_boosting_perfomance(use_indutstrial_pyboost, X, X_test, y, y_test)
pyboost_default = eval_boosting_perfomance(use_default_pyboost, X, X_test, y, y_test)
pyboost_random = eval_boosting_perfomance(partial(use_default_pyboost, use_random=True), X, X_test, y, y_test)
_ = 1
