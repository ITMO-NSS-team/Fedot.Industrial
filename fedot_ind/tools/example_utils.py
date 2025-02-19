import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.metrics.metrics_implementation import calculate_forecasting_metric
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH

ts_datasets = {
    'm4_yearly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Yearly.csv'),
    'm4_weekly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Weekly.csv'),
    'm4_daily': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Daily.csv'),
    'm4_monthly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Monthly.csv'),
    'm4_quarterly': Path(PROJECT_PATH, 'examples', 'data', 'ts', 'M4Quarterly.csv')}


def evaluate_metric(target, prediction):
    try:
        if len(np.unique(target)) > 2:
            metric = f1_score(target, prediction, average='weighted')
        else:
            metric = roc_auc_score(target, prediction, average='weighted')
    except Exception:
        metric = f1_score(target, np.argmax(
            prediction, axis=1), average='weighted')
    return metric


def compare_forecast_with_sota(dataset_name, horizon):
    autogluon = PROJECT_PATH + f'/benchmark/results/benchmark_results/autogluon/' \
                               f'{dataset_name}_{horizon}_forecast_vs_actual.csv'
    n_beats = PROJECT_PATH + f'/benchmark/results/benchmark_results/nbeats/' \
                             f'{dataset_name}_{horizon}_forecast_vs_actual.csv'
    n_beats = pd.read_csv(n_beats)
    autogluon = pd.read_csv(autogluon)

    n_beats_forecast = calculate_forecasting_metric(
        target=n_beats['value'].values,
        labels=n_beats['predict'].values)
    autogluon_forecast = calculate_forecasting_metric(
        target=autogluon['value'].values,
        labels=autogluon['predict'].values)
    return n_beats['predict'].values, n_beats_forecast, autogluon['predict'].values, autogluon_forecast


def read_results(forecast_result_path):
    results = os.listdir(forecast_result_path)
    df_forecast = []
    df_metrics = []
    for file in results:
        df = pd.read_csv(f'{forecast_result_path}/{file}')
        name = file.split('_')[0]
        df['dataset_name'] = name
        if file.__contains__('forecast'):
            df_forecast.append(df)
        else:
            df_metrics.append(df)
    return df_forecast, df_metrics


def create_comprasion_df(df, metric: str = 'rmse'):
    df_full = pd.concat(df)
    df_full = df_full[df_full['Unnamed: 0'] == metric]
    df_full = df_full.drop('Unnamed: 0', axis=1)
    df_full['Difference_industrial_All'] = (
        df_full.iloc[:, 1:3].min(axis=1) - df_full['industrial'])
    df_full['Difference_industrial_AG'] = (
        df_full.iloc[:, 1:2].min(axis=1) - df_full['industrial'])
    df_full['Difference_industrial_NBEATS'] = (
        df_full.iloc[:, 2:3].min(axis=1) - df_full['industrial'])
    df_full['industrial_Wins_All'] = df_full.apply(
        lambda row: 'Win' if row.loc['Difference_industrial_All'] > 0 else 'Loose', axis=1)
    df_full['industrial_Wins_AG'] = df_full.apply(
        lambda row: 'Win' if row.loc['Difference_industrial_AG'] > 0 else 'Loose', axis=1)
    df_full['industrial_Wins_NBEATS'] = df_full.apply(
        lambda row: 'Win' if row.loc['Difference_industrial_NBEATS'] > 0 else 'Loose',
        axis=1)
    return df_full
