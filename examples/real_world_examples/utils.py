import pandas as pd

from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH


def sota_compare(dataset_name, metrics, data_path=PROJECT_PATH + '/benchmark'):
    df = pd.read_csv(data_path + '/ts_regression_sota_results.csv', sep=';')
    df = df[df['ds/type'] == dataset_name].iloc[:, :25]
    df.index = df['algorithm']
    df = df.drop(['algorithm', 'ds/type'], axis=1)
    df = df.replace(',', '.', regex=True).astype(float)
    df.loc['min', 'Fedot_Industrial_AutoML'] = metrics['rmse'].min()
    df.loc['max', 'Fedot_Industrial_AutoML'] = metrics['rmse'].max()
    df.loc['average', 'Fedot_Industrial_AutoML'] = metrics['rmse'].mean()
    df = df.T
    return df
