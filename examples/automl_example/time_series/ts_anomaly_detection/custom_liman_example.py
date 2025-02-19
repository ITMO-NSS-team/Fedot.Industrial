import os
import pickle

import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.operation.transformation.representation.statistical.quantile_extractor import QuantileExtractor
from fedot_ind.core.repository.constanst_repository import FEDOT_TASK
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH

# liman experiment setup
sampling_rate = 4096
prediction_window = 10
failure_type = ['bearing_failure', 'rotor_failure', 'stator_failure']
power_load = ['0%', '20%', '40%', '60%', '80%', '100%']

# paths to data
dataset_name = 'liman'
train_type = 'without_failure'

# industrial setup
initial_assumptions = {
    'fourier_iforest_detector': PipelineBuilder().add_node(
        'fourier_basis').add_node('iforest_detector'),
    'iforest_detector': PipelineBuilder().add_node('iforest_detector')
}
repo = IndustrialModels().setup_repository()
task = 'classification'
task_params = FEDOT_TASK[task]
industrial_strategy = 'anomaly_detection',
industrial_strategy_params = {
    'sampling_rate': sampling_rate,
    'detection_window': prediction_window,
    'data_type': 'tensor'}
finetune = False
metric_names = ('accuracy')


def read_data(path_train, path_test, sampling_rate: int = 4096):
    path_train = PROJECT_PATH + '/examples/data/' + path_train
    path_test = PROJECT_PATH + '/examples/data/' + path_test
    train_data = [
        pd.read_csv(
            f'{path_train}/{i}',
            index_col=0, header=None).values for i in os.listdir(path_train)]
    train_data = np.concatenate(train_data, axis=1)
    n_samples, n_channels = round(train_data.shape[0] / sampling_rate), train_data.shape[1]
    train_data = train_data.reshape((n_samples, n_channels, sampling_rate))
    train_target = np.zeros((train_data.shape[0], 1))
    test_data = [
        pd.read_csv(
            f'{path_test}/{i}',
            index_col=0, header=None).values for i in os.listdir(path_test)]
    test_data = np.concatenate(test_data, axis=1)
    test_data = test_data.reshape((n_samples, n_channels, sampling_rate))
    test_target = np.ones((test_data.shape[0], 1))
    return (train_data, train_target), (test_data, test_target)


def get_train_spectrum(regime_type: str = train_type):
    spectrum_dict = {}
    feature_dict = {}
    for load in power_load:
        path_train = f'{dataset_name}/{regime_type}/{load}'
        train_data, test_data = read_data(path_train, path_train)
        input_preproc = DataCheck(
            input_data=train_data,
            task=task,
            task_params=task_params,
            industrial_task_params=industrial_strategy_params)
        train_data = input_preproc.check_input_data()
        spectrum_model = PipelineBuilder().add_node('fourier_basis',
                                                    params={'output_format': 'spectrum'}).build()

        spectral_embedings = spectrum_model.fit(train_data)
        spectral_embedings.features = spectral_embedings.predict
        stat_model = QuantileExtractor(dict(window_size=10,
                                            stride=1,
                                            add_global_features=False,
                                            use_sliding_window=False))
        stat_features = stat_model.transform(spectral_embedings)
        spectrum_dict.update({load: spectral_embedings.predict})
        feature_dict.update({load: stat_features.predict})
    return spectrum_dict, feature_dict


if __name__ == "__main__":
    for regime in failure_type:
        train_spectrum, train_features = get_train_spectrum(regime)
        with open(f'{regime}_spectrum.pkl', 'wb') as f:
            pickle.dump(train_spectrum, f)
            print(f'Saved_{regime}_spectrum')
        with open(f'{regime}_features.pkl', 'wb') as f:
            pickle.dump(train_features, f)
            print(f'Saved_{regime}_features')

    for failure in failure_type:
        for load in power_load:
            path_train, path_test = f'{dataset_name}/{train_type}/{load}', \
                                    f'{dataset_name}/{failure}/{load}'
            train_data, test_data = read_data(path_train, path_test)
            data_dict = dict(train_data=train_data, test_data=test_data)
            api_config = dict(
                problem='classification',
                metric='accuracy',
                timeout=0.1,
                pop_size=10,
                cv_folds=2,
                with_tuning=False,
                initial_assumption=initial_assumptions['fourier_iforest_detector'],
                industrial_strategy='anomaly_detection',
                industrial_strategy_params={
                    'sampling_rate': sampling_rate,
                    'detection_window': prediction_window,
                    'data_type': 'tensor'},
                with_tunig=False,
                n_jobs=2,
                logging_level=50)
            result_dict = ApiTemplate(api_config=api_config,
                                      metric_list=metric_names).eval(dataset=dataset_name, finetune=finetune)
            metric = result_dict['metrics'].values[0]
            print(f'FAILURE_TYPE - {failure}. POWER_LOAD - {load}')
            print(f'___________________________________________________')
            print(f'ACCURACY_OF_DETECTION - {metric}')
            print(f'___________________________________________________')
