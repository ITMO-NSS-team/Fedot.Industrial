import pickle

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_TSF_AUTOML_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG
from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH


if __name__ == "__main__":
    METRIC_NAMES = ('rmse', 'smape')
    HORIZON = 8
    TASK_PARAMS = {'forecast_length': HORIZON}
    BENCH = 'M4'
    GROUP = 'M'

    DATASET_NAMES = [data for data in M4_FORECASTING_BENCH if data.__contains__(GROUP)]

    DEFAULT_AUTOML_LEARNING_CONFIG['timeout'] = 5

    API_CONFIG = {'industrial_config': {'problem': 'ts_forecasting',
                                        'data_type': 'time_series',
                                        'learning_strategy': 'forecasting_assumptions',
                                        'task_params': TASK_PARAMS},
                  'automl_config': {**DEFAULT_TSF_AUTOML_CONFIG,
                                    'task_params': TASK_PARAMS},
                  'learning_config': {'learning_strategy': 'from_scratch',
                                      'learning_strategy_params': DEFAULT_AUTOML_LEARNING_CONFIG,
                                      'optimisation_loss': {'quality_loss': 'rmse'}},
                  'compute_config': DEFAULT_COMPUTE_CONFIG}

    result_dict = {}

    for dataset_name in DATASET_NAMES:
        dataset_dict = {'benchmark': BENCH,
                        'dataset': dataset_name,
                        'task_params': TASK_PARAMS}
        result_dict = ApiTemplate(api_config=API_CONFIG,
                                  metric_list=METRIC_NAMES).eval(dataset=dataset_dict,
                                                                 finetune=False)
        result_dict.update({dataset_name: result_dict})

    with open(f'{BENCH}_{GROUP}_forecast_length_{HORIZON}.pkl', 'wb') as f:
        pickle.dump(result_dict, f)
