import os

from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_TSF_AUTOML_CONFIG, \
    DEFAULT_TSF_LEARNING_CONFIG, DEFAULT_TSF_INDUSTRIAL_CONFIG
from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH, M4_FORECASTING_LENGTH


def get_benchmark_setup():
    DEEPAR_LEARNING_PARAMS = {'epochs': 150, 'lr': 0.001, 'device': 'cpu'}
    model_to_compare = [{0: ['lagged_forecaster']}, {}, {0: [('deepar_model', DEEPAR_LEARNING_PARAMS)]}, {0: ['ar']}]
    model_name = ['lagged_regression', 'industrial', 'deepar', 'ar']
    finutune_existed_model = [True, False, True, True]
    BENCHMARK = 'M4'
    BENCHMARK_PARAMS = {'experiment_date': '24_01_25',
                        'metadata': M4_FORECASTING_LENGTH,
                        'datasets': M4_FORECASTING_BENCH,
                        'model_to_compare': (model_to_compare, model_name, finutune_existed_model)}
    return BENCHMARK, BENCHMARK_PARAMS


API_CONFIG = {'industrial_config': DEFAULT_TSF_INDUSTRIAL_CONFIG,
              'automl_config': DEFAULT_TSF_AUTOML_CONFIG,
              'learning_config': DEFAULT_TSF_LEARNING_CONFIG,
              'compute_config': DEFAULT_COMPUTE_CONFIG}

BENCHMARK_CONFIG = {'task': 'ts_forecasting',
                    'task_params': {'forecast_length': 14},
                    'timeout': 5,
                    'n_workers': 2,
                    'threads_per_worker': 2,
                    'with_tuning': False,
                    'logging_level': 20
                    }
config_agent = ApiConfigCheck()

if __name__ == "__main__":
    UPD_API_CONFIG = config_agent.update_config_with_kwargs(config_to_update=API_CONFIG,
                                                            **BENCHMARK_CONFIG)

    config_agent.compare_configs(API_CONFIG, UPD_API_CONFIG)
    api_agent = ApiTemplate(api_config=UPD_API_CONFIG, metric_list=('rmse', 'mae'))
    BENCHMARK, BENCHMARK_PARAMS = get_benchmark_setup()
    EVALUATED = os.listdir('./M4_24_01_25/ar')
    DATASETS = [x for x in M4_FORECASTING_BENCH if x not in EVALUATED]
    BENCHMARK_PARAMS['datasets'] = DATASETS
    api_agent.evaluate_benchmark(benchmark_name=BENCHMARK,
                                 benchmark_params=BENCHMARK_PARAMS)
