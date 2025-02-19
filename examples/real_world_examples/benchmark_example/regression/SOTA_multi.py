import gc

from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_CLF_AUTOML_CONFIG, \
    DEFAULT_CLF_LEARNING_CONFIG, DEFAULT_CLF_INDUSTRIAL_CONFIG
from fedot_ind.core.repository.constanst_repository import MULTI_REG_BENCH

gc.collect()

bagging_ensemble = {0: ['industrial_freq_reg'],
                    1: ['industrial_stat_reg'],
                    'head': ['bagging']}


def get_benchmark_setup():
    RESNET_LEARNING_PARAMS = {'epochs': 150,
                              'lr': 0.001,
                              'device': 'cpu'
                              }

    model_to_compare = [bagging_ensemble,
                        {},
                        {0: [('resnet_model', RESNET_LEARNING_PARAMS)]}]
    model_name = ['stat_reg', 'industrial', 'resnet']
    finutune_existed_model = [True, False, True]
    BENCHMARK = 'UCR_REG'
    BENCHMARK_PARAMS = {'experiment_date': '18_02_25',
                        'datasets': MULTI_REG_BENCH,
                        'model_to_compare': (model_to_compare, model_name, finutune_existed_model)}
    return BENCHMARK, BENCHMARK_PARAMS


API_CONFIG = {'industrial_config': DEFAULT_CLF_INDUSTRIAL_CONFIG,
              'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
              'learning_config': DEFAULT_CLF_LEARNING_CONFIG,
              'compute_config': DEFAULT_COMPUTE_CONFIG}

BENCHMARK_CONFIG = {'task': 'regression',
                    'timeout': 15,
                    'n_workers': 2,
                    'threads_per_worker': 4,
                    'with_tuning': False,
                    'logging_level': 20
                    }
config_agent = ApiConfigCheck()

if __name__ == "__main__":
    UPD_API_CONFIG = config_agent.update_config_with_kwargs(config_to_update=API_CONFIG,
                                                            **BENCHMARK_CONFIG)
    config_agent.compare_configs(API_CONFIG, UPD_API_CONFIG)
    api_agent = ApiTemplate(api_config=UPD_API_CONFIG, metric_list=('accuracy', 'f1'))
    BENCHMARK, BENCHMARK_PARAMS = get_benchmark_setup()
    api_agent.evaluate_benchmark(benchmark_name=BENCHMARK,
                                 benchmark_params=BENCHMARK_PARAMS)
