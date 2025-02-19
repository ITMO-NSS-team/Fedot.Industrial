from fedot_ind.api.utils.checkers_collections import ApiConfigCheck
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_TSF_AUTOML_CONFIG, \
    DEFAULT_TSF_LEARNING_CONFIG, DEFAULT_TSF_INDUSTRIAL_CONFIG
from fedot_ind.core.repository.constanst_repository import UNI_CLF_BENCH


def get_benchmark_setup():
    RESNET_LEARNING_PARAMS = {'epochs': 150,
                              'lr': 0.001,
                              'device': 'cpu'
                              }
    STAT_CLF_PARAMS = {'window_size': 10,
                       'add_global_features': True,
                       'use_sliding_window': False,
                       'stride': 1,
                       'channel_model': 'logit',
                       }
    model_to_compare = [{0: [('industrial_stat_clf', STAT_CLF_PARAMS)]},
                        {},
                        {0: [('resnet_model', RESNET_LEARNING_PARAMS)]}]
    model_name = ['stat_clf', 'industrial', 'resnet']
    finutune_existed_model = [True, False, True]
    BENCHMARK = 'UCR_CLF'
    BENCHMARK_PARAMS = {'experiment_date': '04_02_25',
                        'datasets': UNI_CLF_BENCH,
                        'model_to_compare': (model_to_compare, model_name, finutune_existed_model)}
    return BENCHMARK, BENCHMARK_PARAMS


API_CONFIG = {'industrial_config': DEFAULT_TSF_INDUSTRIAL_CONFIG,
              'automl_config': DEFAULT_TSF_AUTOML_CONFIG,
              'learning_config': DEFAULT_TSF_LEARNING_CONFIG,
              'compute_config': DEFAULT_COMPUTE_CONFIG}

BENCHMARK_CONFIG = {'task': 'classification',
                    'timeout': 15,
                    'n_workers': 2,
                    'threads_per_worker': 4,
                    'with_tuning': False,
                    'logging_level': 20
                    }
config_agent = ApiConfigCheck()

if __name__ == "__main__":
    config_agent.compare_configs(API_CONFIG, UPD_API_CONFIG)
    api_agent = ApiTemplate(api_config=UPD_API_CONFIG, metric_list=('accuracy', 'f1'))
    BENCHMARK, BENCHMARK_PARAMS = get_benchmark_setup()
    api_agent.evaluate_benchmark(benchmark_name=BENCHMARK,
                                 benchmark_params=BENCHMARK_PARAMS)
