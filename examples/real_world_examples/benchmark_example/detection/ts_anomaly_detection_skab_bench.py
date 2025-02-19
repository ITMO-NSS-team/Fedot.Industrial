import os

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_CLF_AUTOML_CONFIG
from fedot_ind.tools.serialisation.path_lib import EXAMPLES_DATA_PATH


def prepare_skab_benchmark():
    ENCODER_LEARNING_PARAMS = {'epochs': 150,
                               'lr': 0.001,
                               'device': 'cpu'
                               }
    model_to_compare = [
        # {0: ['iforest_detector']},
        {0: [('conv_ae_detector', ENCODER_LEARNING_PARAMS)]},
        # {0: ['stat_detector']},
        # {}
    ]
    model_name = [
        # 'iforest',
        'conv_encoder',
        # 'stat_detector',
        # 'industrial'
    ]
    finutune_existed_model = [
        True,
        True,
        # True, False
    ]
    BENCHMARK = 'SKAB'
    folder = 'valve1'
    datasets = os.listdir(EXAMPLES_DATA_PATH + f'/benchmark/detection/data/{folder}')
    datasets = [x.split('.')[0] for x in datasets]
    BENCHMARK_PARAMS = {'experiment_date': '23_01_25',
                        'metadata': {'folder': folder},
                        'datasets': datasets,
                        'model_to_compare': (model_to_compare, model_name, finutune_existed_model)}
    return BENCHMARK, BENCHMARK_PARAMS


METRIC_NAMES = ('nab', 'accuracy')
EVAL_REGIME = True

COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = DEFAULT_CLF_AUTOML_CONFIG
AUTOML_LEARNING_STRATEGY = dict(timeout=1,
                                n_jobs=2,
                                pop_size=10,
                                logging_level=0)

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'accuracy'}}

INDUSTRIAL_CONFIG = {'strategy': 'anomaly_detection',
                     'problem': 'classification',
                     'strategy_params': {'detection_window': 10,
                                         'train_data_size': 'anomaly-free',
                                         'data_type': 'time_series'}}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    api_agent = ApiTemplate(api_config=API_CONFIG, metric_list=METRIC_NAMES)
    BENCHMARK, BENCHMARK_PARAMS = prepare_skab_benchmark()
    if EVAL_REGIME:
        api_agent.evaluate_benchmark(benchmark_name=BENCHMARK,
                                     benchmark_params=BENCHMARK_PARAMS)
