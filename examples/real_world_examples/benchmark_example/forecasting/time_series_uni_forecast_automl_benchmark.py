import os

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG
from fedot_ind.core.repository.constanst_repository import M4_FORECASTING_BENCH, M4_FORECASTING_LENGTH

DEEPAR_LEARNING_PARAMS = {'epochs': 150,
                          'lr': 0.001,
                          'device': 'cpu'
                          }
model_to_compare = [
    {0: ['smoothing', 'lagged', 'ridge']},
    {},
    {0: [('deepar_model', DEEPAR_LEARNING_PARAMS)]},
    {0: ['ar']}
]
model_name = ['lagged_regression', 'industrial', 'deepar', 'ar']
finutune_existed_model = [True, False, True, True]
BENCHMARK = 'M4'
EVALUATED = []
DATASETS = [x for x in M4_FORECASTING_BENCH if x not in EVALUATED]
BENCHMARK_PARAMS = {'experiment_date': '23_01_25',
                    'metadata': M4_FORECASTING_LENGTH,
                    'datasets': DATASETS,
                    'model_to_compare': (model_to_compare, model_name, finutune_existed_model)}
EVAL_REGIME = True

FORECASTING_BENCH = 'automl_univariate'
path = 'examples/real_world_examples/benchmark_example/forecasting/automl/shell/data/univariate_libra'
COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG

AUTOML_CONFIG = {'task': 'ts_forecasting',
                 'task_params': {'forecast_length': 14},
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'random',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
AUTOML_LEARNING_STRATEGY = dict(timeout=10,
                                n_jobs=4,
                                with_tuning=True,
                                pop_size=10,
                                logging_level=40)

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'rmse'}}

INDUSTRIAL_CONFIG = {'problem': 'ts_forecasting',
                     'task_params': {'forecast_length': 14}}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    api_agent = ApiTemplate(api_config=API_CONFIG, metric_list=('rmse', 'mae'))
    if EVAL_REGIME:
        EVALUATED = os.listdir('./M4_23_01_25/ar')
        DATASETS = [x for x in M4_FORECASTING_BENCH if x not in EVALUATED]
        BENCHMARK_PARAMS = {'experiment_date': '23_01_25',
                            'metadata': M4_FORECASTING_LENGTH,
                            'datasets': DATASETS,
                            'model_to_compare': (model_to_compare, model_name, finutune_existed_model)}
        api_agent.evaluate_benchmark(benchmark_name=BENCHMARK,
                                     benchmark_params=BENCHMARK_PARAMS)
