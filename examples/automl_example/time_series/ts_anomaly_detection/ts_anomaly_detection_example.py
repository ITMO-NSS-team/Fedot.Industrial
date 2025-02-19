from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG

DATASET_NAME = {'benchmark': 'valve1',
                'dataset': '1'}
METRIC_NAMES = ('nab', 'accuracy')

COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'classification',
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
AUTOML_LEARNING_STRATEGY = dict(timeout=1,
                                n_jobs=2,
                                pop_size=10,
                                logging_level=0)

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'accuracy'}}

INDUSTRIAL_CONFIG = {'problem': 'anomaly_detection',
                     'strategy_params': {'detection_window': 10,
                                         'data_type': 'time_series'}}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    result_dict = ApiTemplate(api_config=API_CONFIG,
                              metric_list=METRIC_NAMES).eval(dataset=DATASET_NAME,
                                                             finetune=False)
    print(result_dict['metrics'])
