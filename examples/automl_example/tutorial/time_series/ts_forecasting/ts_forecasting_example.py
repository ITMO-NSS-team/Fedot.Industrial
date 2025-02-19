from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG

dataset_name = {'benchmark': 'M4',
                'dataset': 'D3257',
                'task_params': {'forecast_length': 14}}

pipeline_for_finetune = {0: ['ar'],
                         1: ['lagged', 'ridge'],
                         'head': ['bagging']}
finutune_existed_model = False

COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'ts_forecasting',
                 'task_params': {'forecast_length': 14},
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
AUTOML_LEARNING_STRATEGY = dict(timeout=5,
                                n_jobs=2,
                                logging_level=30)

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
    result_dict = ApiTemplate(
        api_config=API_CONFIG,
        metric_list=('rmse', 'mae')).eval(
        dataset=dataset_name,
        initial_assumption=pipeline_for_finetune,
        finetune=finutune_existed_model)
    current_metric = result_dict['metrics']
