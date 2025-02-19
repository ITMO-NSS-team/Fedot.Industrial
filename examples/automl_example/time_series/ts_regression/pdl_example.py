from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_REG_AUTOML_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG


if __name__ == "__main__":
    DATASET_NAME = 'AppliancesEnergy'

    API_CONFIG = {'industrial_config': {'problem': 'regression'},
                  'automl_config': DEFAULT_REG_AUTOML_CONFIG,
                  'learning_config': {'learning_strategy': 'from_scratch',
                                      'learning_strategy_params': DEFAULT_AUTOML_LEARNING_CONFIG,
                                      'optimisation_loss': {'quality_loss': 'rmse'}},
                  'compute_config': DEFAULT_COMPUTE_CONFIG}

    METRIC_NAMES = ('r2', 'rmse', 'mae')
    COMPARISON_DICT = dict(pairwise_approach={0: ['quantile_extractor', 'pdl_reg']},
                           baseline={0: ['quantile_extractor', 'treg']})

    for approach, node_list in COMPARISON_DICT.items():
        result_dict = ApiTemplate(api_config=API_CONFIG,
                                  metric_list=METRIC_NAMES).eval(dataset=DATASET_NAME,
                                                                 initial_assumption=node_list,
                                                                 finetune=False)
        print(f'Approach: {approach}. Metrics: {result_dict["metrics"]}')
