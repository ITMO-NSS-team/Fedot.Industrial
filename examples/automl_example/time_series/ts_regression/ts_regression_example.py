from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_AUTOML_LEARNING_CONFIG, DEFAULT_REG_AUTOML_CONFIG, \
    DEFAULT_COMPUTE_CONFIG

if __name__ == "__main__":
    dataset_name = 'AppliancesEnergy'
    METRIC_NAMES = ('r2', 'rmse', 'mae')
    API_CONFIG = {'industrial_config': {'problem': 'regression'},
                  'automl_config': DEFAULT_REG_AUTOML_CONFIG,
                  'learning_config': {'learning_strategy': 'from_scratch',
                                      'learning_strategy_params': DEFAULT_AUTOML_LEARNING_CONFIG,
                                      'optimisation_loss': {'quality_loss': 'rmse'}},
                  'compute_config': DEFAULT_COMPUTE_CONFIG}

    result_dict = ApiTemplate(api_config=API_CONFIG,
                              metric_list=METRIC_NAMES).eval(dataset=dataset_name,
                                                             finetune=False)
    print(result_dict['metrics'])
