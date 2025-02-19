from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_CLF_AUTOML_CONFIG

DATASET_NAME = 'Handwriting'
METRIC_NAMES = ('f1', 'accuracy', 'precision', 'roc_auc')

COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = DEFAULT_CLF_AUTOML_CONFIG
AUTOML_LEARNING_STRATEGY = dict(timeout=3,
                                pop_size=10,
                                n_jobs=2)

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'f1'}}

INDUSTRIAL_CONFIG = {'problem': 'classification'}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    result_dict = ApiTemplate(api_config=API_CONFIG,
                              metric_list=METRIC_NAMES).eval(dataset=DATASET_NAME, finetune=False)
    print(result_dict['metrics'])
