from examples.automl_example.custom_strategy.big_data.big_dataset_utils import create_big_dataset
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_AUTOML_LEARNING_CONFIG, \
    DEFAULT_CLF_AUTOML_CONFIG

DATASET_NAME = 'airlines'
METRIC_NAMES = ('f1', 'accuracy', 'precision', 'roc_auc')

INDUSTRIAL_PARAMS = {'data_type': 'tensor',
                     'learning_strategy': 'big_dataset',
                     'sampling_strategy': {'CUR': {'rank': None}}}

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': DEFAULT_AUTOML_LEARNING_CONFIG,
                   'optimisation_loss': {'quality_loss': 'f1'}}

INDUSTRIAL_CONFIG = {'problem': 'classification',
                     'strategy': 'tabular',
                     'strategy_params': INDUSTRIAL_PARAMS}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': DEFAULT_COMPUTE_CONFIG}

if __name__ == "__main__":
    metric_by_fold = {}
    for fold in range(10):
        dataset_dict = create_big_dataset(DATASET_NAME, fold)
        result_dict = ApiTemplate(api_config=API_CONFIG,
                                  metric_list=METRIC_NAMES).eval(dataset=dataset_dict,
                                                                 finetune=False)
        metric_by_fold.update({fold: result_dict})
    print(metric_by_fold)
