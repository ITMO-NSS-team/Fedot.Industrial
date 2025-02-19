from examples.example_utils import create_feature_generator_strategy
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG, DEFAULT_CLF_AUTOML_CONFIG


feature_generator, sampling_dict = create_feature_generator_strategy()

INDUSTRIAL_PARAMS = {'feature_generator': feature_generator,
                     'data_type': 'tensor',
                     'learning_strategy': 'all_classes',
                     'head_model': 'rf',
                     'sampling_strategy': sampling_dict}

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': DEFAULT_AUTOML_LEARNING_CONFIG,
                   'optimisation_loss': {'quality_loss': 'f1'}}
INDUSTRIAL_CONFIG = {'problem': 'classification',
                     'strategy': 'kernel_automl',
                     'strategy_params': INDUSTRIAL_PARAMS}
API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': DEFAULT_COMPUTE_CONFIG}

DATASET_NAME = 'Lightning7'
METRIC_NAMES = ('f1', 'accuracy')

if __name__ == "__main__":
    result_dict = ApiTemplate(api_config=API_CONFIG,
                              metric_list=METRIC_NAMES).eval(dataset=DATASET_NAME)
    print(result_dict['metrics'])
