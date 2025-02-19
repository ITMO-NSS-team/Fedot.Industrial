from examples.example_utils import create_feature_generator_strategy
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG, DEFAULT_CLF_AUTOML_CONFIG

feature_generator, sampling_dict = create_feature_generator_strategy()

INDUSTRIAL_PARAMS = {'feature_generator': feature_generator,
                     'data_type': 'tensor',
                     'learning_strategy': 'ts2tabular',
                     'sampling_strategy': sampling_dict
                     }

DATASET_NAME = 'Lightning7'
METRIC_NAMES = ('f1', 'accuracy')

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': DEFAULT_AUTOML_LEARNING_CONFIG,
                   'optimisation_loss': {'quality_loss': 'f1'}}
INDUSTRIAL_CONFIG = {'problem': 'classification',
                     'strategy': 'tabular',
                     'strategy_params': INDUSTRIAL_PARAMS
                     }
API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': DEFAULT_COMPUTE_CONFIG}

if __name__ == "__main__":
    multimodal_pipeline = {0: [
        # ('recurrence_extractor', {'window_size': 30, 'stride': 5, 'image_mode': True}),
        ('quantile_extractor', {'window_size': 30, 'stride': 5, 'image_mode': True}),
        ('resnet_model', {'epochs': 1, 'batch_size': 16, 'model_name': 'ResNet50'})
    ]}

    explain_config = {'method': 'recurrence',
                      'samples': 1,
                      'metric': 'mean'}

    result_dict = ApiTemplate(api_config=API_CONFIG,
                              metric_list=METRIC_NAMES).eval(dataset=DATASET_NAME,
                                                             finetune=False,
                                                             initial_assumption=multimodal_pipeline)
    result_dict['industrial_model'].explain(explain_config)
