import numpy as np

from examples.example_utils import create_feature_generator_strategy
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG


if __name__ == "__main__":
    DATASET_NAME = 'Libras'

    feature_generator, sampling_dict = create_feature_generator_strategy()
    DEFAULT_AUTOML_LEARNING_CONFIG['timeout'] = 0.1

    AUTOML_CONFIG = {'task': 'classification'}
    LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                       'learning_strategy_params': DEFAULT_AUTOML_LEARNING_CONFIG,
                       'optimisation_loss': {'quality_loss': 'f1'}}
    INDUSTRIAL_CONFIG = {'problem': 'classification',
                         'strategy': 'tabular',
                         'strategy_params': {'feature_generator': feature_generator,
                                             'data_type': 'tensor',
                                             'learning_strategy': 'ts2tabular',
                                             'sampling_strategy': sampling_dict}}

    API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
                  'automl_config': AUTOML_CONFIG,
                  'learning_config': LEARNING_CONFIG,
                  'compute_config': DEFAULT_COMPUTE_CONFIG}

    api_client = ApiTemplate(api_config=API_CONFIG,
                             metric_list=('f1', 'accuracy'))
    result_dict = api_client.eval(dataset=DATASET_NAME, finetune=False)
    uncalibrated_labels, uncalibrated_probs = result_dict['industrial_model'].manager.predicted_labels, \
        result_dict['industrial_model'].manager.predicted_probs
    calibrated_probs = result_dict['industrial_model'].predict_proba(predict_data=api_client.test_data,
                                                                     calibrate_probs=True)
    calibrated_labels = np.argmax(calibrated_probs, axis=1) + np.min(uncalibrated_labels)
