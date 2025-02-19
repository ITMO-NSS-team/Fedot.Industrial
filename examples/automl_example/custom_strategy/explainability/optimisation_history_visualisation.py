from examples.example_utils import create_feature_generator_strategy
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG, DEFAULT_CLF_AUTOML_CONFIG


feature_generator, sampling_dict = create_feature_generator_strategy()
METRIC_NAMES = ('f1', 'accuracy', 'precision', 'roc_auc')
INDUSTRIAL_PARAMS = {'feature_generator': feature_generator,
                     'data_type': 'tensor',
                     'learning_strategy': 'ts2tabular',
                     'sampling_strategy': sampling_dict}

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
    result_dict = ApiTemplate(api_config=API_CONFIG,
                              metric_list=METRIC_NAMES).eval(dataset='Lightning7',
                                                             finetune=False)
    metrics = result_dict['metrics']
    metrics.to_csv('./metrics.csv')
    hist = result_dict['industrial_model'].save_optimization_history(return_history=True)
    result_dict['industrial_model'].vis_optimisation_history(hist)
    result_dict['industrial_model'].save_best_model()
    result_dict['industrial_model'].solver.current_pipeline.show()
    _ = 1
