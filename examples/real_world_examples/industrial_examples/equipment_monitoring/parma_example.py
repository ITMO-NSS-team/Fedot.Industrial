import gc

import matplotlib
import numpy as np
from sklearn.utils import shuffle

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.operation.transformation.data.park_transformation import park_transform
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG

matplotlib.use('TkAgg')
gc.collect()


def load_data(use_park_transform: bool = False):
    train_features, train_target = np.load('./dataset/X_train.npy').swapaxes(1, 2), np.load('./dataset/y_train.npy')
    test_features, test_target = np.load('./dataset/X_test.npy').swapaxes(1, 2), np.load('./dataset/y_test.npy')
    train_features, train_target = shuffle(train_features, train_target)
    if use_park_transform:
        train_features, test_features = park_transform(train_features), park_transform(test_features)
    input_train = (train_features, train_target)
    input_test = (test_features, test_target)

    dataset = dict(test_data=input_test, train_data=input_train)
    return dataset


def create_feature_generator_strategy():
    stat_params = {'window_size': 0, 'stride': 1, 'add_global_features': True,
                   'channel_independent': False, 'use_sliding_window': False}
    fourier_params = {'low_rank': 5, 'output_format': 'signal', 'compute_heuristic_representation': True,
                      'approximation': 'smooth', 'threshold': 0.9, 'sampling_rate': 64e3}
    wavelet_params = {'n_components': 3, 'wavelet': 'bior3.7', 'compute_heuristic_representation': True}
    rocket_params = {"num_features": 200}
    sampling_dict = dict(samples=dict(start_idx=0,
                                      end_idx=None),
                         channels=dict(start_idx=0,
                                       end_idx=None),
                         elements=dict(start_idx=0,
                                       end_idx=None))
    feature_generator = {
        # 'minirocket': [('minirocket_extractor', rocket_params)],
        'stat_generator': [('quantile_extractor', stat_params)],
        'fourier': [('fourier_basis', fourier_params)],
        'wavelet': [('wavelet_basis', wavelet_params)],
    }
    return feature_generator, sampling_dict


feature_generator, sampling_dict = create_feature_generator_strategy()

INDUSTRIAL_PARAMS = {'feature_generator': feature_generator,
                     'data_type': 'tensor',
                     'learning_strategy': 'ts2tabular',
                     'sampling_strategy': sampling_dict
                     }

# DEFINE ALL CONFIG FOR API
AUTOML_LEARNING_STRATEGY = DEFAULT_AUTOML_LEARNING_CONFIG
COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'classification',
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'f1'}}
INDUSTRIAL_CONFIG = {'problem': 'classification',
                     'strategy': 'tabular',
                     'strategy_params': INDUSTRIAL_PARAMS
                     }
API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')
    dataset = load_data(use_park_transform=True)
    result_dict = ApiTemplate(api_config=API_CONFIG, metric_list=metric_names).eval(dataset=dataset)
    metrics = result_dict['metrics']
    metrics.to_csv('./metrics.csv')
    hist = result_dict['industrial_model'].save_optimization_history(return_history=True)
    result_dict['industrial_model'].vis_optimisation_history(hist)
    result_dict['industrial_model'].save_best_model()
    result_dict['industrial_model'].solver.current_pipeline.show()
    _ = 1
