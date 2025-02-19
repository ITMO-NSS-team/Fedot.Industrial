import os

import pandas as pd

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_CLF_AUTOML_CONFIG
from fedot_ind.core.repository.constanst_repository import UNI_CLF_BENCH
from fedot_ind.tools.serialisation.path_lib import PROJECT_PATH


def filter_datasets(UNI_CLF_BENCH, max_classes: int = 10, max_samples: int = 1000):
    UNI_CLF_BENCH_METADATA = pd.read_csv(PROJECT_PATH + '/fedot_ind/core/repository/data/ts_benchmark_metadata.csv')
    datasets_filtred_by_classes = UNI_CLF_BENCH_METADATA[UNI_CLF_BENCH_METADATA['Class'] <= max_classes]
    datasets_filtred_by_samples = datasets_filtred_by_classes[datasets_filtred_by_classes['Train ']
                                                              <= max_samples]
    datasets_filtred_by_samples = datasets_filtred_by_samples[datasets_filtred_by_samples['Test ']
                                                              <= max_samples]['Name'].values.tolist()
    already_eval = os.listdir('./UCR_UNI_23_01_25/rf')
    UNI_CLF_BENCH = [x for x in UNI_CLF_BENCH if x in datasets_filtred_by_samples and x not in already_eval]
    UNI_CLF_BENCH_METADATA = UNI_CLF_BENCH_METADATA[UNI_CLF_BENCH_METADATA['Name'].isin(datasets_filtred_by_samples)]
    return UNI_CLF_BENCH, UNI_CLF_BENCH_METADATA


model_to_compare = [{0: ['quantile_extractor', 'rf']},
                    {0: ['quantile_extractor', 'pdl_clf']}
                    ]
model_name = ['rf', 'pdl_rf']
finutune_existed_model = [True, True]
BENCHMARK = 'UCR_UNI'
UNI_CLF_BENCH, UNI_CLF_BENCH_METADATA = filter_datasets(UNI_CLF_BENCH)
BENCHMARK_PARAMS = {'experiment_date': '23_01_25',
                    'metadata': None,
                    'datasets': UNI_CLF_BENCH,
                    'model_to_compare': (model_to_compare, model_name, finutune_existed_model)}
METRIC_NAMES = ('f1', 'accuracy', 'precision', 'roc_auc')
EVAL_REGIME = True

COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_LEARNING_STRATEGY = dict(timeout=2,
                                pop_size=10,
                                n_jobs=-1,
                                num_of_generations=15)

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'accuracy'}}

INDUSTRIAL_CONFIG = {'problem': 'classification'}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    api_agent = ApiTemplate(api_config=API_CONFIG, metric_list=METRIC_NAMES)
    if EVAL_REGIME:
        api_agent.evaluate_benchmark(benchmark_name=BENCHMARK,
                                     benchmark_params=BENCHMARK_PARAMS)
