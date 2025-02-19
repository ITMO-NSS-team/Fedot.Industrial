from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG
from fedot_ind.core.repository.constanst_repository import MULTI_CLF_BENCH

model_to_compare = [{0: ['quantile_extractor', 'rf']},
                    {0: ['quantile_extractor', 'pdl_clf']}
                    ]
model_name = ['rf', 'pdl_rf']
finutune_existed_model = [True, True]
BENCHMARK = 'UCR_MULTI'
BENCHMARK_PARAMS = {'experiment_date': '22_01_25',
                    'metadata': None,
                    'datasets': MULTI_CLF_BENCH,
                    'model_to_compare': (model_to_compare, model_name, finutune_existed_model)}
METRIC_NAMES = ('f1', 'accuracy', 'precision', 'roc_auc')
EVAL_REGIME = True

COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'classification',
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}
AUTOML_LEARNING_STRATEGY = dict(timeout=2,
                                pop_size=10,
                                n_jobs=-1,
                                num_of_generations=15)

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'accuracy'}}

INDUSTRIAL_CONFIG = {'problem': 'classification'}

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

if __name__ == "__main__":
    api_agent = ApiTemplate(api_config=API_CONFIG, metric_list=METRIC_NAMES)
    if EVAL_REGIME:
        api_agent.evaluate_benchmark(benchmark_name=BENCHMARK,
                                     benchmark_params=BENCHMARK_PARAMS)
