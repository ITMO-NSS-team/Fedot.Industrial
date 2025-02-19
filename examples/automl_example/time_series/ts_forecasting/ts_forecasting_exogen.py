import numpy as np

from examples.example_utils import load_monash_dataset
from fedot_ind.api.main import FedotIndustrial
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, DEFAULT_TSF_AUTOML_CONFIG

if __name__ == "__main__":
    HORIZON = 60
    METRIC_NAMES = ('smape', 'rmse', 'median_absolute_error')

    train_data = load_monash_dataset('bitcoin')
    exog_var = ['send_usd', 'market_cap', 'median_transaction_value', 'google_trends']
    exog_ts = np.vstack([train_data[column].values for column in exog_var])
    exog_ts = exog_ts[0, :]
    ts = train_data['price'].values
    target = ts[-HORIZON:].flatten()
    input_data = (ts, target)

    TASK_PARAMS = {'forecast_length': HORIZON}
    AUTOML_LEARNING_STRATEGY = dict(timeout=3,
                                    with_tuning=False,
                                    n_jobs=2,
                                    pop_size=10,
                                    logging_level=30)

    API_CONFIG = {'industrial_config': {'problem': 'ts_forecasting',
                                        'task_params': TASK_PARAMS,
                                        'strategy': 'forecasting_exogenous',
                                        'strategy_params': {'exog_variable': exog_ts,
                                                            'data_type': 'time_series'}},
                  'automl_config': {'task_params': TASK_PARAMS,
                                    **DEFAULT_TSF_AUTOML_CONFIG},
                  'learning_config': {'learning_strategy': 'from_scratch',
                                      'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                                      'optimisation_loss': {'quality_loss': 'rmse'}},
                  'compute_config': DEFAULT_COMPUTE_CONFIG}

    industrial = FedotIndustrial(**API_CONFIG)
    industrial.fit(input_data)
