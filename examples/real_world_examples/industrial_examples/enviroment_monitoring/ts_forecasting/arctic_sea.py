from pathlib import Path

from matplotlib import use
import matplotlib.pyplot as plt
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.tools.serialisation.path_lib import EXAMPLES_DATA_PATH


if __name__ == '__main__':
    use('TKagg')
    horizon = 365

    time_series_df = pd.read_csv(Path(EXAMPLES_DATA_PATH,
                                      'real_world/ice_forecasting/ices_areas_ts.csv'))
    time_series_df = time_series_df.iloc[:, 1:]
    target_series = time_series_df['Карское'].values

    input_data = InputData.from_numpy_time_series(
        target_series,
        task=Task(TaskTypesEnum.ts_forecasting,
                  task_params=TsForecastingParams(forecast_length=horizon)))
    train_data, test_data = train_test_data_setup(input_data)

    pipeline_based = (
        PipelineBuilder()
        .add_node('lagged')
        .add_node('rfr')
        .build()
    )
    pipeline_based.fit(train_data)

    topological_pipeline = (
        PipelineBuilder()
        .add_node('lagged')
        .add_node('topological_features')
        .add_node('lagged', branch_idx=2)
        .join_branches('rfr')
        .build()
    )
    topological_pipeline.fit(train_data)

    forecast_base = np.ravel(pipeline_based.predict(test_data).predict)
    forecast_topo = np.ravel(topological_pipeline.predict(test_data).predict)

    forecast_base[forecast_base < 0] = 0
    forecast_topo[forecast_topo < 0] = 0

    plt.plot(input_data.features, label='real data')
    plt.plot(np.arange(len(target_series) - horizon, len(target_series)),
             forecast_base, label='forecast base')
    plt.plot(np.arange(len(target_series) - horizon, len(target_series)),
             forecast_topo, label='forecast topo')

    plt.grid()
    plt.legend()
    plt.show()

    print('base')
    print(mean_squared_error(test_data.target, forecast_base, squared=False))
    print(mean_absolute_percentage_error(test_data.target + 1000, forecast_base + 1000))

    print('topo')
    print(mean_squared_error(test_data.target, forecast_topo, squared=False))
    print(mean_absolute_percentage_error(test_data.target + 1000, forecast_topo + 1000))
