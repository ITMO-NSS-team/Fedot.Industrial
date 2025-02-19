from fedot.api.main import Fedot
from fedot.core.composer.metrics import F1
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from golem.core.tuning.sequential import SequentialTuner

from fedot_ind.core.models.ts_forecasting.lagged_strategy.eigen_forecaster import EigenAR
from fedot_ind.core.operation.dummy.dummy_operation import init_input_data
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
from fedot_ind.tools.loader import DataLoader


def test_fedot_multi_series():
    with IndustrialModels():
        train_data, test_data = initialize_multi_data()
        pipeline = EigenAR({})
        pipeline.fit(train_data)
        predict = pipeline.predict(test_data, output_mode='labels')
        print(F1.metric(test_data, predict))


def initialize_uni_data():
    train_data, test_data = DataLoader('Lightning7').load_data()
    train_input_data = init_input_data(train_data[0], train_data[1])
    test_input_data = init_input_data(test_data[0], test_data[1])
    return train_input_data, test_input_data


def initialize_multi_data():
    train_data, test_data = DataLoader('Epilepsy').load_data()
    train_input_data = init_input_data(train_data[0], train_data[1])
    test_input_data = init_input_data(test_data[0], test_data[1])
    return train_input_data, test_input_data


def test_industrial_uni_series():
    with IndustrialModels():
        train_data, test_data = initialize_uni_data()

        metrics = {}
        for extractor_name in ['topological_extractor',
                               'quantile_extractor',
                               # 'signal_extractor',
                               'recurrence_extractor']:
            pipeline = PipelineBuilder() \
                .add_node('eigen_basis') \
                .add_node(extractor_name) \
                .add_node('rf').build()
            model = Fedot(problem='classification', timeout=1,
                          initial_assumption=pipeline, n_jobs=1)
            model.fit(train_data)
            model.predict(test_data)
            model.get_metrics()
        print(metrics)


def test_tuner_industrial_uni_series():
    with IndustrialModels():
        train_data, test_data = initialize_uni_data()
        # search_space = SearchSpace(get_industrial_search_space(1))
        pipeline_builder = PipelineBuilder()
        pipeline_builder.add_node('eigen_basis')
        pipeline_builder.add_node('quantile_extractor')
        pipeline_builder.add_node('rf')

        pipeline = pipeline_builder.build()

        pipeline_tuner = TunerBuilder(train_data.task) \
            .with_tuner(SequentialTuner) \
            .with_timeout(2) \
            .with_iterations(2) \
            .build(train_data)

        pipeline = pipeline_tuner.tune(pipeline)

    pipeline.fit(train_data)
    pipeline.predict(test_data)
