import torchvision.datasets as datasets
import torchvision.transforms as transforms
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG, DEFAULT_CLF_AUTOML_CONFIG
from fedot_ind.tools.serialisation.path_lib import EXAMPLES_DATA_PATH


if __name__ == '__main__':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST train and test dataset
    train_data = (datasets.MNIST(
        root=EXAMPLES_DATA_PATH,
        train=True,
        download=True,
        transform=transform), 'torchvision_dataset')

    test_data = (datasets.MNIST(
        root=EXAMPLES_DATA_PATH,
        train=False,
        download=True,
        transform=transform), 'torchvision_dataset')

    METRIC_NAMES = ('f1', 'accuracy', 'precision', 'roc_auc')

    DEFAULT_AUTOML_LEARNING_CONFIG['timeout'] = 0.1
    AUTOML_LEARNING_STRATEGY = DEFAULT_AUTOML_LEARNING_CONFIG
    COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
    AUTOML_CONFIG = DEFAULT_CLF_AUTOML_CONFIG

    LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                       'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                       'optimisation_loss': {'quality_loss': 'accuracy'}}

    INDUSTRIAL_PARAMS = {'rank': 2,
                         'sampling_share': 0.5,
                         'lora_init': 'random',
                         'epochs': 1,
                         'batch_size': 10,
                         'data_type': 'tensor'
                         }

    INDUSTRIAL_CONFIG = {'problem': 'classification',
                         'strategy': 'lora_strategy',
                         'strategy_params': INDUSTRIAL_PARAMS
                         }

    API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
                  'automl_config': AUTOML_CONFIG,
                  'learning_config': LEARNING_CONFIG,
                  'compute_config': COMPUTE_CONFIG}

    dataset_dict = dict(test_data=(test_data[0].data.numpy(), test_data[0].targets.numpy()),
                        train_data=(train_data[0].data.numpy(), train_data[0].targets.numpy()))

    industrial = ApiTemplate(api_config=API_CONFIG,
                             metric_list=METRIC_NAMES).eval(dataset=dataset_dict)
    industrial.fit(train_data)
    predict = industrial.predict(test_data)
