import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fedot_ind.api.main import FedotIndustrial
from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.repository.constanst_repository import KERNEL_BASELINE_FEATURE_GENERATORS
from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


def generate_composite_features(input_data):
    IndustrialModels().setup_repository()
    feature_matrix = []
    for model_name in KERNEL_BASELINE_FEATURE_GENERATORS.keys():
        model = KERNEL_BASELINE_FEATURE_GENERATORS[model_name]
        model.heads[0].parameters['use_sliding_window'] = False
        model = model.build()
        feature_matrix.append(model.fit(input_data).predict)
    return feature_matrix


if __name__ == "__main__":
    sig_X = np.load('sig_data.npy').swapaxes(1, 2)
    sig_y = np.load('sig_target.npy')
    metric_names = ('f1', 'accuracy')
    scaler = StandardScaler()
    pca = PCA(.975)
    industrial = FedotIndustrial(problem='classification',
                                 use_cache=False,
                                 preset='classification_tabular',
                                 timeout=15,
                                 n_jobs=-1,
                                 logging_level=20)

    x_train, x_test, y_train, y_test = train_test_split(sig_X, sig_y, test_size=0.2, random_state=42)
    train_data, test_data = (x_train, y_train), (x_test, y_test)

    input_train = DataCheck(input_data=train_data, task='classification', fit_stage=True)
    input_data_train = input_train.check_input_data()

    input_test = DataCheck(input_data=test_data, task='classification', fit_stage=True)
    input_data_test = input_test.check_input_data()

    composite_feature_train = generate_composite_features(input_data_train)
    composite_feature_test = generate_composite_features(input_data_test)
    feature_matrix_train = np.concatenate([x.reshape(x.shape[0], x.shape[1] * x.shape[2])
                                           for x in composite_feature_train], axis=1)
    feature_matrix_test = np.concatenate([x.reshape(x.shape[0], x.shape[1] * x.shape[2])
                                          for x in composite_feature_test], axis=1)

    feature_matrix_train = pca.fit_transform(scaler.fit_transform(feature_matrix_train, y_train))
    feature_matrix_test = pca.transform(scaler.transform(feature_matrix_test))

    train_data, test_data = (feature_matrix_train, y_train), (feature_matrix_test, y_test)

    industrial.fit(train_data)
    labels = industrial.predict(test_data)
    industrial.predict_proba(test_data)
    metrics = industrial.get_metrics(target=test_data[1],
                                     rounding_order=3,
                                     metric_names=metric_names)
    _ = 1
