import pytest

from fedot_ind.api.utils.checkers_collections import DataCheck
from fedot_ind.core.architecture.settings.computational import backend_methods as np


def get_corrupted_input_data(fill_value):
    array = np.random.rand(10, 10)
    for i in range(10):
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        array[x, y] = fill_value
    target = np.random.rand(10)
    return array, target


@pytest.mark.parametrize("input_data", (
    get_corrupted_input_data(np.nan),
    get_corrupted_input_data(np.inf)
), ids=['nan_filled', 'inf_filled'])
def test_data_check(input_data):
    features, target = input_data
    data_check = DataCheck(input_data=(features, target),
                           task='classification')
    clean_data = data_check.check_input_data()
    assert clean_data is not None
