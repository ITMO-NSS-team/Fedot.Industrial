import numpy as np


def create_big_dataset(dataset_name, fold):
    fold_number = f'fold{fold}'
    train_X, test_X = np.load(
        f'big_dataset/{dataset_name}/{fold_number}/train_{dataset_name}_{fold_number}.npy'), \
        np.load(f'big_dataset/{dataset_name}/{fold_number}/test_{dataset_name}_{fold_number}.npy')
    train_y, test_y = np.load(
        f'big_dataset/{dataset_name}/{fold_number}/trainy_{dataset_name}_{fold_number}.npy'), \
        np.load(f'big_dataset/{dataset_name}/{fold_number}/testy_{dataset_name}_{fold_number}.npy')
    dataset_dict = dict(train_data=(train_X, train_y),
                        test_data=(test_X, test_y))
    return dataset_dict
