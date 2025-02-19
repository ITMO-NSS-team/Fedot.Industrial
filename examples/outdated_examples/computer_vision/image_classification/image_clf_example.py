import random

import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Resize, Compose

from fedot_ind.api.main import FedotIndustrial

DATASETS_PATH = 'Land-Use_Scene_Classification/images'
TEST_IMAGE_FOLDER = 'Land-Use_Scene_Classification/images_train_test_val/test'
NUM_CLASSES = 21
TASK = 'image_classification'

model_dict = {
    'basic': FedotIndustrial(
        task=TASK,
        num_classes=NUM_CLASSES),
    'advanced': FedotIndustrial(
        task=TASK,
        num_classes=NUM_CLASSES,
        optimization='svd',
        optimization_params={
            'energy_thresholds': [0.99]})}


def run_industrial_model(model_type: str = 'basic'):
    fed = model_dict[model_type]

    trained_model = fed.fit(dataset_path=DATASETS_PATH, transform=Compose(
        [ToTensor(), Resize((256, 256), antialias=None)]))

    predict = fed.predict(data_path=TEST_IMAGE_FOLDER, transform=Compose(
        [ToTensor(), Resize((256, 256), antialias=None)]))

    plt.figure(figsize=(20, 10))
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        image_path, prediction = random.choice(list(predict.items()))
        image = plt.imread(image_path)
        plt.imshow(image)
        plt.title(prediction)
    plt.show()
    return trained_model


if __name__ == "__main__":
    basic_model = run_industrial_model('basic')
    advanced_model = run_industrial_model('advanced')
