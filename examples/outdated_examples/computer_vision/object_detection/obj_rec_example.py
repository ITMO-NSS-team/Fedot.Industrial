import os
import random

import yaml
from fedot_ind.core.architecture.datasets.visualization import draw_sample_with_bboxes

from fedot_ind.api.main import FedotIndustrial

DATASETS_PATH = os.path.abspath('Warp-D')
TEST_IMAGE_FOLDER = 'Land-Use_Scene_Classification/images_train_test_val/test'
NUM_CLASSES = 29
TASK = 'object_detection'

model_dict = {'basic': FedotIndustrial(task=TASK, num_classes=NUM_CLASSES)}


def run_industrial_model(model_type: str = 'basic'):
    with open(os.path.join(DATASETS_PATH, 'classes.txt'), 'r') as f:
        classes = f.read().splitlines()
    config = {
        'train': 'train/images',
        'val': 'test/images',
        'nc': 28,
        'names': classes
    }
    with open(os.path.join(DATASETS_PATH, 'warp.yaml'), 'w') as f:
        yaml.dump(config, f)

    fed = model_dict[model_type]
    trained_model = fed.fit(dataset_path=os.path.join(
        DATASETS_PATH, 'warp.yaml'), dataset_name='WaRP', )
    predict = fed.predict(data_path=os.path.join(DATASETS_PATH, 'test/images'))
    predict_proba = fed.predict_proba(
        data_path=os.path.join(DATASETS_PATH, 'test/images'))
    image = random.choice(list(predict.keys()))
    fig = draw_sample_with_bboxes(
        image=image,
        target=predict[image],
        prediction=predict_proba[image],
        threshold=0.2)

    return trained_model


if __name__ == "__main__":
    basic_model = run_industrial_model('basic')
