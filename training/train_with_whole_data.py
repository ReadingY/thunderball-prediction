import os.path
import settings
from data.dataset import LottoDataSet
from model.models import model


def train_whole_data():
    dataset = LottoDataSet(train_data_rate=1)

    if not os.path.exists(settings.CHECKPOINTS_PATH):
        os.mkdir(settings.CHECKPOINTS_PATH)

    model.fit(dataset.train_np_x, dataset.train_np_y)

    model.save_weights('{}/model_checkpoint_x'.format(settings.CHECKPOINTS_PATH))
