import os.path
import numpy as np
import settings
from model.models import model
import utils
from data.dataset import LottoDataSet


def simulate(test_np_x, test_np_y):
    income = 0
    expenses = 0

    predictions = model.predict(test_np_x, batch_size=settings.BATCH_SIZE)

    # Total group of number
    samples_num = len(test_np_x['x1'])
    for j in range(samples_num):
        # The real result
        outputs = []
        for k in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            outputs.append(np.argmax(test_np_y['y{}'.format(k + 1)][j]))

        # Buy 5 group numbers every time
        expenses += 5
        for k in range(5):
            probabilities = []
            for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
                probabilities.append(predictions[i][j])

            balls = utils.select_seqs(probabilities)

            # Calculate income
            award = utils.lotto_calculate(outputs, balls)
            income += award

    print('For this buy, your expense is £{}, and winning £{}, and net income is £{}.'
          .format(expenses, income, income - expenses))
    return income - expenses


def train_and_show():
    dataset = LottoDataSet(train_data_rate=0.9)

    if not os.path.exists(settings.CHECKPOINTS_PATH):
        os.mkdir(settings.CHECKPOINTS_PATH)

    results = []

    for epoch in range(1, settings.EPOCHS + 1):
        model.fit(dataset.train_np_x, dataset.train_np_y, batch_size=settings.BATCH_SIZE, epochs=1)
        model.save_weights('{}/model_checkpoint_{}'.format(settings.CHECKPOINTS_PATH, epoch))
        print('{} round training gets completed，simulating...'.format(epoch))
        results.append(simulate(dataset.test_np_x, dataset.test_np_y))

    print(results)
    utils.draw_graph(results)
