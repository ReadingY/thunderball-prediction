import numpy as np
import os.path
import requests
import settings
import time


def request_data():
    try:
        r = requests.get(settings.LOTTO_DOWNLOAD_PATH)

        if r.status_code == 200:
            with open(settings.DATA_PATH, 'wb') as f:
                f.write(r.content)
    except Exception as e:
        print(e)
        

class LottoDataSet:

    def __init__(self, path=settings.DATA_PATH, train_data_rate=0.9, shuffle=True):
        self.path = path
        self.train_data_rate = train_data_rate
        self.shuffle = shuffle

        self.test_np_y = {}
        self.test_np_x = {}
        self.train_np_y = {}
        self.train_np_x = {}

        self.clean_data()

    def load_data(self, path=None):
        if not path:
            path = self.path

        if not os.path.exists(path):
            request_data()

        with open(path) as f:
            lines = f.readlines()[1:][::-1]
            lines = [line.strip() for line in lines if line.strip()]

        return lines

    def clean_data(self):
        lines = self.load_data()
        x_nums = []
        for line in lines:
            nums = line.split(',')[1:7]
            x_nums.append([int(x) - 1 for x in nums])

        num_seqs = {}
        for line in x_nums:
            for index, num in enumerate(line):
                num_seqs.setdefault(index, []).append(num)

        x = {}
        y = {}
        for index, seqs in num_seqs.items():
            x[index] = []
            y[index] = []

            total = len(seqs)
            for i in range(settings.MAX_STEPS, total, 1):
                tmp_x = []

                if index < settings.FRONT_SIZE:
                    tmp_y = [0 for _ in range(settings.FRONT_NUMBERS_RANGE)]
                else:
                    tmp_y = [0 for _ in range(settings.BACK_THUNDERBALL_RANGE)]

                for j in range(i - settings.MAX_STEPS, i, 1):
                    tmp_x.append(seqs[j])

                x[index].append(tmp_x)
                tmp_y[seqs[i]] = 1
                print(seqs[i])
                y[index].append(tmp_y)

        np_x = {}
        np_y = {}
        for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            x_len = len(x[i])

            if i < settings.FRONT_SIZE:
                tmp_x = np.zeros((x_len, settings.MAX_STEPS, settings.FRONT_NUMBERS_RANGE))
                tmp_y = np.zeros((x_len, settings.FRONT_NUMBERS_RANGE))
            else:
                tmp_x = np.zeros((x_len, settings.MAX_STEPS, settings.BACK_THUNDERBALL_RANGE))
                tmp_y = np.zeros((x_len, settings.BACK_THUNDERBALL_RANGE))

            for j in range(x_len):
                for k, num in enumerate(x[i][j]):
                    tmp_x[j][k][num] = 1

                for k, num in enumerate(y[i][j]):
                    tmp_y[j][k] = num

            np_x['x{}'.format(i + 1)] = tmp_x
            np_y['y{}'.format(i + 1)] = tmp_y

        total_batch = len(np_x['x1'])
        train_batch_num = int(total_batch * self.train_data_rate)
        train_np_x = {}
        train_np_y = {}
        test_np_x = {}
        test_np_y = {}
        for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            x_index = 'x{}'.format(i + 1)
            y_index = 'y{}'.format(i + 1)
            train_np_x[x_index] = np_x[x_index][:train_batch_num]
            train_np_y[y_index] = np_y[y_index][:train_batch_num]
            test_np_x[x_index] = np_x[x_index][train_batch_num:]
            test_np_y[y_index] = np_y[y_index][train_batch_num:]

        if self.shuffle:
            random_seed = int(time.time())

            for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
                np.random.seed(random_seed)
                np.random.shuffle(train_np_x['x{}'.format(i + 1)])
                np.random.seed(random_seed)
                np.random.shuffle(train_np_y['y{}'.format(i + 1)])
        self.train_np_x = train_np_x
        self.train_np_y = train_np_y
        self.test_np_x = test_np_x
        self.test_np_y = test_np_y

    @property
    def predict_data(self):
        lines = self.load_data()

        x_nums = []
        for line in lines:
            nums = line.split(',')[1:7]
            x_nums.append([int(x) - 1 for x in nums])

        num_seqs = {}
        for line in x_nums:
            for index, num in enumerate(line):
                num_seqs.setdefault(index, []).append(num)

        x = {}
        for index, seqs in num_seqs.items():
            x[index] = []
            total = len(seqs)

            tmp_x = []
            for j in range(total - settings.MAX_STEPS, total, 1):
                tmp_x.append(seqs[j])
            x[index].append(tmp_x)

        np_x = {}
        for i in range(settings.FRONT_SIZE + settings.BACK_SIZE):
            x_len = len(x[i])

            if i < settings.FRONT_SIZE:
                tmp_x = np.zeros((x_len, settings.MAX_STEPS, settings.FRONT_NUMBERS_RANGE))
            else:
                tmp_x = np.zeros((x_len, settings.MAX_STEPS, settings.BACK_THUNDERBALL_RANGE))

            for j in range(x_len):
                for k, num in enumerate(x[i][j]):
                    tmp_x[j][k][num] = 1
            np_x['x{}'.format(i + 1)] = tmp_x
        return np_x
