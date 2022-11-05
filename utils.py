import numpy as np
import settings
import matplotlib.pyplot as plt


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype(np.float64)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def search_award(front_match_num, back_match_num, cache=None):
    if cache is None:
        cache = {}
    if front_match_num == 0 and back_match_num == 0:
        return 0

    award = cache.get((front_match_num, back_match_num), -1)

    if award != -1:
        return award

    award = settings.WINNING_RULES.get((front_match_num, back_match_num), -1)
    if award == -1:
        award = 0
        if front_match_num > 0:
            award = search_award(front_match_num - 1, back_match_num)
        if back_match_num > 0:
            award = max(award, search_award(front_match_num, back_match_num - 1))

    cache[(front_match_num, back_match_num)] = award

    return award


def lotto_calculate(winning_sequence, sequence_selected):
    front_match = len(
        set(winning_sequence[:settings.FRONT_SIZE]).intersection(set(sequence_selected[:settings.FRONT_SIZE])))

    back_match = len(
        set(winning_sequence[settings.FRONT_SIZE:]).intersection(set(sequence_selected[settings.FRONT_SIZE:])))

    award = search_award(front_match, back_match)
    return award


def select_seqs(predicts):
    balls = []

    for predict in predicts:
        try_cnt = 0
        while True:
            try_cnt += 1
            if try_cnt < 100:
                ball = sample(predict)
            else:
                ball = sample([1. / len(predict) for __ in predict])
            if ball in balls:
                continue
            balls.append(ball)
            break
    balls = sorted(balls[:settings.FRONT_SIZE]) + sorted(balls[settings.FRONT_SIZE:])
    return balls


def draw_graph(y):
    x = list(range(len(y)))
    parameter = np.polyfit(x, y, 1)
    f = np.poly1d(parameter)
    plt.plot(x, f(x), "r--")
    plt.plot(y)
    plt.show()
