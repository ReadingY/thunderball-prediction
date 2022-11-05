import settings
import utils
from model.models import model
from data.dataset import LottoDataSet


def predict():
    model.load_weights(settings.PREDICTION_MODEL_PATH)
    dataset = LottoDataSet()
    x = dataset.predict_data
    predictions = model.predict(x, batch_size=1)

    results = []
    probabilities = [prediction[0] for prediction in predictions]
    for i in range(settings.PREDICTION_NUM):
        balls = utils.select_seqs(probabilities)
        results.append([ball + 1 for ball in balls])

    print('The {} group prediction numbers:'.format(settings.PREDICTION_NUM))
    for balls in results:
        print(' '.join(map(str, balls)))
