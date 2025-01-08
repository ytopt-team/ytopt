import numpy as np
import logging
from numpy.typing import NDArray
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from static import CLASSES
from experimental_kernel import ExperimentalKernel
from typing import List

from os import path
from utils import Utils

class Classification:
    def __init__(self, x_train: NDArray[np.int64], y_train: NDArray[np.int64],
                 x_test: NDArray[np.int64], y_test: NDArray[np.int64]) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __score(self, y_test: NDArray[np.int64],
                y_predict: NDArray[np.int64]) -> List[float]:
        predict_count = [0] * len(CLASSES)
        true_count = [0] * len(CLASSES)

        for i, value in enumerate(y_predict):
            true_count[y_test[i]] += 1
            if y_test[i] == value:
                predict_count[y_test[i]] += 1

        score = [predict_count[i] / true_count[i] for i in range(len(CLASSES))]

        return score

    def run(self) -> NDArray[np.float64]:
        result: List[List[float]] = []

        mixing_ratio = 0.5
        sigmoid_ratio = 0.0001
        gaussian_ratio = 1.0
        coef0_param = 0.0
        C_param = 1.0

        logging.info(f"Classification mixing ratio: {mixing_ratio} begins")
        clf = make_pipeline(
            StandardScaler(),
            SVC(C=C_param,kernel=lambda x, y: ExperimentalKernel().mixed_kernel(
              x, y, mixing_ratio,sigmoid_ratio,coef0_param,gaussian_ratio)))

        logging.info("\tTraining...")
        clf.fit(self.x_train, self.y_train)
        logging.info("\tPredicting...")
        y_predict = np.asarray(clf.predict(self.x_test)).astype(np.int64)

        score = self.__score(self.y_test, y_predict)
        logging.info(f"\tScore: {score}")
        score.append(float(clf.score(self.x_test, self.y_test)))
        result.append(score)
        logging.info(
                f"Classification mixing ratio: {mixing_ratio} completes")

        return np.array(result)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()])

data_path = path.join(path.dirname(__file__), "../", "data")

utils = Utils(data_path)
train_data = utils.get_train_data()

result = Classification(train_data["x_train"], train_data["y_train"],
                        train_data["x_test"], train_data["y_test"]).run()

print("Result: ", np.mean(result))
