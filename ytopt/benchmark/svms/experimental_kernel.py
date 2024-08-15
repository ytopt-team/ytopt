import numpy as np
from numpy.typing import NDArray


class ExperimentalKernel:
    def sigmoid_kernel(self, x: NDArray[np.int64], y: NDArray[np.int64],
                       ratio: float) -> NDArray[np.float64]:
        C = 0
        output = np.asarray(np.tanh(ratio * np.dot(x, y.T) + C))
        return output

    def euclidean_dist_matrix(
            self, data_1: NDArray[np.int64],
            data_2: NDArray[np.int64]) -> NDArray[np.float64]:
        norms_1 = (data_1**2).sum(axis=1)
        norms_2 = (data_2**2).sum(axis=1)
        return np.asarray(
            np.abs(
                norms_1.reshape(-1, 1) + norms_2 -
                2 * np.dot(data_1, data_2.T)))

    def gaussian_kernel(self, x: NDArray[np.int64], y: NDArray[np.int64],
                        ratio: float) -> NDArray[np.float64]:
        dists_sq = self.euclidean_dist_matrix(x, y)
        output = np.asarray(np.exp(-ratio * dists_sq))
        return output

    def mixed_kernel(self,
                     x: NDArray[np.int64],
                     y: NDArray[np.int64],
                     mixing_ratio: float,
                     sigmoid_ratio: float = 0.0001,
                     gaussian_ratio: float = 1) -> NDArray[np.float64]:
        return np.asarray(
            (1 - mixing_ratio) * self.sigmoid_kernel(x, y, sigmoid_ratio) +
            mixing_ratio * self.gaussian_kernel(x, y, gaussian_ratio))
