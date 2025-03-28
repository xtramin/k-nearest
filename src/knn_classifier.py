import numpy as np


class KNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    @staticmethod
    def calculate_distances(self, X_test):
        """
        Calculate pairwise Euclidian distances between X coordinates.
        """
        return np.sqrt(
            np.sum(X_test**2, axis=1)[:, None]
            + np.sum(self.X_train**2, axis=1)
            - 2 * X_test @ self.X_train.T
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

    def _get_neighbors(self, X_test: np.ndarray):
        distances = self.calculate_distances(self, X_test)
        return np.argsort(distances, axis=1)[:, : self.k]

    def predict(self, X_test):
        neighbors = self._get_neighbors(X_test)

        return np.round(np.mean(self.y_train[neighbors], axis=1))
