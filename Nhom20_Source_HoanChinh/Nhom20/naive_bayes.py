"""Gaussian Naive Bayes cho dữ liệu liên tục. Các cột đầu vào phải cùng thứ tự."""
import numpy as np


class GaussianNB:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes, counts = np.unique(y, return_counts=True)
        self.prior = counts / len(y)
        groups = [X[y == label] for label in self.classes]
        self.mean = np.array([group.mean(axis=0) for group in groups])
        # Cộng lượng rất nhỏ để tránh chia cho phương sai bằng 0.
        epsilon = max(1e-9 * X.var(axis=0).max(), np.finfo(float).tiny)
        self.var = np.array([group.var(axis=0) + epsilon for group in groups])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = []
        for prior, mean, var in zip(self.prior, self.mean, self.var):
            # Log P(lớp) + tổng log mật độ Gaussian của từng thuộc tính.
            log_density = -0.5 * (np.log(2 * np.pi * var) + (X - mean) ** 2 / var)
            scores.append(np.log(prior) + log_density.sum(axis=1))
        return self.classes[np.argmax(scores, axis=0)]
