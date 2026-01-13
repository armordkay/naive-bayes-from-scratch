import numpy as np


class NaiveBayesFromScratch:
    """
    Multinomial Naive Bayes tự cài đặt
    Hoạt động với dữ liệu đã vector hóa (TF-IDF hoặc Bag of Words)
    """

    def __init__(self):
        self.class_log_prior = {}     # log P(y)
        self.feature_log_prob = {}    # log P(x_i | y)
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        y_np = np.array(y)

        for c in self.classes:
            # Lấy các mẫu thuộc lớp c
            X_c = X[y_np == c]

            # Prior
            self.class_log_prior[c] = np.log(X_c.shape[0] / X.shape[0])

            # Tổng TF-IDF mỗi feature
            feature_sum = np.array(X_c.sum(axis=0)).flatten()

            # Laplace smoothing
            feature_sum += 1
            total_sum = feature_sum.sum()

            self.feature_log_prob[c] = np.log(feature_sum / total_sum)


    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới
        X: sparse matrix
        """

        predictions = []

        for i in range(X.shape[0]):
            sample = X[i]

            class_scores = {}

            for c in self.classes:
                # log P(y)
                score = self.class_log_prior[c]

                # log P(x | y) = sum(x_i * log P(x_i | y))
                score += sample.dot(self.feature_log_prob[c])

                class_scores[c] = score

            # Chọn lớp có score cao nhất
            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)
