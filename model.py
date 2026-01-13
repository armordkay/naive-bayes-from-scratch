import numpy as np
from scipy.sparse import issparse


class NaiveBayesFromScratch:
  
    def __init__(self, alpha=1.0):
        # Hệ số làm trơn Laplace
        self.alpha = alpha

        # log P(y): xác suất tiên nghiệm của mỗi lớp
        self.class_log_prior = None

        # log P(x_i | y): xác suất có điều kiện của đặc trưng theo từng lớp
        self.feature_log_prob = None

        # Danh sách các lớp (ví dụ: ham, spam)
        self.classes = None

    def fit(self, X, y):
    
        # Lấy các lớp duy nhất
        self.classes = np.unique(y)

        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        # Khởi tạo mảng lưu log prior và log likelihood
        self.class_log_prior = np.zeros(n_classes)
        self.feature_log_prob = np.zeros((n_classes, n_features))

        # Chuyển nhãn sang numpy array để dễ xử lý
        y_np = np.array(y)

        # Tính toán cho từng lớp
        for idx, c in enumerate(self.classes):
            # Lọc các mẫu thuộc lớp c
            mask = (y_np == c)
            X_c = X[mask]

            # Tính log xác suất tiên nghiệm P(y = c)
            n_samples_c = X_c.shape[0]
            self.class_log_prior[idx] = np.log(n_samples_c / n_samples)

            # Tính tổng trọng số của từng đặc trưng trong lớp c
            if issparse(X_c):
                feature_count = np.array(X_c.sum(axis=0)).flatten()
            else:
                feature_count = X_c.sum(axis=0)

            # Áp dụng Laplace smoothing
            smoothed_count = feature_count + self.alpha
            total_count = smoothed_count.sum()

            # Tính log xác suất có điều kiện P(x_i | y = c)
            self.feature_log_prob[idx, :] = np.log(smoothed_count / total_count)

        return self

    def predict_log_proba(self, X):
       
        # Nhân ma trận để tính log likelihood
        if issparse(X):
            log_likelihood = X.dot(self.feature_log_prob.T)
        else:
            log_likelihood = X @ self.feature_log_prob.T

        # Cộng log prior cho mỗi lớp
        log_probs = log_likelihood + self.class_log_prior

        return log_probs

    def predict(self, X):

        log_probs = self.predict_log_proba(X)

        # Chọn lớp có log xác suất lớn nhất
        class_indices = np.argmax(log_probs, axis=1)
        return self.classes[class_indices]

    def predict_proba(self, X):

        log_probs = self.predict_log_proba(X)

        # Chuẩn hóa bằng kỹ thuật log-sum-exp để tránh tràn số
        log_probs_shifted = log_probs - np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs_shifted)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def score(self, X, y):
        
        predictions = self.predict(X)
        return np.mean(predictions == y)
