import numpy as np
from scipy.sparse import issparse


class NaiveBayesFromScratch:
    """
    Multinomial Naive Bayes - Fully Vectorized với Matrix Multiplication
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior = None      # shape: (n_classes,)
        self.feature_log_prob = None     # shape: (n_classes, n_features)
        self.classes = None

    def fit(self, X, y):
        """
        Huấn luyện model - Fully vectorized
        
        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        
        # Khởi tạo ma trận
        self.class_log_prior = np.zeros(n_classes)
        self.feature_log_prob = np.zeros((n_classes, n_features))
        
        y_np = np.array(y)
        
        # Tính toán cho tất cả classes cùng lúc
        for idx, c in enumerate(self.classes):
            # Mask cho class c
            mask = (y_np == c)
            X_c = X[mask]
            
            # Prior: log P(y_c)
            n_samples_c = X_c.shape[0]
            self.class_log_prior[idx] = np.log(n_samples_c / n_samples)
            
            # Likelihood: log P(x_i | y_c)
            if issparse(X_c):
                feature_count = np.array(X_c.sum(axis=0)).flatten()
            else:
                feature_count = X_c.sum(axis=0)
            
            # Laplace smoothing
            smoothed_count = feature_count + self.alpha
            total_count = smoothed_count.sum()
            
            self.feature_log_prob[idx, :] = np.log(smoothed_count / total_count)
        
        return self

    def predict_log_proba(self, X):
        """
        Tính log probability - Pure matrix multiplication
        
        log P(y | x) ∝ log P(y) + log P(x | y)
                     = log P(y) + sum_i(x_i * log P(x_i | y))
                     = log P(y) + X @ log P(x_i | y).T
        
        Parameters:
        -----------
        X : shape (n_samples, n_features)
        
        Returns:
        --------
        log_probs : shape (n_samples, n_classes)
        """
        # X @ feature_log_prob.T = (n_samples, n_features) @ (n_features, n_classes)
        #                        = (n_samples, n_classes)
        if issparse(X):
            log_likelihood = X.dot(self.feature_log_prob.T)
        else:
            log_likelihood = X @ self.feature_log_prob.T
        
        # Broadcast cộng log prior: (n_samples, n_classes) + (n_classes,)
        log_probs = log_likelihood + self.class_log_prior
        
        return log_probs

    def predict(self, X):
        """
        Dự đoán nhãn - Vectorized
        
        Returns:
        --------
        predictions : array, shape (n_samples,)
        """
        log_probs = self.predict_log_proba(X)
        # Tìm index của class có log prob cao nhất
        class_indices = np.argmax(log_probs, axis=1)
        return self.classes[class_indices]

    def predict_proba(self, X):
        """
        Tính probability (normalized)
        
        Returns:
        --------
        probs : array, shape (n_samples, n_classes)
        """
        log_probs = self.predict_log_proba(X)
        
        # Numerical stability: log-sum-exp trick
        # P(y|x) = exp(log P(y|x)) / sum_y(exp(log P(y|x)))
        log_probs_shifted = log_probs - np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs_shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        
        return probs

    def score(self, X, y):
        """
        Tính accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)