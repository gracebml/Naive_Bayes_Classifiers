import numpy as np
from scipy.sparse import issparse

class NaiveBayes:
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._class_count = np.zeros(n_classes, dtype=np.float64)
        self._feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        self._class_log_prior = np.zeros(n_classes, dtype=np.float64)
        self._feature_log_prob = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self._classes):
            mask = (y == c)
            X_c = X[mask]
            
            self._class_count[idx] = X_c.shape[0]
            
            if issparse(X_c):
                self._feature_count[idx, :] = np.array(X_c.sum(axis=0)).flatten()
            else:
                self._feature_count[idx, :] = X_c.sum(axis=0)
        
        self._class_log_prior = np.log(self._class_count) - np.log(n_samples)
        
        # Calculate feature log probabilities with Laplace smoothing
        for idx in range(n_classes):
            total_count = self._feature_count[idx].sum()
            
            smoothed_count = self._feature_count[idx] + self.alpha
            smoothed_total = total_count + self.alpha * n_features
            
            self._feature_log_prob[idx, :] = np.log(smoothed_count) - np.log(smoothed_total)

    def predict(self, X):
        """Predict class for samples in X"""
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=self._classes.dtype)
        
        for i in range(n_samples):
            if issparse(X):
                x = X[i].toarray().flatten()
            else:
                x = X[i]
            y_pred[i] = self._predict_single(x)
            
        return y_pred
    
    def _predict_single(self, x):
        log_posteriors = []
        
        for idx in range(len(self._classes)):
            log_posterior = self._class_log_prior[idx]
            
            # Add log likelihood: sum(x_i * log(theta_i))
            # For multinomial NB, this is the sum of feature_value * log_prob
            log_posterior += np.sum(x * self._feature_log_prob[idx, :])
            
            log_posteriors.append(log_posterior)
        
        # Return class with highest log posterior
        return self._classes[np.argmax(log_posteriors)]
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self._classes)
        probas = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            if issparse(X):
                x = X[i].toarray().flatten()
            else:
                x = X[i]
            
            log_posteriors = []
            for idx in range(n_classes):
                log_posterior = self._class_log_prior[idx]
                log_posterior += np.sum(x * self._feature_log_prob[idx, :])
                log_posteriors.append(log_posterior)
            
            # Convert log probabilities to probabilities using log-sum-exp 
            log_posteriors = np.array(log_posteriors)
            max_log_posterior = np.max(log_posteriors)
            exp_posteriors = np.exp(log_posteriors - max_log_posterior)
            probas[i, :] = exp_posteriors / np.sum(exp_posteriors)
            
        return probas
    
    def accuracy(self, y_true, y_pred):
        correct = np.sum(y_pred == y_true)
        return float(correct) / len(y_true)
    
    def precision(self, y_true, y_pred):
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        return float(TP / (TP + FP))
    
    def recall(self, y_true, y_pred):
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        return float(TP / (TP + FN))
    
    def f1_score(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2* (precision * recall) / (precision + recall)
    
    def evaluate(self, y_true, y_pred):
        accuracy = self.accuracy(y_true, y_pred)
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        f1_score = self.f1_score(y_true, y_pred)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}
