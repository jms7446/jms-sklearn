import numpy as np

import src.functions as functions


class LogisticRegression(object):

    def __init__(self):
        self.penalty = "l2"
        self.dual = False
        self.tol = 0.0001
        self.C = 1.0
        self.fit_intercept = True
        self.intercept_scaling = 1
        self.class_weight = None
        self.random_state = None
        self.solver = "gd"
        self.max_iter = 100
        self.multi_class = "ovr"
        self.verbose = 0
        self.warm_start = False
        self.n_jobs = 1

        self.w = None
        self.b = None

        self.coef_ = None
        self.intercept_ = None
        self.n_iter = None

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        self : object
            Returns self
        """
        if self.solver == "gd" and sample_weight is None:
            self._fit_gd(X, y)
        return self

    def _fit_gd(self, X, y):
        """y는 0 or 1"""

        learning_rate = 0.01
        C = self.C
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        w = np.zeros(n)
        b = np.zeros(1)

        pre_loss = np.inf
        for i in range(self.max_iter * 1000):
            z = np.dot(X, w) + b
            h = 1.0 / (1.0 + np.exp(-z))
            grad_z = (h - y)
            grad_b = np.sum(grad_z)
            grad_w = np.dot(X.transpose(), grad_z)
            w -= learning_rate * (C * grad_w + w) / m
            b -= learning_rate * grad_b / m
            loss = functions.calc_reg_penalty(w, self.penalty) + C * functions.calc_cross_entropy(y, h)
            if np.abs(pre_loss - loss) <= self.tol * 0.00001:
                if self.verbose:
                    print("loss diff : {}, break ({}th iteration".format(pre_loss - loss, i))
                break
            pre_loss = loss
            # if i % 100 == 0:
            #     print("{:5d} loss : {}".format(i, loss))

        self.w = w
        self.b = b
        self.coef_ = w.reshape(1, n)
        self.intercept_ = b

    def predict(self, X):
        """Predict class labels for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.
        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_log_proba(self, X):
        """Log of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        """Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        X = np.array(X)
        z = np.dot(X, self.w) + self.b
        h = functions.sigmoid(z)
        return np.array([1 - h, h]).transpose()

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        if sample_weight is None:
            return np.mean((self.predict(X) == y).astype(float))

    @staticmethod
    def calc_cross_entropy_with_weight(X, y, w, b):
        z = np.dot(X, w) + b
        h = functions.sigmoid(z)
        loss = functions.calc_cross_entropy(y, h)
        return loss
