from unittest import TestCase, main
import numpy as np

from sklearn.linear_model import LogisticRegression as CmpLogisticRegression

from linear_model.logistic import LogisticRegression
import functions


# X1 = [[-1, 0], [0, 1], [1, 1], [2, 1.5]]
# X_sp = sp.csr_matrix(X1)
# Y1 = [0, 1, 1, 1]
# Y2 = [2, 1, 0, 2]


class SampleTest(TestCase):

    def setUp(self):
        np.random.seed(4)
        m = 20
        n = 3
        num_class = 2
        self.X = np.random.random((m, n))
        self.y = np.random.randint(0, num_class, m)

    def assert_equal_clf(self, clf1, clf2, X, y):
        np.testing.assert_almost_equal(clf1.coef_, clf2.coef_, decimal=3)
        np.testing.assert_almost_equal(clf1.intercept_, clf2.intercept_, decimal=3)

        np.testing.assert_equal(clf1.predict(X), clf2.predict(X))
        np.testing.assert_almost_equal(clf1.predict_proba(X), clf2.predict_proba(X), decimal=3)
        np.testing.assert_almost_equal(clf1.predict_log_proba(X), clf2.predict_log_proba(X), decimal=3)
        np.testing.assert_almost_equal(clf1.score(X, y), clf2.score(X, y), decimal=3)

    def test_gd(self):
        X, y = self.X, self.y
        clf = LogisticRegression()
        clf.fit(X, y)

        clf2 = CmpLogisticRegression(solver="saga")
        clf2.fit(X, y)

        self.assert_equal_clf(clf, clf2, X, y)

    def test_fit_of_sklearn(self):
        X, y = self.X, self.y
        cmp_clf = CmpLogisticRegression(fit_intercept=True, solver="saga", max_iter=100000, tol=0.000001)
        cmp_clf.fit(X, y)
        print("saga   : ", cmp_clf.classes_, cmp_clf.coef_, cmp_clf.intercept_, cmp_clf.score(X, y))

        cmp_clf = CmpLogisticRegression(fit_intercept=True, solver="liblinear", max_iter=100000, tol=0.000001)
        cmp_clf.fit(X, y)
        print("default: ", cmp_clf.classes_, cmp_clf.coef_, cmp_clf.intercept_, cmp_clf.score(X, y))

    def test_coordinate_check(self):
        X, y = self.X, self.y
        m, n = X.shape

        # cmp_clf = CmpLogisticRegression(solver="saga")
        clf = CmpLogisticRegression(solver="liblinear")
        clf.fit(X, y)
        # print("default: ", clf.classes_, clf.coef_, clf.intercept_, clf.score(X, y))

        pre = 0.0
        interval = 0.0001
        r = 5
        for xi in range(3):
            print("{} direction".format(xi))
            for i in range(-r, r, 1):
                w = clf.coef_[0]
                b = clf.intercept_[0]
                gap = interval * i
                if xi < n:
                    w += functions.get_one_hot(xi, n) * gap
                else:
                    b += gap
                loss = LogisticRegression.calc_cross_entropy_with_weight(X, y, w, b)
                print("{:.8f}, {:10.8e}, {:4d}".format(loss, loss - pre, i))
                pre = loss


if __name__ == "__main__":
    main()
