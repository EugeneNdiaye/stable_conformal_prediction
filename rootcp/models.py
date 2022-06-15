import numpy as np
import cvxpy as cp


class ridge:
    """ Ridge estimator.
    """

    def __init__(self, lmd=0.1):

        self.lmd = lmd
        self.hat = None
        self.hatn = None

    def fit(self, X, y):

        if self.hat is None:
            G = X.T.dot(X) + self.lmd * np.eye(X.shape[1])
            self.hat = np.linalg.solve(G, X.T)

        if self.hatn is None:
            y0 = np.array(list(y[:-1]) + [0])
            self.hatn = self.hat.dot(y0)

        self.beta = self.hatn + y[-1] * self.hat[:, -1]

    def predict(self, X):

        return X.dot(self.beta)

    def conformity(self, y, y_pred):

        return 0.5 * np.square(y - y_pred)


class lad:
    """ Ridge estimator.
    """

    def __init__(self, lmd=0.5):

        self.lmd = lmd
        self.hat = None
        self.hatn = None

    def fit(self, X, y):

        beta = cp.Variable(X.shape[1])
        lmd = cp.Parameter(nonneg=True)
        lmd.value = self.lmd

        def objective(X, y, beta, lmd):

            loss = cp.norm1(X @ beta - y) / X.shape[0]
            reg = lmd * cp.norm2(beta) ** 2

            return loss + reg

        problem = cp.Problem(cp.Minimize(objective(X, y, beta, lmd)))
        problem.solve(solver='ECOS')

        self.beta = beta.value

    def predict(self, X):

        return X.dot(self.beta)

    def conformity(self, y, y_pred):

        return np.abs(y - y_pred)


class regressor:

    def __init__(self, model=None, s_eps=0., conform=None):

        self.model = model
        self.coefs = []
        self.s_eps = s_eps
        self.conform = conform

    def fit(self, X, y):

        refit = True

        for t in range(len(self.coefs)):

            if self.s_eps == 0:
                break

            if abs(self.coefs[t][0] - y[-1]) <= self.s_eps:
                self.beta = self.coefs[t][1].copy()
                refit = False
                break

        if refit:
            self.beta = self.model.fit(X, y)
            if self.s_eps != 0:
                self.coefs += [[y[-1], self.beta.copy()]]

    def predict(self, X):

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        return self.model.predict(X)

    def conformity(self, y, y_pred):

        if self.conform is None:
            return np.abs(y - y_pred)

        else:
            return self.conform(y, y_pred)
