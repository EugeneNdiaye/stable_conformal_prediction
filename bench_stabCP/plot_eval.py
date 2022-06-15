import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


random_state = np.random.randint(1000)

print("random_state", random_state)
alpha = 0.1

n_samples, n_features = (1000, 500)
X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       random_state=random_state)

X /= np.linalg.norm(X, axis=0)
y = (y - y.mean()) / y.std()

X = np.asfortranarray(X)
y = np.asfortranarray(y)

X_seen = X[:-1, :]
y_seen = list(y[:-1])
X_next = X[-1, :]
y_next = y[-1]

lmd_max = np.log(n_features)
lmd = lmd_max / 50.

hat = np.linalg.solve(X.T.dot(X) + lmd * np.eye(n_features), X.T)


def mu(z):

    return X.dot(hat.dot(y_seen + [z]))


def F(z):

    yz = y_seen + [z]
    scores = np.abs(yz - mu(z))

    return np.sum(scores <= scores[-1]) / scores.shape[0]


hatz = mu(0)[-1]
# hatz = 0
mu_hatz = mu(hatz)


def Fup(z, mu_hatz):

    yz = y_seen + [z]
    scores = np.abs(yz - mu_hatz)
    loose = 2 / (scores.shape[0] * lmd)

    return np.sum(scores <= scores[-1] - loose) / scores.shape[0]


zs = np.linspace(np.min(y_seen), np.max(y_seen), 100)
F_zs = np.array([F(z) for z in zs])
Fup_zs = np.array([Fup(z, mu_hatz) for z in zs])

plt.figure()
plt.plot(zs, 1 - F_zs, label="exact")
plt.plot(zs, 1 - Fup_zs, label="approx")
plt.legend()
plt.tight_layout()
plt.show()
