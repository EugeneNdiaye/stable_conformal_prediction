import numpy as np
import cvxpy as cp
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn import datasets


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=1.5)
    sns.set_palette('muted')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


plt.rcParams["text.usetex"] = True
set_style()


def l1_ridge(X, y, lamda):

    beta = cp.Variable(X.shape[1])
    lmd = cp.Parameter(nonneg=True)
    lmd.value = lamda

    def objective(X, y, beta, lmd):

        return cp.norm1(X @ beta - y) / X.shape[0] + lmd * cp.norm2(beta) ** 2

    problem = cp.Problem(cp.Minimize(objective(X, y, beta, lmd)))
    problem.solve(solver='ECOS')

    return beta.value


random_state = np.random.randint(1000)
print("random_state", random_state)

random_state = 414

datas = ["synthetic", "diabetes"]
dataset = "synthetic"
dataset = 'diabetes'

if dataset == "synthetic":
    n_samples, n_features = (30, 100)
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           noise=1, random_state=random_state)


if dataset == "diabetes":
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target


X /= np.linalg.norm(X, axis=0)
mask = np.sum(np.isnan(X), axis=0) == 0
if np.any(mask):
    X = X[:, mask]

y = (y - y.mean()) / y.std()
X = np.asfortranarray(X)
y = np.asfortranarray(y)

n_samples, n_features = X.shape

X_seen = X[:-1, :]
y_seen = list(y[:-1])
X_next = X[-1, :]
y_next = y[-1]

lmd = 0.5


def mu(z):

    yz = y_seen + [z]
    betaz = l1_ridge(X, yz, lmd)

    return X.dot(betaz)


def pi(z):

    yz = y_seen + [z]
    Ez = np.abs(yz - mu(z))

    return 1 - np.sum(Ez <= Ez[-1]) / Ez.shape[0]


y_min = np.min(y_seen)
y_max = np.max(y_seen)
z_min = y_min - 0.25 * (y_max - y_min)
z_max = y_max + 0.25 * (y_max - y_min)
zs = np.linspace(z_min, z_max, 100)

plt.figure(figsize=(8, 6))

pi_zs = np.array([pi(z) for z in zs])
plt.plot(zs, pi_zs, label=r"$\pi(z)$")

hatzs = [y_min, 0, 0.5 * (y_min + y_max), mu(0)[-1], y_next, y_max]
labels = [r"$\pi_{\rm{up}}(z, y_{\min})$",
          r"$\pi_{\rm{up}}(z, 0)$",
          r"$\pi_{\rm{up}}(z, (y_{\min} + y_{\max}) / 2)$",
          r"$\pi_{\rm{up}}(z, \mu_0(x_{n+1}))$",
          r"$\pi_{\rm{up}}(z, y_{n+1}$)",
          r"$\pi_{\rm{up}}(z, y_{\max})$"]


for t, hatz in enumerate(hatzs):

    print("t is", t)

    mu_hatz = mu(hatz)
    E_hatz = np.abs(y_seen - mu_hatz[:n_samples - 1])
    norm_X_is = np.linalg.norm(X, axis=1)
    rho = 1
    stab = 2 * rho * norm_X_is / (n_samples * lmd)

    L_hatz = E_hatz - stab[:-1]
    U_hatz = E_hatz + stab[:-1]

    def pi_up(z):

        n_samples = mu_hatz.shape[0]
        S_zhatz = np.abs(z - mu_hatz[-1])
        L_zhatz = S_zhatz - stab[-1]

        Flo = (1 + np.sum(U_hatz <= L_zhatz)) / n_samples

        return 1 - Flo

    Fup_zs = np.array([pi_up(z) for z in zs])
    plt.plot(zs, Fup_zs, label=labels[t])

plt.xlabel("Candidate z")
plt.ylabel("conformity of z")
# plt.legend(ncol=2)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(None)
strn = str(n_samples)
strp = str(n_features)
plt.tight_layout()
plt.savefig("hatz_" + dataset + ".pdf", format="pdf")
plt.show()
