import numpy as np
import cvxpy as cp
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn import datasets
from sklearn.neural_network import MLPRegressor


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

random_state = np.random.randint(1000)
print("random_state", random_state)

random_state = 414

# datas = ["synthetic", "diabetes"]
dataset = "synthetic"
# dataset = 'diabetes'
# dataset = "housingcalifornia"

if dataset == "synthetic":
    n_samples, n_features = (300, 100)
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           noise=1, random_state=random_state)


if dataset == "diabetes":
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target


if dataset == "housingcalifornia":
    housing = datasets.fetch_california_housing()
    X, y = housing.data, housing.target


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
# max_iter = 100

scales = np.array([0.1, 0.5, 0.9, 2])
max_iters = scales * X.shape[0]
linestyles = ["-.", ":"]
labels = [r"$\pi_{\rm{up}}(z, 0)$, $\tau_i = \frac{T\|x_i\|}{n}$",
          r"$\pi_{\rm{up}}(z, 0)$, $\tau_i = \frac{\|x_i\|}{n}$"]

plt.figure(figsize=(8, 6))

for t, max_iter in enumerate(max_iters):

    max_iter = int(max_iter)

    def mu(z):

        yz = y_seen + [z]
        reg = MLPRegressor(random_state=404, max_iter=max_iter, alpha=lmd,
                           solver="sgd")
        reg.fit(X, yz)

        return reg.predict(X)

    def pi(z):

        yz = y_seen + [z]
        Ez = np.abs(yz - mu(z))

        return 1 - np.sum(Ez <= Ez[-1]) / Ez.shape[0]

    y_min = np.min(y_seen)
    y_max = np.max(y_seen)
    z_min = y_min - 0.25 * (y_max - y_min)
    z_max = y_max + 0.25 * (y_max - y_min)
    zs = np.linspace(z_min, z_max, 100)

    pi_zs = np.array([pi(z) for z in zs])
    plt.plot(zs, pi_zs, label=r"$\pi(z)$")

    mu_hatz = mu(0)
    E_hatz = np.abs(y_seen - mu_hatz[:n_samples - 1])
    norm_X_is = np.linalg.norm(X, axis=1)

    stabs = [max_iter * norm_X_is / n_samples, norm_X_is / n_samples]
    # stab = max_iter * norm_X_is / n_samples
    # stab = norm_X_is / n_samples

    for tt, stab in enumerate(stabs):

        L_hatz = E_hatz - stab[:-1]
        U_hatz = E_hatz + stab[:-1]

        def pi_up(z):

            n_samples = mu_hatz.shape[0]
            S_zhatz = np.abs(z - mu_hatz[-1])
            L_zhatz = S_zhatz - stab[-1]

            Flo = (1 + np.sum(U_hatz <= L_zhatz)) / n_samples

            return 1 - Flo

        Fup_zs = np.array([pi_up(z) for z in zs])
        plt.plot(zs, Fup_zs, label=labels[tt], ls=linestyles[tt])

    plt.xlabel("Candidate z")
    plt.ylabel("conformity of z")
    plt.grid(None)
    plt.title(r"\texttt{n_iter} = $%s \times$ \texttt{n_samples}" % scales[t])
    if t == 0:
        plt.legend()
    plt.tight_layout()
    img_name = str(scales[t]).replace(".", "_")
    plt.savefig(img_name + dataset + ".pdf", format="pdf")
    plt.show()
