import numpy as np
import cvxpy as cp
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import time
import seaborn as sns


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
    # if problem.status != "optimal":
    #     import pdb; pdb.set_trace()

    return beta.value


random_state = np.random.randint(1000)
print("random_state", random_state)

random_state = 414
samples_sizes = np.linspace(5, 10000, 100)
sup_gaps = []
mean_gaps = []
max_stabs = []
mean_stabs = []

for ns in samples_sizes:

    n_samples, n_features = (int(ns), 100)
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           noise=1, random_state=random_state)

    X /= np.linalg.norm(X, axis=0)
    y = (y - y.mean()) / y.std()
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    X_seen = X[:-1, :]
    y_seen = list(y[:-1])
    X_next = X[-1, :]
    y_next = y[-1]

    lmd = 0.5

    y_min = np.min(y_seen)
    y_max = np.max(y_seen)
    z_min = y_min - 0.25 * (y_max - y_min)
    z_max = y_max + 0.25 * (y_max - y_min)
    zs = np.linspace(z_min, z_max, 100)

    def mu(z):

        yz = y_seen + [z]
        betaz = l1_ridge(X, yz, lmd)

        return X.dot(betaz)

    hatz = 0
    mu_hatz = mu(hatz)
    E_hatz = np.abs(y_seen - mu_hatz[:n_samples - 1])
    norm_X_is = np.linalg.norm(X, axis=1)
    rho = 1
    stab = 2 * rho * norm_X_is / (n_samples * lmd)

    max_stabs += [np.max(stab)]
    mean_stabs += [np.mean(stab)]

    L_hatz = E_hatz - stab[:-1]
    U_hatz = E_hatz + stab[:-1]

    def R_bound(z):

        n_samples = mu_hatz.shape[0]
        S_zhatz = np.abs(z - mu_hatz[-1])
        L_zhatz = S_zhatz - stab[-1]
        U_zhatz = S_zhatz + stab[-1]

        Flo = (1 + np.sum(U_hatz <= L_zhatz)) / n_samples
        Fup = (1 + np.sum(L_hatz <= U_zhatz)) / n_samples

        return Fup, Flo

    def F(z):

        yz = y_seen + [z]
        Ez = np.abs(yz - mu(z))

        return np.sum(Ez <= Ez[-1]) / Ez.shape[0]

    R_zs = np.array([R_bound(z) for z in zs])

    pi_lo = 1 - R_zs[:, 0],
    pi_up = 1 - R_zs[:, 1]
    sup_gaps += [np.max(pi_up - pi_lo)]
    mean_gaps += [np.mean(pi_up - pi_lo)]

# \pi_{\rm{up}}(z, 0) - \pi_{\rm{lo}}(z, 0)

plt.plot(samples_sizes, sup_gaps,
         label=r"$\displaystyle \sup_{z \in Z} \rm{Gap}(z, 0)$")
plt.plot(samples_sizes, mean_gaps,
         label=r"$\displaystyle \frac{1}{|Z|} \sum_{z \in Z} \rm{Gap}(z, 0)$")

plt.plot(samples_sizes, max_stabs, label=r"$\displaystyle \max_{i \in [n+1]} \tau_i$")
plt.plot(samples_sizes, mean_stabs, label=r"$\displaystyle \frac{1}{n+1} \sum_{i \in [n+1]} \tau_i$")
plt.xlabel(r"Sample size $n$")
plt.ylabel("Approximation Gap")
plt.legend(ncol=2)
plt.yscale("log")
plt.grid(None)
strn = str(n_samples)
strp = str(n_features)
# plt.title(r"$(\rm{n + 1, p}) = (%s, %s)$" % (strn, strp))
plt.tight_layout()
plt.savefig("convergence_gap" + ".pdf", format="pdf")
plt.show()
