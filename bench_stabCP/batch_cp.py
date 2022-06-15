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

    return beta.value


random_state = np.random.randint(1000)
print("random_state", random_state)

random_state = 414
n_samples, n_features = (30, 100)

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

# lmd_max = np.log(n_features)
# lmd = lmd_max / 50.
lmd = 0.5


def mu(z):

    yz = y_seen + [z]
    betaz = l1_ridge(X, yz, lmd)

    return X.dot(betaz)


y_min = np.min(y_seen)
y_max = np.max(y_seen)
z_min = y_min - 0.25 * (y_max - y_min)
z_max = y_max + 0.25 * (y_max - y_min)
zs = np.linspace(z_min, z_max, 100)

hatzs = zs[[25, 50, 75]]
L_hatzs, U_hatzs, mu_hatzs = [], [], []

norm_X_is = np.linalg.norm(X, axis=1)
rho = 1
stab = 2 * rho * norm_X_is / (n_samples * lmd)

for hatz in hatzs:

    mu_hatz = mu(hatz)
    E_hatz = np.abs(y_seen - mu_hatz[:n_samples - 1])
    L_hatzs += [E_hatz - stab[:-1]]
    U_hatzs += [E_hatz + stab[:-1]]
    mu_hatzs += [mu_hatz]


mu_hatz0 = mu(0)
E_hatz0 = np.abs(y_seen - mu_hatz0[:n_samples - 1])
L_hatz0 = E_hatz0 - stab[:-1]
U_hatz0 = E_hatz0 + stab[:-1]


def R_bound0(z):

    S_zhatz = np.abs(z - mu_hatz0[-1])
    L_zhatz = S_zhatz - stab[-1]
    U_zhatz = S_zhatz + stab[-1]

    Flo = (1 + np.sum(U_hatz0 <= L_zhatz)) / n_samples
    Fup = (1 + np.sum(L_hatz0 <= U_zhatz)) / n_samples

    return Fup, Flo


def R_bound(z):

    i_z = np.argsort(abs(z - hatzs))[0]
    mu_hatz = mu_hatzs[i_z]
    S_zhatz = np.abs(z - mu_hatz[-1])
    L_zhatz = S_zhatz - stab[-1]
    U_zhatz = S_zhatz + stab[-1]

    Flo = (1 + np.sum(U_hatzs[i_z] <= L_zhatz)) / n_samples
    Fup = (1 + np.sum(L_hatzs[i_z] <= U_zhatz)) / n_samples

    return Fup, Flo


def R_bound2(z):

    Flo = -np.inf
    Fup = +np.inf

    for i_z in range(len(hatzs)):

        mu_hatz = mu_hatzs[i_z]
        S_zhatz = np.abs(z - mu_hatz[-1])
        L_zhatz = S_zhatz - stab[-1]
        U_zhatz = S_zhatz + stab[-1]

        Flo = max(Flo, (1 + np.sum(U_hatzs[i_z] <= L_zhatz)) / n_samples)
        Fup = min(Fup, (1 + np.sum(L_hatzs[i_z] <= U_zhatz)) / n_samples)

    return Fup, Flo


def F(z):

    yz = y_seen + [z]
    Ez = np.abs(yz - mu(z))

    return np.sum(Ez <= Ez[-1]) / Ez.shape[0]


F_zs = np.array([F(z) for z in zs])
R0_zs = np.array([R_bound0(z) for z in zs])
R_zs = np.array([R_bound(z) for z in zs])
R2_zs = np.array([R_bound2(z) for z in zs])

labello = r"$\pi_{\rm{lo}}(z, \hat z)$"
labelup = r"$\pi_{\rm{up}}(z, \hat z)$"

# TODO: proper naming

plt.plot(zs, 1 - R_zs[:, 0], label=labello)
plt.plot(zs, 1 - R2_zs[:, 0], linestyle="-.", label=labello + " batch")
plt.plot(zs, 1 - R0_zs[:, 0], linestyle="--", label=labello + " single")

plt.plot(zs, 1 - F_zs, marker="*", label=r"$\pi(z)$")

plt.plot(zs, 1 - R_zs[:, 1], label=r"$\pi_{\rm{up}}(z, \hat z)$")
plt.plot(zs, 1 - R2_zs[:, 1], linestyle="-.", label=labelup + " batch")
plt.plot(zs, 1 - R0_zs[:, 1], linestyle="-.", label=labelup + " single")

plt.xlabel("Candidate z")
plt.ylabel("conformity of z")
plt.legend()
plt.grid(None)
strn = str(n_samples)
strp = str(n_features)
# plt.title(r"$(\rm{n + 1, p}) = (%s, %s)$" % (strn, strp))
plt.tight_layout()
plt.show()
