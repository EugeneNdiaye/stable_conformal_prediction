import numpy as np
from scipy.optimize import root_scalar
from scipy.special import expit


def rank(u, gamma=None):

    if gamma is None:
        return np.sum(u - u[-1] <= 0)

    sigmoid = 1 - expit(gamma * (u - u[-1]))
    return np.sum(sigmoid)


def conformalset(X, y, model, stab=0, alpha=0.1, gamma=None, tol=1e-3, nqp=10,
                 algo=None, stat=False, hatz=0):
    """ Compute full conformal prediction set with root-finding solver.

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape = (n_samples - 1,)
        Target values.
    model : class that represents a regressor with a model.fit,
            model.predict and model.conformity.
    alpha : float in (0, 1)
            Coverage level.
    gamma : float
            Smoothing parameter.
    tol : float
          Tolerance error for the root-finding solver.
    Returns
    -------
    list : list of size 2
           conformal set [l(alpha), u(alpha)].
    """

    y_min = np.min(y)
    y_max = np.max(y)
    z_min = y_min - 0.5 * (y_max - y_min)
    z_max = y_max + 0.5 * (y_max - y_min)
    yz = np.array(list(y) + [0])
    model.fit(X, yz)
    z_0 = model.predict(X[-1, :])
    min_b = min(abs(z_0 - y_min), abs(y_max - z_0))
    n_samples = X.shape[0]

    yz[-1] = hatz
    model.fit(X, yz)

    # get as input, along with lmd
    # lmd = 0.1
    norm_X_is = np.linalg.norm(X, axis=1)
    # for LAD
    # rho = 1
    # stab = 2 * rho * norm_X_is / (n_samples * lmd)
    # for MLP
    n_iter = int(0.1 * y.shape[0])
    stab = n_iter * norm_X_is / n_samples

    E_hatz = model.conformity(y, model.predict(X[:n_samples - 1]))
    U_hatz = E_hatz + stab[:-1]

    def pvalue(z):

        S_zhatz = model.conformity(z, model.predict(X[-1]))
        L_zhatz = S_zhatz - stab[-1]

        return 1 - (1 + np.sum(U_hatz <= L_zhatz)) / n_samples

    def objective(z):
        # we use a bit more conservative alpha to include all roots when
        # the p_value function is piecewise constant.
        return pvalue(z) - (alpha - 1e-11)

    if objective(z_0) < 0:

        init_found = False

        # TODO: avoid multiple and useless rerun by storing the fitted model
        # residual = model.conformity(yz, model.predict(X))
        # q_alpha = np.quantile(residual, 1 - alpha)
        # scp = z_0 - q_alpha, z_0 + q_alpha
        # scp = z_0 - 2 * q_alpha, z_0 + 2 * q_alpha
        # i_lb = scp[0] - 0.5 * abs(scp[1] - scp[0])
        # i_ub = scp[1] + 0.5 * abs(scp[1] - scp[0])
        i_lb, i_ub = z_0 - min_b, z_0 + min_b
        xs = np.linspace(i_lb, i_ub, nqp)
        np.random.shuffle(xs)

        count = 0
        # objs = []
        for z in xs:

            count += 1
            obj = objective(z)
            # objs += [obj]

            if obj > 0:
                z_0 = z
                print("stab: fail and found", z, count)
                init_found = True
                break

        if init_found is False:
            print("stab: initialization z_0 failed")
            # TODO: Other option is to jump on interpolation
            return [i_lb, i_ub]

    # root-finding
    # algo = "bisect" if gamma is None else "brenth"
    if algo is None:
        algo = "brenth"

    if objective(z_min) < 0:
        left = root_scalar(objective, bracket=[z_min, z_0], method=algo,
                           xtol=tol)
        lb = left.root
        nf_left = left.function_calls
    else:
        # lb = y_min
        lb = z_0 - min_b
        nf_left = 0
        #  use lower bound of scp
        print("stab: lb failed")

    if objective(z_max) < 0:
        right = root_scalar(objective, bracket=[z_0, z_max], method=algo,
                            xtol=tol)
        ub = right.root
        nf_right = right.function_calls
    else:
        # ub = y_max
        ub = z_0 + min_b
        nf_right = 0
        #  TODO: use upper bound of scp
        print("stab: ub failed")

    if stat:
        return [lb, ub], nf_right + nf_left

    return [lb, ub]
