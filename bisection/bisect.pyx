# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

from libc.math cimport fabs, fmax, log2

cdef float inf = float('inf')

cdef inline double isclose(double a, double b, double rel_tol=1e-09,
                           double abs_tol=1e-12) nogil:

    # see https://docs.python.org/3/library/math.html#math.isclose"
    return fabs(a - b) <= fmax(rel_tol * fmax(fabs(a), fabs(b)), abs_tol)


def root(f, double a, double b, double fa=inf, double fb=inf, int max_iter=0,
         double tol=1e-16):
    '''
    return: (converged, sol, f(sol), a, f(a), b, f(b), number of iterations)
    Note that sol can be equal to a or b.
    '''

    cdef:
        double c = 0.
        double fc = 0.
        int i = 0
        int converged = 0

    if fa == inf:
        fa = f(a)

    if fb == inf:
        fb = f(b)

    if isclose(fa, 0.):
        converged = 1
        return converged, a, fa, a, fa, b, fb, 1

    if isclose(fb, 0.):
        converged = 1
        return converged, b, fb, a, fa, b, fb, 1

    if fa * fb >= 0 or b - a <= 0:
        converged = 0
        print("warning: f(a) and f(b) must have opposite signs.", a, fa, b, fb)
        return converged, inf, inf, a, fa, b, fb, 1

    if max_iter is 0:
        max_iter = int(log2((b - a) / tol)) + 1

    for i in range(max_iter):

        c = 0.5 * (a + b)
        fc = f(c)

        if isclose(fc, 0.) or c - a <= tol:
            converged = 1
            return converged, c, fc, a, fa, b, fb, i

        if fc * fa < 0 and c - a <= tol:
            # When f is discontinuous, there may be no zero in [a, b].
            # In that case, we output a point c such that f takes non negative
            # and positive values in [c +/- tol]
            converged = 1
            return converged, c, fc, a, fa, b, fb, i

        if fc * fa < 0:
            b = c
            fb = fc

        else:
            a = c
            fa = fc

    converged = 0
    print("warning: the algorithm did not converge. Please, increase the\
            number of iterations if necessary")
    print("(Interval length, tol)", c - a, tol)

    return converged, c, fc, a, fa, b, fb, max_iter
