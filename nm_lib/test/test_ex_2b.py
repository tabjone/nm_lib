import numpy as np
from nm_lib import nm_lib as nm
db = np.float64

"""
TO:DO: Test for different derivation schemes
"""

def u(x):
    """
    u(x) function from eq. (2)"""
    return np.cos(db(6)*np.pi*x/db(5))**2/(np.cosh(db(5)*x**2))

def shift_xx(xx, a, t, xf, x0):
    #let x-grid move with velocity a to the side and wrap around to other side
    #takes in the velocity a and the current time t. And end of grid xf and start x0
    xx_new = ((xx - a * t) - x0) % (xf - x0) + x0
    return xx_new

def test_ex_2b():
    x0 = db(-1.6)
    xf = db(1.6)

    nump = 64
    nt = 50

    xx = np.arange(nump, dtype=db)/(nump-db(1.0)) * (xf-x0) + x0
    hh = u(xx)

    a = db(-1)

    #calculating numerical solution
    tt, ut = nm.evolv_adv_burgers(xx, hh, nt, a, ddx=nm.deriv_finite)
    #calulating numerical solution
    X = np.zeros((len(tt), len(xx)))
    uu_analytic = np.zeros((len(tt),len(xx)))

    for i in range(0, len(tt)):
        X[i, :] = shift_xx(xx, a, tt[i], xf, x0)
        uu_analytic[i,:] = u(X[i,:])

    diff = np.abs(uu_analytic[-1,:] - ut[-1,:])
    assert np.all(diff < 1e-1)
