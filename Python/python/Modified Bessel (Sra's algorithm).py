
# coding: utf-8

# Here is a partial implementation of the paper: <br />
# "*A short note on parameter approximation for von Mises-Fisher distributions
# And a fast implementation of Is(x)*"<br /> by Suvrit Sra

# In[1]:

import numpy as np
from joblib import Parallel, delayed
import itertools as it
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from scipy.special import iv as iv_scipy
from numpy import pi, power, exp, sqrt
from gmpy2 import mpfr, gamma, factorial, const_pi
from gmpy2 import exp as gmpexp
from gmpy2 import sqrt as gmpsqrt
from gmpy2 import get_context

get_context().precision = 300


# In[2]:

def iv_s_paper(s, x, tau):
    """
    Computing Is(x) via truncated power-series (Paper version) 
    Beware : algorithmic issue !!
    """
    R, t1, t2 = 1., pow(x * exp(1.) / (2 * s), s),             1.             + 1. / (12 * s)             + 1. / (288 * s * s)             - 139. / (51840 * s * s * s)
    t1 = t1 * sqrt(s / (2 * pi)) / t2
    M, k, const_rat = 1. / s, 1, 0.25 * x * x
    convergence = False
    while not convergence:
        R = R * const_rat / (k * (s + k))
        M += R
        if R / M < tau:
            convergence = True
        k += 1
    return t1 * M


# In[3]:

def iv_s(s, x, tau):
    """
    Computing Is(x) via truncated power-series (corrected version) 
    """
    t1, t2 = pow(x * exp(1.) / (2 * s), s),             1.             + 1. / (12 * s)             + 1. / (288 * s * s)             - 139. / (51840 * s * s * s)
    t1 *= sqrt(s / (2 * pi)) / t2
    k, tk, const_rat = 0, 1. / s, x * x / 4.    
    M = tk
    convergence = False
    while not convergence:
        R = const_rat / ((k + 1) * (s + k + 1))
        tk *= R
        k += 1
        M += tk
        if tk / M < tau:
            convergence = True
    return t1 * M


# In[4]:

def iv_s(s, x, tau):
    """
    Computing Is(x) via truncated power-series (corrected version) 
    """
    t1, t2 = power(x * exp(1.) / (2 * s), s),             1.             + 1. / (12 * s)             + 1. / (288 * s * s)             - 139. / (51840 * s * s * s)
    t1 *= sqrt(s / (2 * pi)) / t2
    k, tk, const_rat = 0, 1. / s, x * x / 4.
    M = tk
    convergence = False
    while not convergence:
        ratio = const_rat / ((k + 1) * (s + k + 1))
        tk *= ratio
        M += tk
        k += 1
        if tk / M < tau:
            convergence = True
    return t1 * M


# In[5]:

def iv_s_mpfr(s, x, tau):
    s, x = mpfr(s), mpfr(x)
    t1 = pow(x * gmpexp(1.) / (2 * s), s)
    t2 = 1.+ 1. / (12 * s) + 1. / (288 * s * s) - 139. / (51840 * s * s * s)
    t1 = t1 * gmpsqrt(s / (2 * const_pi())) / t2
    k, tk, const_rat = 0, 1. / s, x * x / 4
    M = tk
    convergence = False
    while not convergence:
        ratio = const_rat / ((k + 1) * (s + k + 1))
        tk *= ratio
        M += tk
        if tk / M < tau:
            convergence = True
        k += 1
    return t1 * M


# In[6]:

def iv_s_eq7 (s, x, order=100):
    s, x = mpfr(s), mpfr(x)
    t1 = pow(x / 2, s)
    t2 = mpfr(0)
    k = 0
    t2 = sum(pow(x * x / 4, k) / (gamma(k + s + 1) * factorial(k)) for k in range(order))
    return t1 * t2

def iv_s_eq8 (s, x, order=100):
    s, x = mpfr(s), mpfr(x)
    t1 = pow(x / 2, s) / gamma(s)
    t2 = mpfr(0)
    k = 0
    s_prod = s
    while k < order:
        t2 += pow(x * x / 4, k) / (s_prod * factorial(k))   
        k += 1
        s_prod *= s + k
    return t1 * t2


# ### Verify correctness

# In[7]:

tau = 1e-16
order = 1000
s = 200
v = 200
print ('--', s , '--', v, '--')
print ('Scipy : ', iv_scipy(s, v))
print ('paper : ', iv_s_paper(s, v, tau), '(incorrect)')
print ('corrected : ',iv_s(s, v, tau))
print ('mpfr : ',iv_s_mpfr(s, v, tau))
print ('eq.7 : ',iv_s_eq7(s, v, order))
print ('eq.8 : ',iv_s_eq8(s, v, order))




# ### Speed test

# In[8]:

S = [2**i for i in range(1, 16)]

t_scipy = []
t_s = []
t_s_mpfr = []
t_s_eq7 = []
t_s_eq8 = []

for s in S:
    t1 = get_ipython().magic('timeit -oq iv_scipy(s,s)')
    t2 = get_ipython().magic('timeit -oq iv_s(s,s, tau)')
    t3 = get_ipython().magic('timeit -oq iv_s_mpfr(s,s, tau)')
    t4 = get_ipython().magic('timeit -oq iv_s_eq7(s,s)')
    t5 = get_ipython().magic('timeit -oq iv_s_eq8(s,s)')

    t_scipy.append(t1.best)
    t_s.append(t2.best)
    t_s_mpfr.append(t3.best)
    t_s_eq7.append(t4.best)
    t_s_eq8.append(t5.best)


# In[ ]:

plt.figure(figsize=(25, 25))
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
matplotlib.rcParams.update({'font.size': 24})
plt.plot(S, t_scipy, label ='Scipy')
plt.plot(S, t_s, label ='paper')
plt.plot(S, t_s_mpfr, label ='mpfr')
plt.plot(S, t_s_eq7, label ='eq7')
plt.plot(S, t_s_eq8, label ='eq8')
plt.legend(shadow=True, fancybox=True, fontsize=24)


# ### Precision test

# In[ ]:

S = [2, 22, 222, 2222, 22222]
best_tau = 1e-300

best = Parallel(n_jobs=8)(delayed(iv_s_mpfr)(s, s, tau=best_tau) for s in S)
order_max = 20

O = range(order_max)
eq7  = Parallel(n_jobs=8)([delayed(iv_s_eq7)(s, s, order) 
                            for order, s in it.product(O, S)])
rel_err = np.array(eq7).reshape(len(O), len(S)) / np.array(best)


# In[ ]:

plt.figure(figsize=(25, 25))
for s, err in zip(S, rel_err.T):
    plt.plot(O, err, label='s = {}'.format(s))
plt.legend(shadow=True, fancybox=True, fontsize=24)

