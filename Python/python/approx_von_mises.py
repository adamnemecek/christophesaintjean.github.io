
# coding: utf-8

# In[1]:

import sys
print (sys.version)


# In[22]:

from scipy.special import iv as besseli_scipy
from mpmath import besseli as besseli_mpmath, log
import matplotlib
get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from functools import lru_cache


# In[27]:

@lru_cache(maxsize=None)
def f_scsp(y):
    return besseli_scipy(1,y)/besseli_scipy(0,y)
def f_mpmath(y):
    return besseli_mpmath(1,y)/besseli_mpmath(0,y)
kappa = np.linspace(1e-6,100, 10000)
fig1 = plt.figure(1)
f_scsp_kappa = np.array([f_scsp(k) for k in kappa])
f_mpmath_kappa = np.array([f_mpmath(k) for k in kappa])
diff = [log(1+(f1-f2)) for f1, f2 in zip(f_scsp_kappa, f_mpmath_kappa)]
ax1 = fig1.add_subplot(121)
ax1.plot(kappa, f_scsp_kappa)
ax1.plot(kappa, f_mpmath_kappa)
ax2 = fig1.add_subplot(122)
ax2.plot(kappa, diff)


# In[5]:

def par_optim(a):
    f_scsp_inv = lambda kappa: f_scsp(kappa) - a
    k = brentq(f_scsp_inv,1e-6, 713.5, maxiter=100000)
    return k

def approximation_Amos(A):
    """
    Amos, D.E., 1974. Computation of modified Bessel function and their ratios. Math. Comp., 28: 239-242.
    """
    k = A*(0.5+np.sqrt(1.46*(1-A*A)+0.25))/(1-A*A)
    return k

def approximation_Dobson(A):
    """
    Approximation for k given by Dobson (1978)
    """
    k = (1.28 - 0.53*A*A) * np.tan(np.pi/2*A)
    return k


def approximation_Hussin(A):
    """
    Approximation for k given by Hussin and Mohamed (2008)
    """
    p = (3*(A-1)-2)/(24*((A-1.)**2))
    q = (4-9*(A-1)+54*((A-1)**2))/(432*((A-1.)**3))
    sqrtD = np.sqrt((p/3.)**3+(q/2.)**2)
    k = ((-q/2)+sqrtD)**(1./3.)+(-(q/2.)-sqrtD)**(1./3.)-1/(6*(A-1.))
    return k


# In[29]:

kappa = np.linspace(1e-6,25, 10000)
A = np.array([f_scsp(k) for k in kappa])
Ainv = []
for a in A:
    Ainv.append(par_optim(a))
    
fig2 = plt.figure(2)
ax1 = fig2.add_subplot(121)
ax1.plot(A, kappa,'-.')
ax1.plot(A, Ainv)
ax1.plot(A, approximation_Amos(A))
ax1.plot(A, approximation_Dobson(A))
ax1.plot(A, approximation_Hussin(A))
ax1.legend(['True', 'numeric', 'Amos', 'Dobson', 'Hussin'], loc='upper left')
ax2 = fig2.add_subplot(122)
ax2.plot(A, np.log(1+(Ainv-kappa)))
ax2.plot(A, np.log(1+(approximation_Amos(A)-kappa)))
ax2.plot(A, np.log(1+(approximation_Dobson(A)-kappa)))
ax2.plot(A, np.log(1+(approximation_Hussin(A)-kappa)))
ax2.legend(['numeric','Amos', 'Dobson', 'Hussin'], loc='lower left')


# In[30]:

def mse(kappa_est):
    return np.mean((kappa_est-kappa)**2)
def mre(kappa_est):
    return np.max(np.abs(kappa_est-kappa) / kappa_est)
    
mse_numeric, mre_numeric = mse(Ainv), mre(Ainv) 
mse_Amos, mre_Amos = mse(approximation_Amos(A)), mre(approximation_Amos(A))
mse_Dobson, mre_Dobson = mse(approximation_Dobson(A)), mre(approximation_Dobson(A)) 
mse_Hussin, mre_Hussin = mse(approximation_Amos(A)), mre(approximation_Hussin(A)) 
print (mse_numeric, mse_Amos, mse_Dobson, mse_Hussin)
print (mre_numeric, mre_Amos, mre_Dobson, mre_Hussin)


# In[ ]:

get_ipython().magic('timeit for a in A: par_optim(a)')
get_ipython().magic('timeit for a in A: approximation_Amos(a)')
get_ipython().magic('timeit approximation_Amos(A)')
get_ipython().magic('timeit for a in A: approximation_Dobson(a)')
get_ipython().magic('timeit approximation_Dobson(A)')
get_ipython().magic('timeit for a in A: approximation_Hussin(a)')
get_ipython().magic('timeit approximation_Hussin(A)')


# In[14]:

# Appendix 2.4 Mardia
for x in np.arange(0.01,1,0.01):
    print (x, par_optim(x))


# In[ ]:



