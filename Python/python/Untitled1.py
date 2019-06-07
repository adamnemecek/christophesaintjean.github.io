
# coding: utf-8

# # Friedval's algorithm
# Randomized algorithm for verify matrix multiplication

# In[1]:

import numpy as np
from numba import jit


# In[2]:

def standard(a, b, c):
    return np.allclose(np.dot(a, b), c)

def freivalds_1(a, b, c, k=1):
    n, m = b.shape 
    for _ in range(k):
        r = np.random.binomial(1, 0.5, m)
        if not np.allclose(np.dot(a, np.dot(b,r)), np.dot(c,r)):
            return False
    return True  # should be a maybe


def freivalds_2(a, b, c, k=1):
    n, m = b.shape
    l, dummy = a.shape
    assert (dummy == n)
    for _ in range(k):
        r = np.random.binomial(1, 0.5, m)
        br = np.dot(b, r)
        for al, cl in zip(a, c):
            if not np.allclose(np.dot(al,br), np.dot(cl,r)):
                return False
    return True  # should be a maybe

@jit
def freivalds_1_numba(a, b, c, k=1):
    n, m = b.shape 
    for _ in range(k):
        r = np.random.binomial(1, 0.5, m)
        if not np.allclose(np.dot(a, np.dot(b,r)), np.dot(c,r)):
            return False
    return True  # should be a maybe

@jit
def freivalds_2_numba(a, b, c, k=1):
    n, m = b.shape
    l, dummy = a.shape
    assert (dummy == n)
    for _ in range(k):
        r = np.random.binomial(1, 0.5, m)
        br = np.dot(b, r)
        for al, cl in zip(a, c):
            if not np.allclose(np.dot(al,br), np.dot(cl,r)):
                return False
    return True  # should be a maybe


# In[4]:

n = 10000
fail = None

def test(n, method, fail=None, **kwargs):
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = np.dot(a,b)
    if fail:
        nb = np.random.choice(n*n, np.uint(fail*n*n)
        c[nb] += 100000
    return method(a, b, c, **kwargs)

#test(n, freivalds_2_numba, fail, k=10)
    
    
%timeit test(n, standard, fail)
%timeit test(n, freivalds_1, fail, k=10)
%timeit test(n, freivalds_1_numba, fail, k=10)
%timeit test(n, freivalds_2, fail, k=10)
%timeit test(n, freivalds_2_numba, fail, k=10)
    
    


# In[12]:

a = np.random.rand(3, 3)
print(a)
mask = np.random.choice(9, np.uint(0.25*9))
print(mask)
print(a._ix(mask))


# In[ ]:



