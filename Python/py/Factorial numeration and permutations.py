
# coding: utf-8

# # Factorial numeration and permutations
# 
# Implementation of article:  
# 
# Laisant, Charles-Ange (1888), "Sur la numération factorielle, application aux permutations", Bulletin de la Société Mathématique de France (in French) 16: 176–183.
# 
# http://archive.numdam.org/ARCHIVE/BSMF/BSMF_1888__16_/BSMF_1888__16__176_0/BSMF_1888__16__176_0.pdf

# In[1]:

from math import factorial


# In[2]:

def number2pattern(N, n = 0):
    pattern = []
    if n == 0:
        i = 2
        while N > 0:
            pattern = [N % i] + pattern
            N = N // i
            i += 1
    else:
        for i in range(2,n+1):
            pattern = [N % i] + pattern
            N = N // i    
    return pattern

def number2pattern_dec(N,n):
    pattern = []
    for i in range(n,0,-1):
        pattern.append(N // factorial(i))
        N = N % factorial(i)
    return pattern

N = 303805
alpha = number2pattern(N)
n = len(alpha)
print(alpha, number2pattern_dec(N,n))
assert (N == sum(alpha[i]*factorial(n-i) for i in range(n)))

print(number2pattern(4))     # automatic setting of n
print(number2pattern(4, 5))  # manual setting of n


# In[3]:

def pattern2permutation(pattern, s = None):
    pattern.append(0)
    n = len(pattern)
    if s is None:
        s = list(range(1,n+1)) # integers to permute
    else:
        assert len(s) == n
    permutation = []
    for p_i in pattern:
        permutation.append(s[p_i]) 
        del s[p_i]
    return permutation

N = 5
alpha = number2pattern(N)
print(alpha)
print(pattern2permutation(alpha, ['a', 'b', 'c']))


# ## Enumerate all n-permutations

# In[5]:

n = 3
for N in range(factorial(n)):
    print(pattern2permutation(number2pattern(N,n)))


# In[ ]:




# In[ ]:



