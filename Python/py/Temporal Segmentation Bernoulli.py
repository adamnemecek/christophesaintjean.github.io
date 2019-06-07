
# coding: utf-8

# ## Temporal segmentation of Bernoulli time series
# 
# Implementation pour Bernoulli de :
# "Online change detection in exponential families with unknown parameters"
# Arnaud Dessein, Arshia Cont
# 
# 
# 

# In[ ]:

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sympy import symbols, log, exp, diff, solve, integrate, simplify
import itertools as it


# In[ ]:

theta, eta = symbols('theta eta', positive=True)
F = log(1 + exp(theta))
print(F)
gradF = diff(F, theta)
print('GradF :', gradF)
gradG = solve(gradF - eta, theta)
try:
    G = simplify(integrate(gradG, eta))
except:
    gradG = log(eta/(1-eta))
    G = simplify(integrate(gradG, eta))
print('GradG :', gradG)
print('G :', G)
G_n = eta*log(eta/(1 - eta)) + log(1 - eta)
print('G_n :', G_n)
gradG_n = diff(G_n, eta)
gradG_bis = diff(G, eta)
print('Difference entre les expressions : ', simplify(gradG_n - gradG_bis))

def F_star(x):
    return x* np.log(x / (1 - x)) + np.log(1 - x)


# In[ ]:

cuts  = [500, 700]
probs = [0.4, 0.6]
data = [st.bernoulli(p).rvs(size=n) for p, n in zip(probs, np.diff([0] + cuts))]
probs_est = [np.mean(seg) for seg in data]    # mle
print(probs_est)
data = np.concatenate(data).astype(np.float)
suff_stat_bernoulli = data
cum_suff_stat_bernoulli = np.cumsum(suff_stat_bernoulli)

N = len(data)
i = np.arange(1, N)
eta_inf_i = np.cumsum(suff_stat_bernoulli[:-1]) / i
eta_sup_i = np.cumsum(suff_stat_bernoulli[-1:0:-1]) / i
eta_sup_i = eta_sup_i[::-1]
eta_n = np.mean(suff_stat_bernoulli)
#print(eta_inf_i)
#print(F_star(eta_inf_i))
GLRT = 2 * (i * F_star(eta_inf_i) + (N-i) * F_star(eta_sup_i) - N * F_star(eta_n))
GLRT_fix = np.ma.masked_invalid(GLRT)

print(np.argmax(GLRT_fix), np.nanmax(GLRT_fix) )
f, ax1 = plt.subplots()
ax1.plot(cum_suff_stat_bernoulli, '-b')
ax1.set_xlim(0,N)
ax1.set_ylim(0,N)
ax1.set_ylabel('cumsum', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(i, GLRT,'-r')
ax2.set_ylabel('GLRT', color='r')
ax2.set_ylim(0,40)
ax2.tick_params('y', colors='r')


# In[ ]:

#cuts  = [500, 700] #, 1000, 1500]
#probs = [0.15, 0.8] #, 0.5, 0.9]
#data = [st.bernoulli(p).rvs(size=n) for p, n in zip(probs, np.diff([0] + cuts))]
#probs_est = [np.mean(seg) for seg in data]    # mle
#print(probs_est)
#data = np.concatenate(data).astype(np.float)
#suff_stat_bernoulli = data
#cum_suff_stat_bernoulli = np.cumsum(suff_stat_bernoulli)


# In[ ]:

def GLRT_bernoulli_gen(data):
    GLRT = np.zeros(len(data))
    eta_0 = 0
    eta_inf = np.zeros(len(data)-1)
    eta_sup = np.zeros(len(data)-1)
    for n, x in enumerate(data):
        if n == 0:  
            eta_0 = x
            continue
        # new split 
        eta_inf[n-1] = eta_0  # n-1 elements
        eta_sup[n-1] = x      # 1 element
        # update previous splits
        eta_sup[:(n-1)] += (x - eta_sup[:(n-1)]) / np.arange(n,1,-1)
        # Update no split
        eta_0 += (x - eta_0) / (n + 1)
        n_inf = np.arange(1, n+1)
        n_sup = np.arange(n,0,-1)
        GLRT[:n] = 2 * ( n_inf * F_star(eta_inf[:n]) + 
                         n_sup * F_star(eta_sup[:n]) -
                         (n+1) * F_star(eta_0))
        yield n, GLRT
    while True:
        yield n, GLRT
        


# In[ ]:

from matplotlib import animation, rc
from IPython.display import HTML

f, ax1 = plt.subplots()
ax1.plot(np.cumsum(data), '-b')
ax1.set_xlim(0,len(data))
ax1.set_ylim(0,len(data))
ax1.set_ylabel('cumsum', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
line_GLRT, = ax2.plot(np.ones_like(data),'-r', animated=True)
ax2.set_ylabel('GLRT', color='r')
ax2.set_ylim(0,40)
ax2.tick_params('y', colors='r')


def GLRT_update(onlinedata):
    n, GLRT = onlinedata
    line_GLRT.set_ydata(GLRT)
    return line_GLRT,

anim = animation.FuncAnimation(f, func=GLRT_update, 
                               frames=GLRT_bernoulli_gen(data),
                               repeat=False,
                               interval=20,
                               save_count=len(data)-1)  
HTML(anim.to_html5_video())


# In[ ]:



