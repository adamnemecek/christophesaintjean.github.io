
# coding: utf-8

# In[1]:

import numpy as np
print np.__version__


# In[2]:

d = 2
stype = np.dtype([
            ("p1", np.double),
            ("p2", (np.double, (d, d)))
    ])


# In[3]:

K = 3
m1 = np.empty((), dtype=stype)
m1['p1'], m1['p2'] = 3., np.arange(d*d).reshape((d,d))
m2 = np.zeros(K, dtype=stype)
m3 = np.zeros(K, dtype=stype)

print m1
for k in range(K):
    print "-" * 5 + str(k) + "-" * 5
    for name in stype.names:
        print "-" * 5 + name + "-" * 5
        print "value", (k+1) * m1[name]
        m2[name][k] = (k+1) * m1[name]
        m3[k][name] = (k+1) * m1[name]
        print "m2 : ", m2[name], m2[name][k]
        print "m3 : ", m3[k], m3[k][name]


# In[ ]:



