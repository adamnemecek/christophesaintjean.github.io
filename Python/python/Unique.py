
# coding: utf-8

# # Test speed unique

# In[5]:

from itertools import chain


# In[1]:

a = ['a', 'b', 'c', 'd']
b = ['c', 'x', 'g', 'h']
c = ['1', 'as', 'ci', 'v']
d = ['1', '2', '3', '4']


# In[7]:

get_ipython().magic('timeit list(set().union(a, b, c, d))')
get_ipython().magic('timeit list({key: None for key in chain(a,b,c,d)})')


# In[ ]:



