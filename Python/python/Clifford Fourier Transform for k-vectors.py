
# coding: utf-8

# # Clifford Fourier Transform for k-vectors
# 
# This notebook shows how to use the outermorphism property for encoding a Fourier Transform on k-vectors
# $$e^{\frac{1}{2}B} ( v_1 \wedge v_2 ) e^{-\frac{1}{2}B} = e^{\frac{1}{2}B}v_1e^{-\frac{1}{2}B}\wedge e^{\frac{1}{2}B}v_2e^{-\frac{1}{2}B}$$

# In[10]:

import sympy.galgebra.ga as ga


# In[5]:

n = 3


# In[12]:

metric = ga.diagpq(4,0)
print(metric)


# In[7]:

ls()


# In[9]:

help(GA)


# In[ ]:



