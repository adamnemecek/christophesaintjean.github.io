
# coding: utf-8

# # [Isserlis Theorem](http://en.wikipedia.org/wiki/Isserlis%27_theorem "Page on Wikipedia")
# 

# In[12]:

import itertools as it

def group(iterable, n=2):
    return zip(*([iter(iterable)] * n))

def set_partition(iterable, n=2):
    set_partitions = set()

    for permutation in it.permutations(iterable):
        grouped = group(list(permutation), n)
        sorted_group = tuple(sorted([tuple(sorted(partition)) for partition in grouped]))
        set_partitions.add(sorted_group)

    return set_partitions

partitions = set_partition([0,1,2,3,4,5], 2)
for part in partitions:
    print(part)
    


# ## The unknows
# 
# [http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Higher_moments]

# In[14]:

from sympy import MatrixSymbol, symbols
dim = 6
Sigma = MatrixSymbol('S', dim, dim)
X = symbols('X0:%d'%dim)
print X
def mu_expand(var_list, order_list):
    assert(len(var_list) == len(order_list))
    orders = list(it.chain(*(it.repeat(var, order) 
                            for var, order in zip(var_list, order_list))))
    return orders

def mu(var_list=X, order_list=[1]*dim):
    orders = mu_expand(var_list, order_list)
    index_orders = range(len(orders))
    partitions = set_partition(index_orders, 2)
    for part in partitions:
        print(part)
        
mu(X,(1,1,1,1,1,1))
 


# In[55]:

[555]*0


# In[ ]:




# In[73]:

def set_partition(iterable, n=2):
    def rec_set_partition(atuple, tail):
        print "atuple", atuple, tail
        if alist == []:
            return
        else:
            first = atuple[0]
            for completion in it.combinations(atuple[1:], n-1):
                element = tuple(it.chain((first,), completion))
                print first, ' ---- ', element
                other = rec_set_partition(alist[1:], tail.append(element))
                yield (element, *other)
    return rec_set_partition(list(iterable), [])
partitions = set_partition([1,2,3,4,5,6], 2)
for part in partitions:
    print(part)


# In[81]:

print [_ for _ in it.combinations(range(4), 2)]
print [_ for _ in it.combinations(range(4),4)]



# In[ ]:



