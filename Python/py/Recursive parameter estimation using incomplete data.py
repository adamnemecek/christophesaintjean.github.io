
# coding: utf-8

# In[1]:

from sympy import *
from sympy.functions.special.tensor_functions import KroneckerDelta as indic
init_printing(use_latex=True)
#init_session()


# In[ ]:




# # General case
# ## Formulas

# In[ ]:

t1, t2, t3, t4, y, z, k = var('t1'), var('t2'), var('t3'), var('t4'), var('x'), var('z'), var('k')
w1, w2, w3 = Symbol('w1'), Symbol('w2'), Symbol('w3')
w4 = 1 - w1 - w2 - w3
g1 = Function('g1')(x, t1)
g2 = Function('g2')(x, t2)
g3 = Function('g3')(x, t3)
g4 = Function('g4')(x, t4)

# f : loi marginale sur x
# p : loi jointe sur x,z

f = w1*g1 + w2*g2 + w3*g3 + w4*g4
p = (w1*g1)**indic(z, 1) * (w2*g2)**indic(z, 2) * (w3*g3)**indic(z, 3) * (w4*g4)**indic(z, 4)
log_p  = indic(z, 1)*(log(w1)+log(g1))
log_p += indic(z, 2)*(log(w2)+log(g2))
log_p += indic(z, 3)*(log(w3)+log(g3))
log_p += indic(z, 4)*(log(w4)+log(g4))


params_th = [w1, w2, w3, t1, t2, t3, t4]
score_f = Matrix(7,1, [diff(log(f), param) for param in params_th])
score_p = Matrix(7,1, [diff(log_p, param) for param in params_th])
H_p = simplify(hessian(log(p), params_th))
## IC: Sans doute faux car il faut faire une intégration numérique.
#Ic = -simplify(w1*H_p.subs([(z, 1)])+ w2*H_p.subs([(z, 2)]) + w3*H_p.subs([(z, 3)]) + w4*H_p.subs([(z, 4)]))
#Ic_inv  = simplify(Ic**-1)
print(' ----- Gradient -------- ')
pprint(score_f)
print(' ----- Hessian -------- ')
pprint(H_p)
#print(' ----- Ic -------- ')
#pprint(Ic)
#print(' ----- invIc -------- ')
#pprint(Ic_inv)
#print(' ----- MAJ -------- ')
#maj = simplify (Ic_inv * score_f / k)
#for i, param in enumerate(params_th):
#    print(' ----- MAJ ' + str(param) + '-------')
#    pprint(maj[i])


# In[ ]:

pprint(H_p)


# ## Checks in the general case

# In[ ]:

print(' ----- Check Fisher Identity -------- ')
#pprint(score_p)
exp_score_p  = (w1*g1/f)*score_p.subs([(z, 1)])
exp_score_p += (w2*g2/f)*score_p.subs([(z, 2)])
exp_score_p += (w3*g3/f)*score_p.subs([(z, 3)])
exp_score_p += (w4*g4/f)*score_p.subs([(z, 4)])
exp_score_p = simplify(exp_score_p)
#pprint(exp_score_p)
pprint (simplify(score_f - exp_score_p))

print(' ----- Check inversion -------- ')
pprint(simplify(Ic * Ic_inv))

print(' ----- Check Updates -------- ')    
pprint(simplify(maj[0] - (w1*g1/f - w1)/k))
pprint(simplify(maj[1] - (w2*g2/f - w2)/k))
pprint(simplify(maj[2] - (w3*g3/f - w3)/k))
print(maj[3])


# ## Conclusion general case

# In[ ]:

print(' ----- Conclusion -------- ')
print(' [ w1 += k^-1 * (w1*g1/f - w1) ] ')
print(' [ w2 += k^-1 * (w2*g2/f - w2) ] ')
print(' [ w3 += k^-1 * (w3*g3/f - w3) ] ')
print(' [ w4 = 1 - w1 - w2 - w3 ] ')
print(' [(estZ1/w1)*H(F1)^-1*(s1(x) - grad F1(t1) ] ')
print(' [(estZ2/w2)*H(F2)^-1*(s2(x) - grad F2(t2) ] ')
print(' [(estZ3/w3)*H(F3)^-1*(s3(x) - grad F3(t3) ] ')
print(' [(estZ4/w4)*H(F4)^-1*(s4(x) - grad F4(t4) ] ')
print(' ----- Conclusion -------- ')


# # Regular exponential family case
# ## Formulas : natural parameters

# In[ ]:

t1, t2, t3, t4, y, z, k = var('t1'), var('t2'), var('t3'), var('t4'), var('x'), var('z'), var('k')
w1, w2, w3 = Symbol('w1'), Symbol('w2'), Symbol('w3')
w4 = 1 - w1 - w2 - w3
s1, k1, F1 = Function('s1')(x), Function('k1')(x), Function('F1')(t1)
g1 = exp(s1*t1 + k1 - F1)
s2, k2, F2 = Function('s2')(x), Function('k2')(x), Function('F2')(t2)
g2 = exp(s2*t2 + k2 - F2)
s3, k3, F3 = Function('s3')(x), Function('k3')(x), Function('F3')(t3)
g3 = exp(s3*t3 + k3 - F3)
s4, k4, F4 = Function('s4')(x), Function('k4')(x), Function('F4')(t4)
g4 = exp(s4*t4 + k4 - F4)

# f : loi marginale sur x
# p : loi jointe sur x,z

f = w1*g1 + w2*g2 + w3*g3 + w4*g4
p = (w1*g1)**indic(z, 1) * (w2*g2)**indic(z, 2) * (w3*g3)**indic(z, 3) * (w4*g4)**indic(z, 4)
log_p  = indic(z, 1)*(log(w1)+log(g1))
log_p += indic(z, 2)*(log(w2)+log(g2))
log_p += indic(z, 3)*(log(w3)+log(g3))
log_p += indic(z, 4)*(log(w4)+log(g4))

params_th = [w1, w2, w3, t1, t2, t3, t4]
score_f = Matrix(7,1, [diff(log(f), param) for param in params_th])
score_p = Matrix(7,1, [diff(log_p, param) for param in params_th])
H_p = simplify(hessian(log_p, params_th))
Ic = -simplify(w1*H_p.subs([(z, 1)])+ w2*H_p.subs([(z, 2)]) + w3*H_p.subs([(z, 3)]) + w4*H_p.subs([(z, 4)]))
Ic_inv  = Ic**-1
print(' ----- Gradient -------- ')
pprint(score_f)
print(' ----- Hessienne -------- ')
pprint(H_p)
print(' ----- Ic -------- ')
pprint(Ic)
print(' ----- invIc -------- ')
pprint(Ic_inv)

print(' ----- MAJ -------- ')
maj = (k*Ic)**-1 * score_f
for i, param in enumerate(params_th):
    #maj[i] = simplify(maj[i])
    print(' ----- MAJ ' + str(param) + '-------')
    pprint(maj[i])


# In[ ]:

pprint(simplify(score_f[3]))
pprint(simplify(maj[3]))


# ## Checks in the EF case

# In[ ]:

print(' ----- Check Fisher Identity -------- ')
#pprint(score_p)
exp_score_p  = (w1*g1/f)*score_p.subs([(z, 1)])
exp_score_p += (w2*g2/f)*score_p.subs([(z, 2)])
exp_score_p += (w3*g3/f)*score_p.subs([(z, 3)])
exp_score_p += (w4*g4/f)*score_p.subs([(z, 4)])
exp_score_p = simplify(exp_score_p)
#pprint(exp_score_p)
pprint (simplify(score_f - exp_score_p))

print(' ----- Check inversion -------- ')
pprint(simplify(Ic * Ic_inv))

print(' ----- Check Updates -------- ')    

pprint(simplify(maj[0] - (w1*g1/f - w1)/k))
pprint(simplify(maj[1] - (w2*g2/f - w2)/k))
pprint(simplify(maj[2] - (w3*g3/f - w3)/k))
pprint(simplify(maj[3] - ((g1/f)*(s1(x) - diff(F1, t1)))/(diff(F1,t1, t1)*k)))
pprint(simplify(maj[4] - ((g2/f)*(s2(x) - diff(F2, t2)))/(diff(F2,t2, t2)*k)))
pprint(simplify(maj[5] - ((g3/f)*(s3(x) - diff(F3, t3)))/(diff(F3,t3, t3)*k)))
pprint(simplify(maj[6] - ((g4/f)*(s4(x) - diff(F4, t4)))/(diff(F4,t4, t4)*k)))


# # Regular exponential family case
# ## Formulas : expectation space

# In[ ]:

e1, e2, e3, e4, y, z, k = var('e1'), var('e2'), var('e3'), var('e4'), var('x'), var('z'), var('k')
w1, w2, w3 = Symbol('w1'), Symbol('w2'), Symbol('w3')
w4 = 1 - w1 - w2 - w3
    
    
    
    
    
s1, k1, F1 = Function('s1')(x), Function('k1')(x), Function('Fs1')(e1)
g1 = exp(s1*t1 + k1 - F1)
s2, k2, F2 = Function('s2')(x), Function('k2')(x), Function('Fs2')(e2)
g2 = exp(s2*t2 + k2 - F2
s3, k3, F3 = Function('s3')(x), Function('k3')(x), Function('Fs3')(e3)
g3 = exp(s3*t3 + k3 - F3)
s4, k4, F4 = Function('s4')(x), Function('k4')(x), Function('Fs4')(e4)
g4 = exp(s4*t4 + k4 - F4)

# f : loi marginale sur x
# p : loi jointe sur x,z

f = w1*g1 + w2*g2 + w3*g3 + w4*g4
p = (w1*g1)**indic(z, 1) * (w2*g2)**indic(z, 2) * (w3*g3)**indic(z, 3) * (w4*g4)**indic(z, 4)
log_p  = indic(z, 1)*(log(w1)+log(g1))
log_p += indic(z, 2)*(log(w2)+log(g2))
log_p += indic(z, 3)*(log(w3)+log(g3))
log_p += indic(z, 4)*(log(w4)+log(g4))

params_th = [w1, w2, w3, t1, t2, t3, t4]
score_f = Matrix(7,1, [diff(log(f), param) for param in params_th])
score_p = Matrix(7,1, [diff(log_p, param) for param in params_th])
H_p = simplify(hessian(log_p, params_th))
Ic = -simplify(w1*H_p.subs([(z, 1)])+ w2*H_p.subs([(z, 2)]) + w3*H_p.subs([(z, 3)]) + w4*H_p.subs([(z, 4)]))
Ic_inv  = Ic**-1
print(' ----- Gradient -------- ')
pprint(score_f)
print(' ----- Hessienne -------- ')
pprint(H_p)
print(' ----- Ic -------- ')
pprint(Ic)
print(' ----- invIc -------- ')
pprint(Ic_inv)

print(' ----- MAJ -------- ')
maj = (k*Ic)**-1 * score_f
for i, param in enumerate(params_th):
    #maj[i] = simplify(maj[i])
    print(' ----- MAJ ' + str(param) + '-------')
    pprint(maj[i])


# # Example 3.4 : Mixture of two univariate normals (parametrization $\mu, \phi = \sigma^2$)

# In[2]:

w1, mu1, mu2, phi1, phi2, x, z, k = symbols('w1 mu1 mu2 phi1 phi2 x z k')
w2 = 1 - w1
def p(x, mu, phi):
    return exp(-(x-mu)**2/(2*phi))/sqrt(2*pi*phi)
  
g1 = p(x, mu1, phi1)
g2 = p(x, mu2, phi2)
#pprint(simplify(g1.subs([(t11, mu/(sigma**2)), (t12, 1/(2*sigma**2))])))

f = w1*g1 + w2*g2
p = (w1*g1)**indic(z, 1) * (w2*g2)**indic(z, 2)
log_p  = indic(z, 1)*(log(w1)+log(g1))
log_p += indic(z, 2)*(log(w2)+log(g2))

params_th = [w1, mu1, mu2, phi1, phi2]
score_f = simplify(Matrix(len(params_th),1, [diff(log(f), param) for param in params_th]))
score_p = simplify(Matrix(len(params_th),1, [diff(log_p, param) for param in params_th]))
H_p = simplify(hessian(log_p, params_th))

## Faux Integration sur x aussi 
#Ic = -simplify(w1*H_p.subs([(z, 1)])+ (1 - w1)*H_p.subs([(z, 2)]))
Ic = diag(1/(w1*(1-w1)), w1/phi1,(1-w1)/phi2, w1/(2*phi1**2),(1-w1)/(2*phi2**2))

Ic_inv  = Ic**-1
print(' ----- Gradient -------- ')
pprint(score_f[0])
print(' ----- Hessienne -------- ')
pprint(H_p)
print(' ----- Ic -------- ')
pprint(Ic)
print(' ----- invIc -------- ')
pprint(Ic_inv)

print(' ----- MAJ -------- ')
maj = (k*Ic)**-1 * score_f
for i, param in enumerate(params_th):
    #maj[i] = simplify(maj[i])
    print(' ----- MAJ ' + str(param) + '-------')
H_p    pprint(maj[i])


# In[5]:

pprint(H_p[3:,3:])
#pprint(integrate(g1,x).doit())
#e = simplify(diff(log_p,w1,w1)*p)
#pprint(e)
#int_z = integrate(e.subs([(z,1)]) + e.subs([(z,2)]),x)
#pprint(simplify(int_z))


# ##  Mixtures of 2 univariate normal (parametrization $\mu, \sigma$)

# In[ ]:

w1, mu1, mu2, sigma1, sigma2, x, z, k = symbols('w1 mu1 mu2 sigma1 sigma2 x z k')
w2 = 1 - w1
def p(x, mu, sigma):
    return exp(-(x-mu)**2/(2*sigma**2))/sqrt(2*pi*sigma**2)
  
g1 = p(x, mu1, sigma1)
g2 = p(x, mu2, sigma2)
#pprint(simplify(g1.subs([(t11, mu/(sigma**2)), (t12, 1/(2*sigma**2))])))

f = w1*g1 + w2*g2
p = (w1*g1)**indic(z, 1) * (w2*g2)**indic(z, 2)
log_p  = indic(z, 1)*(log(w1)+log(g1))
log_p += indic(z, 2)*(log(w2)+log(g2))

params_th = [w1, mu1, mu2, sigma1, sigma2]
score_f = simplify(Matrix(len(params_th),1, [diff(log(f), param) for param in params_th]))
score_p = simplify(Matrix(len(params_th),1, [diff(log_p, param) for param in params_th]))
H_p = factor(simplify(hessian(log_p, params_th)))
Ic = diag(1/(w1*(1-w1)), w1/sigma1**2,(1-w1)/sigma2**2, 2*w1/(sigma1**2),2*(1-w1)/(sigma2**2))
Ic_inv  = simplify(Ic**-1)
print(' ----- Gradient -------- ')
pprint(score_f[0])
print(' ----- Hessienne -------- ')
pprint(H_p)
print(' ----- Ic -------- ')
pprint(Ic)
print(' ----- invIc -------- ')
#pprint(Ic_inv)

print(' ----- MAJ -------- ')
maj = k**-1 * Ic_inv * score_f
for i, param in enumerate(params_th):
    #maj[i] = simplify(maj[i])
    print(' ----- MAJ ' + str(param) + '-------')
    pprint(maj[i])


# ##  Mixtures of 2 univariate normal (parametrization $\theta_1, \theta_2$)

# In[17]:

w1, th11, th21, th12, th22, x, z, k = symbols('w1 th11 th21 th12 th22 x z k')
w2 = 1 - w1

def s(x):
    return Matrix(2,1,[x,-x*x])

def F(theta1,theta2):
    return theta1*theta1/(4*theta2) + log(pi/theta2)/2

def N(x, theta1, theta2):
    theta = Matrix([theta1, theta2])
    ps = theta.T * s(x) 
    return exp(ps[0,0] - F(theta1,theta2))
  
g1 = N(x, th11, th12)
g2 = N(x, th21, th22)


#pprint(simplify(g1.subs([(t11, mu/(sigma**2)), (t12, 1/(2*sigma**2))])))

f = w1*g1 + w2*g2
p = (w1*g1)**indic(z, 1) * (w2*g2)**indic(z, 2)
log_p  = indic(z, 1)*(log(w1)+log(g1))
log_p += indic(z, 2)*(log(w2)+log(g2))

params_th = [w1, th11, th21, th12, th22]
score_f = simplify(Matrix(len(params_th),1, [diff(log(f), param) for param in params_th]))
score_p = simplify(Matrix(len(params_th),1, [diff(log_p, param) for param in params_th]))
H_p = simplify(hessian(log_p, params_th))

Ic = -simplify(w1*H_p.subs([(z, 1)])+ w2*H_p.subs([(z, 2)]))
Ic_3_direct = Ic
Ic_inv  = simplify(Ic**-1)
#Ic = diag(1/(w1*(1-w1)), w1/phi1,(1-w1)/phi2, 2*w1/(sigma1**2),2*(1-w1)/(sigma2**2))
#Ic_inv  = simplify(Ic**-1)
#print(' ----- Gradient -------- ')
pprint(score_f)
print(' ----- Hessienne -------- ')
pprint(H_p)
print(' ----- Ic -------- ')
pprint(Ic)
print(' ----- invIc -------- ')
pprint(Ic_inv)

print(' ----- MAJ -------- ')
#maj = k**-1 * Ic_inv * score_f
#for i, param in enumerate(params_th):
    #maj[i] = simplify(maj[i])
#    print(' ----- MAJ ' + str(param) + '-------')
#    pprint(maj[i])


# In[ ]:

F = th11*th11/(4*th12) + log(pi/th12)/2
pprint(hessian(F, [th11,th12]))
M = Matrix(2,2,[Ic[1,1], Ic[1,3], Ic[3,1], Ic[3,3]])
pprint(M)


# ##  Mixtures of 2 univariate normal (parametrization $\eta_1, \eta_2$)

# In[2]:

w1, et11, et21, et12, et22, x, z, k = symbols('w1 et11 et21 et12 et22 x z k')
w2 = 1 - w1

def s(x):
    return Matrix(2,1,[x,-x*x])

def F(theta1,theta2):
    return theta1*theta1/(4*theta2) + log(pi/theta2)/2

def N(x, eta1, eta2):
    theta1 = - eta1 / (eta1*eta1 + eta2)
    theta2 = - 1  / (2*(eta1*eta1 + eta2))
    theta = Matrix([theta1, theta2])
    ps = theta.T * s(x) 
    return exp(ps[0,0] - F(theta1,theta2))
  
g1 = N(x, et11, et12)
g2 = N(x, et21, et22)


#pprint(simplify(g1.subs([(t11, mu/(sigma**2)), (t12, 1/(2*sigma**2))])))

f = w1*g1 + w2*g2
p = (w1*g1)**indic(z, 1) * (w2*g2)**indic(z, 2)
log_p  = indic(z, 1)*(log(w1)+log(g1))
log_p += indic(z, 2)*(log(w2)+log(g2))

params_th = [w1, et11, et21, et12, et22]
score_f = simplify(Matrix(len(params_th),1, [diff(log(f), param) for param in params_th]))
score_p = simplify(Matrix(len(params_th),1, [diff(log_p, param) for param in params_th]))
H_p = simplify(hessian(log_p, params_th))

Ic = -simplify(w1*H_p.subs([(z, 1)])+ w2*H_p.subs([(z, 2)]))
Ic_4_direct = Ic
Ic_inv  = simplify(Ic**-1)
#Ic = diag(1/(w1*(1-w1)), w1/phi1,(1-w1)/phi2, 2*w1/(sigma1**2),2*(1-w1)/(sigma2**2))
#Ic_inv  = simplify(Ic**-1)
#print(' ----- Gradient -------- ')
pprint(score_f)
print(' ----- Hessienne -------- ')
pprint(H_p)
print(' ----- Ic -------- ')
pprint(Ic)
print(' ----- invIc -------- ')
pprint(Ic_inv)

print(' ----- MAJ -------- ')


# In[21]:

#pprint(H_p[1,1])
pprint(factor(simplify(w1*(et11**4 - 2*et11**3*et11 - 3*et11**2*et12 - 3*et11**2*et12 + 6*et11*et12*et11 + et12*et12)/(et11**6 + 3*et11**4*et12 + 3*et11**2*et12**2 + et12**3))))
#pprint(H_p[1,3])
pprint(factor(simplify(w1*(2*et11**3 - 3*et11**2*et11 - 2*et11*et12 + et12*et11)/(et11**6 + 3*et11**4*et12 + 3*et11**2*et12**2 + et12**3))))
#pprint(H_p[3,3])
pprint(factor(simplify(w1*(3*et11**2 - 4*et11*et11 + et12 - 2*et12)/(2*(et11**6 + 3*et11**4*et12 + 3*et11**2*et12**2 + et12**3)))))


# ## Changement de base et autres parametrisations

# In[18]:

w1, mu1, mu2, phi1, phi2 = symbols('w1 mu1 mu2 phi1 phi2')
dw1, dmu1, dmu2, dphi1, dphi2 = symbols('dw1 dmu1 dmu2 dphi1 dphi2')
dlb = Matrix(5,1, [dw1, dmu1, dmu2, dphi1, dphi2])
w2 = 1 - w1
coords_1 = [w1,mu1,mu2, phi1, phi2]
Ic_1 = diag(1/(w1*(1-w1)), w1/phi1,(1-w1)/phi2, w1/(2*phi1**2),(1-w1)/(2*phi2**2))
pprint(Ic_1)

# transform (w1, mu1, mu2, phi1, phi2) parameters to (w1, mu1, mu2, sigma1, sigma2)
sigma1, sigma2 = symbols('sigma1 sigma2')

Phi = Matrix([w1, mu1, mu2, sqrt(phi1), sqrt(phi2)])
actual = [w1,mu1,mu2, phi1, phi2]
P =  Phi.jacobian(coords_1)
Ic_2 = P.inv().T * Ic_1 * P.inv()
Ic_2_bis = simplify(Ic_2.subs([(phi1, sigma1**2), (phi2, sigma2**2)]))
Ic_2_direct = diag(1/(w1*(1-w1)), w1/sigma1**2,(1-w1)/sigma2**2, 2*w1/(sigma1**2),2*(1-w1)/(sigma2**2))
#pprint(P)
#pprint(Ic_2)  # ca marche !!!
#pprint(Ic_2_direct)
for i in range(5):
    for j in range(5):
        try:
            assert(simplify(Ic_2_bis[i,j] - Ic_2_direct[i,j]) == 0)
        except:
            pprint(Ic_2_bis[i,j])
            pprint(Ic_2_direct[i,j])


# transform (w1, mu1, mu2, phi1, phi2) parameters to (w1, th11, th12, th21, th22)
th11, th12, th21, th22 = symbols('th11 th12 th21 th22')
dth11, dth12, dth21, dth22 = symbols('dth11 dth12 dth21 dth22')
dth = Matrix(5,1, [dw1, dth11, dth21, dth12, dth22])

Phi = Matrix([w1, mu1/phi1, mu2/phi2, 1/(2*phi1), 1/(2*phi2)])
#Phi2 = Matrix([w1, th11/(2*th12), th21/(2*th22), 1/(2*th12), 1/(2*th22)])
P =  Phi.jacobian(coords_1)
Ic_3 = P.inv().T * Ic_1 * P.inv()
Ic_3_bis = simplify(Ic_3.subs([(mu1, th11/(2*th12)), (mu2, th21/(2*th22)), (phi1, 1/(2*th12)), (phi2,1/(2*th22))]))

pprint(Ic_3_direct)
for i in range(5):
    for j in range(5):
        try:
            assert(simplify(Ic_3_bis[i,j] - Ic_3_direct[i,j]) == 0)
        except:
            pprint(Ic_3_bis[i,j])
            pprint(Ic_3_direct[i,j])
            
# transform (w1, mu1, mu2, phi1, phi2) parameters to (w1, et11, et12, et21, et22)
et11, et12, et21, et22 = symbols('et11 et12 et21 et22')
Phi = Matrix([w1, mu1, mu2, - (mu1**2 + phi1), -(mu2**2 + phi2)])
P =  Phi.jacobian(coords_1)
Ic_4 = P.inv().T * Ic_1 * P.inv()
Ic_4_bis = simplify(Ic_4.subs([(mu1, et11), (mu2, et21), (phi1, -(et11**2+et12)), (phi2, -(et21**2+et22))]))
pprint(Ic_4_bis)


# ## Misc - Hessians

# In[ ]:

th1, th2 = symbols('th1 th2')
et1, et2 = symbols('et1 et2')
F = th1**2/(4*th2) + log(pi/th2)/2
F_star = -(log(pi) + 1 * log(-2*(et1**2+et2)))/2
H_F = simplify(hessian(F, [th1, th2]))
H_F_star = simplify(hessian(F_star, [et1, et2]))
#pprint(H_F)
#pprint(H_F)
#pprint(factor(simplify(H_F.inv()*Matrix(2,1,[x - th1/(2*th2),-x**2 + th1**2/(4*th2**2) - 1/(2*th2)])),x))
pprint(H_F_star)
pprint(simplify(H_F_star.inv().subs([(et1, th1/(2*th2) ),(et2, -th1**2/(4*th2**2) - 1/(2*th2))])))


# # Mixture of 4 univariate normals

# In[11]:

w1, w2, w3 = symbols('w1 w2 w3')
mu1, mu2, mu3, mu4 = symbols('mu1 mu2 mu3 mu4')
phi1, phi2, phi3, phi4 = symbols('phi1 phi2 phi3 phi4')
x, z, k = symbols('x z k')
w4 = 1 - w1 - w2 - w3
def p(x, mu, phi):
    return exp(-(x-mu)**2/(2*phi))/sqrt(2*pi*phi)
  
g1 = p(x, mu1, phi1)
g2 = p(x, mu2, phi2)
g3 = p(x, mu3, phi3)
g4 = p(x, mu4, phi4)
#pprint(simplify(g1.subs([(t11, mu/(sigma**2)), (t12, 1/(2*sigma**2))])))

f = w1*g1 + w2*g2 + w3*g3 + w4*g4
p = (w1*g1)**indic(z, 1) * (w2*g2)**indic(z, 2) * (w3*g3)**indic(z, 3) * (w4*g4)**indic(z, 4)
log_p  = indic(z, 1)*(log(w1)+log(g1))
log_p += indic(z, 2)*(log(w2)+log(g2))
log_p += indic(z, 3)*(log(w3)+log(g3))
log_p += indic(z, 4)*(log(w4)+log(g4))

params_th = [w1, w2, w3, mu1, mu2, mu3, mu4, phi1, phi2, phi3, phi4]
score_f = simplify(Matrix(len(params_th),1, [diff(log(f), param) for param in params_th]))
score_p = simplify(Matrix(len(params_th),1, [diff(log_p, param) for param in params_th]))
H_p = simplify(hessian(log_p, params_th))

## Faux Integration sur x aussi 
#Ic = -simplify(w1*H_p.subs([(z, 1)])+ (1 - w1)*H_p.subs([(z, 2)]))
#Ic = diag(1/(w1*(1-w1)), w1/phi1,(1-w1)/phi2, w1/(2*phi1**2),(1-w1)/(2*phi2**2))

#Ic_inv  = Ic**-1
print(' ----- Gradient -------- ')
pprint(score_f)
print(' ----- Hessienne -------- ')
pprint(H_p[:3,:3])
print(' ----- Ic -------- ')
#pprint(Ic)
#print(' ----- invIc -------- ')
#pprint(Ic_inv)

#print(' ----- MAJ -------- ')
#maj = (k*Ic)**-1 * score_f
#for i, param in enumerate(params_th):
#    #maj[i] = simplify(maj[i])
#    print(' ----- MAJ ' + str(param) + '-------')
#    pprint(maj[i])


# In[15]:

pprint(H_p[:3,:3])


# ## Mixture of Poisson distributions (Cappé Moulines)

# In[2]:

w1, w2, w3 = symbols('w1 w2 w3')
lambda1, lambda2, lambda3, lambda4 = symbols('lambda1 lambda2 lambda3 lambda4')
x, z, k = symbols('x z k')
w4 = 1 - w1 - w2 - w3
def p(x, lambda_p):
    return exp(-lambda_p)*lambda_p**x/gamma(x+1)
  
g1 = p(x, lambda1)
g2 = p(x, lambda2)
g3 = p(x, lambda3)
g4 = p(x, lambda4)
#pprint(simplify(g1.subs([(t11, mu/(sigma**2)), (t12, 1/(2*sigma**2))])))

f = w1*g1 + w2*g2 + w3*g3 + w4*g4
p = (w1*g1)**indic(z, 1) * (w2*g2)**indic(z, 2) * (w3*g3)**indic(z, 3) * (w4*g4)**indic(z, 4)
log_p  = indic(z, 1)*(log(w1)+log(g1))
log_p += indic(z, 2)*(log(w2)+log(g2))
log_p += indic(z, 3)*(log(w3)+log(g3))
log_p += indic(z, 4)*(log(w4)+log(g4))

params_th = [w1, w2, w3, lambda1, lambda2, lambda3, lambda4]
score_f = simplify(Matrix(len(params_th),1, [diff(log(f), param) for param in params_th]))
score_p = simplify(Matrix(len(params_th),1, [diff(log_p, param) for param in params_th]))
H_p = simplify(hessian(log_p, params_th))

un_3 = ones(3,1)
Ic = (1 / w4) * un_3 * un_3.T - diag(1/w1, 1/w2, 1/w3)
Ic = Ic.row_join(zeros(3,4))
M = zeros(4,3)
M = M.row_join(diag(w1/lambda1, w2/lambda2, w3/lambda3, w4/lambda4))
Ic = Ic.col_join(M)
Ic_inv  = simplify(Ic**-1)
print(' ----- Gradient -------- ')
#pprint(score_f)
print(' ----- Hessienne -------- ')
#pprint(H_p[:3,:3])
print(' ----- Ic -------- ')
pprint(Ic)
print(' ----- invIc -------- ')
pprint(Ic_inv)

#print(' ----- MAJ -------- ')
maj = simplify(k**-1 * Ic_inv * score_f)
for i, param in enumerate(params_th):
    #maj[i] = simplify(maj[i])
    print(' ----- MAJ ' + str(param) + '-------')
    pprint(maj[i])
    
## Formule


# In[ ]:

pprint(simplify(maj[0] - (w1*g1/f - w1)/k))
pprint(simplify(maj[1] - (w2*g2/f - w2)/k))
pprint(simplify(maj[2] - (w3*g3/f - w3)/k))
pprint(simplify(maj[3] - ((g1/f)*(s1(x) - diff(F1, t1)))/(diff(F1,t1, t1)*k)))
pprint(simplify(maj[4] - ((g2/f)*(s2(x) - diff(F2, t2)))/(diff(F2,t2, t2)*k)))
pprint(simplify(maj[5] - ((g3/f)*(s3(x) - diff(F3, t3)))/(diff(F3,t3, t3)*k)))
pprint(simplify(maj[6] - ((g4/f)*(s4(x) - diff(F4, t4)))/(diff(F4,t4, t4)*k)))


# ## Comparaison Online vs Recursive

# In[15]:

wn, wn1, znp1n, etn, sxnp1, alpha = symbols('wn wn1 znp1n etn sxnp1 alpha')
wnp1 = wn + alpha * (znp1n - wn)
etnp1 = collect(expand((wn*etn+alpha*(znp1n*sxnp1 - wn*etn))/wnp1), alpha)
pprint(etnp1)
etnp1_limit = etnp1.subs([(alpha,0)])
pprint(etnp1_limit)


# In[ ]:



