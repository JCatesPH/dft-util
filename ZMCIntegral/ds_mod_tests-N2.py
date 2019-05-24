#!/usr/bin/env python
# coding: utf-8

# # Test to ensure modifications to function do not change evaluated value
# 
# Beginning modification of the function to handle case where N is not equal to 1.

# In[11]:


import math
from numba import cuda
import ZMCIntegral
import time
import numpy as np
import scipy
import scipy.special
from scipy.integrate import quad


# Define constants in function

# In[12]:


mu = 0.1  # Fermi-level
hOmg = 0.5  # Photon energy eV
a = 4  # AA
A = 4  # hbar^2/(2m)=4 evAA^2 (for free electron mass)
rati = 0.3  # ratio
eE0 = rati * ((hOmg) ** 2) / (2 * np.sqrt(A * mu))
# print(eE0)
Gamm = 0.005  # Gamma in eV.
KT = 1 * 10 ** (-6)
shift = A * (eE0 / hOmg) ** 2


# The original function is declared below.

# In[25]:


def Ds(kx, ky, qx, qy):
    N = 2
    dds = 0
    ds = 0
    ek = A * (math.sqrt((kx) ** 2 + (ky) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((kx + qx) ** 2 + (ky + qy) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((kx) ** 2 + (ky) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((kx + qx) ** 2 + (ky + qy) ** 2) / hOmg ** 2

    sing = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1)
    taninv1kp = 2 * np.arctan2(Gamm, ek - hOmg / 2 + hOmg * sing)
    taninv1kqp = 2 * np.arctan2(Gamm, ekq - hOmg / 2 + hOmg * sing)
    taninv1km = 2 * np.arctan2(Gamm, ek + hOmg / 2 + hOmg * sing)
    taninv1kqm = 2 * np.arctan2(Gamm, ekq + hOmg / 2 + hOmg * sing)

    lg1kp = complex(0, 1) * np.log(Gamm ** 2 + (ek - hOmg / 2 + hOmg * sing) ** 2)
    lg1kqp = complex(0, 1) * np.log(Gamm ** 2 + (ekq - hOmg / 2 + hOmg * sing) ** 2)
    lg1km = complex(0, 1) * np.log(Gamm ** 2 + (ek + hOmg / 2 + hOmg * sing) ** 2)
    lg1kqm = complex(0, 1) * np.log(Gamm ** 2 + (ekq + hOmg / 2 + hOmg * sing) ** 2)

    ferp = np.heaviside(mu - hOmg / 2 - hOmg * sing, 0)
    ferm = np.heaviside(mu + hOmg / 2 - hOmg * sing, 0)

    dbl = np.arange(-(N - 1), (N - 1) + 1, 1)
    taninv2k = 2 * np.arctan2(Gamm, ek - mu + hOmg * dbl)
    taninv2kq = 2 * np.arctan2(Gamm, ekq - mu + hOmg * dbl)

    lg2k = complex(0, 1) * np.log(Gamm ** 2 + (ek - mu + hOmg * dbl) ** 2)
    lg2kq = complex(0, 1) * np.log(Gamm ** 2 + (ekq - mu + hOmg * dbl) ** 2)

    besk = scipy.special.jv(dbl, xk)
    beskq = scipy.special.jv(dbl, xkq)

    fac1 = ek - ekq + hOmg * dbl
    fac2 = fac1 + 2 * complex(0, 1) * Gamm
    fac3 = fac2 - ek + ekq
    
    # DEBUGGING
    # print('taninv2k = ', taninv2k)
    # print('taninv1kp = ', taninv1kp)
    # print('lg2k = ', lg2k)
    # print('lg1kp = ', lg1kp)
    # print('fac3 = ', fac3)
    print('besk = ', besk)
    print('beskq = ', beskq)
    
    for n in range(0, N):
        for alpha in range(0, N):
            for beta in range(0, N):
                for gamma in range(0, N):
                    for s in range(0, N):
                        for l in range(0, N):
                            p1p = fac1[beta - gamma + N - 1] * (
                                    taninv1kp[alpha] - taninv2k[s + alpha] - lg1kp[alpha] + lg2k[s + alpha])
                            p2p = fac2[alpha - gamma + N - 1] * (
                                    taninv1kp[beta] - taninv2k[s + beta] + lg1kp[beta] - lg2k[s + beta])
                            p3p = fac3[alpha - beta + N - 1] * (
                                    -taninv1kqp[gamma] + taninv2kq[s + gamma] - lg1kqp[gamma] + lg2kq[
                                s + gamma])

                            p1m = fac1[beta - gamma + N - 1] * (
                                    taninv1km[alpha] - taninv2k[s + alpha] - lg1km[alpha] + lg2k[s + alpha])

                            p2m = fac2[alpha - gamma + N - 1] * (
                                    taninv1km[beta] - taninv2k[s + beta] + lg1km[beta] - lg2k[s + beta])

                            p3m = fac3[alpha - beta + N - 1] * (
                                    -taninv1kqm[gamma] + taninv2kq[s + gamma] - lg1kqm[gamma] + lg2kq[
                                s + gamma])

                            d1 = -2 * complex(0, 1) * fac1[beta - gamma + N - 1] * fac2[alpha - gamma + N - 1] *                                  fac3[
                                     alpha - beta + N - 1]
                            
                            omint1p = ferp[s] * ((p1p + p2p + p3p) / d1)

                            omint1m = ferm[s] * ((p1m + p2m + p3m) / d1)

                            bess1 = beskq[gamma - n + N - 1] * beskq[gamma - l + N - 1] * besk[beta - l + N - 1] * besk[
                                beta - s + N - 1] * besk[alpha - s + N - 1] * besk[alpha - n + N - 1]

                            grgl = bess1 * (omint1p - omint1m)

                            pp1p = fac1[alpha - beta + N - 1] * (
                                    -taninv1kqp[gamma] + taninv2kq[s + gamma] - lg1kqp[gamma] + lg2kq[
                                s + gamma])

                            pp2p = fac2[alpha - gamma + N - 1] * (
                                    -taninv1kqp[beta] + taninv2kq[s + beta] + lg1kqp[beta] - lg2kq[
                                s + beta])

                            pp3p = fac3[beta - gamma + N - 1] * (
                                    taninv1kp[alpha] - taninv2k[s + alpha] - lg1kp[alpha] + lg2k[s + alpha])

                            pp1m = fac1[alpha - beta + N - 1] * (
                                    -taninv1kqm[gamma] + taninv2kq[s + gamma] - lg1kqm[gamma] + lg2kq[
                                s + gamma])

                            pp2m = fac2[alpha - gamma + N - 1] * (
                                    -taninv1kqm[beta] + taninv2kq[s + beta] + lg1kqm[beta] - lg2kq[
                                s + beta])

                            pp3m = fac3[beta - gamma + N - 1] * (
                                    taninv1km[alpha] - taninv2k[s + alpha] - lg1km[alpha] + lg2k[s + alpha])

                            d2 = -2 * complex(0, 1) * fac1[alpha - beta + N - 1] * fac2[alpha - gamma + N - 1] *                                  fac3[
                                     beta - gamma + N - 1]

                            omint2p = ferp[s] * ((pp1p + pp2p + pp3p) / d2)

                            omint2m = ferm[s] * ((pp1m + pp2m + pp3m) / d2)

                            bess2 = beskq[gamma - n + N - 1] * beskq[gamma - s + N - 1] * beskq[beta - s + N - 1] *                                     beskq[beta - l + N - 1] * besk[alpha - l + N - 1] * besk[alpha - n + N - 1]

                            glga = bess2 * (omint2p - omint2m)

                            dds = dds + 2 * Gamm * (grgl + glga)
                            #DEBUG
                            # print('dds = ', dds)
                            # print('bess1=',  bess1)
    return dds


# In[34]:


def my_Besselv(v, z):
    # WILL NOT WORK IF v IS NOT AN INTEGER
    # Conditional to handle case of negative v.
    if(v < 0):
        v = abs(v)
        resultsign = (-1) ** v
    else:
        resultsign = 1
    result = 0    
    # Loop to construct Bessel series sum.
    for n in range(0,20):
        sign = (-1)**n
        exp = 2 * n + v
        term = z ** exp
        r = n + v + 1
        if(r == 0):
            r = 1e-15
        denom = math.gamma(r)
        denom = denom * math.factorial(n)
        denom = denom * (2 ** exp)
        term = term / denom * sign
        # print('for ', n, ': ',term)
        result = result + term
        
    return result * resultsign
        
    
def myHeaviside(z):
    # Wrote this Heaviside expression with it cast in cuda to avoid error below.
    if z <= 0 :
	    return 0
    else :
	    return 1
    
    
def modDsN2(x):
    N = 2
    dds = 0
    ds = 0
    ek = A * (math.sqrt((x[0]) ** 2 + (x[2]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((x[0] + x[2]) ** 2 + (x[2] + 0) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((x[0]) ** 2 + (x[2]) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((x[0] + x[2]) ** 2 + (x[2] + 0) ** 2) / hOmg ** 2

    j = 0
    i = -(N - 1) / 2
    
    taninv1kp = np.empty(N)
    taninv1kqp = np.empty(N)
    taninv1km = np.empty(N)
    taninv1kqm = np.empty(N)

    lg1kp = np.empty(N, dtype='complex')
    lg1kqp = np.empty(N, dtype='complex')
    lg1km = np.empty(N, dtype='complex')
    lg1kqm = np.empty(N, dtype='complex')

    ferp = np.empty(N)
    ferm = np.empty(N)
    
    while(i < ((N - 1) / 2 + 1)):
        taninv1kp[j] = 2 * math.atan2(Gamm, ek - hOmg / 2 + hOmg * i)
        taninv1kqp[j] = 2 * math.atan2(Gamm, ekq - hOmg / 2 + hOmg * i)
        taninv1km[j] = 2 * math.atan2(Gamm, ek + hOmg / 2 + hOmg * i)
        taninv1kqm[j] = 2 * math.atan2(Gamm, ekq + hOmg / 2 + hOmg * i)

        lg1kp[j] = complex(0, 1) * math.log(Gamm ** 2 + (ek - hOmg / 2 + hOmg * i) ** 2)
        lg1kqp[j] = complex(0, 1) * math.log(Gamm ** 2 + (ekq - hOmg / 2 + hOmg * i) ** 2)
        lg1km[j] = complex(0, 1) * math.log(Gamm ** 2 + (ek + hOmg / 2 + hOmg * i) ** 2)
        lg1kqm[j] = complex(0, 1) * math.log(Gamm ** 2 + (ekq + hOmg / 2 + hOmg * i) ** 2)

        ferp[j] = myHeaviside(mu - hOmg / 2 - hOmg * i)
        ferm[j] = myHeaviside(mu + hOmg / 2 - hOmg * i)
        j = j + 1
        i = i + 1
        
    size_dbl = 2 * N - 1
    
    taninv2k = np.empty(size_dbl)
    taninv2kq = np.empty(size_dbl)

    lg2k = np.empty(size_dbl, dtype='complex')
    lg2kq = np.empty(size_dbl, dtype='complex')

    besk = np.empty(size_dbl)
    beskq = np.empty(size_dbl)

    fac1 = np.empty(size_dbl)
    fac2 = np.empty(size_dbl, dtype='complex')
    fac3 = np.empty(size_dbl, dtype='complex')

    j = 0
    for i in range(-(N - 1), (N - 1) + 1, 1):
        taninv2k[j] = 2 * math.atan2(Gamm, ek - mu + hOmg * i)
        taninv2kq[j] = 2 * math.atan2(Gamm, ekq - mu + hOmg * i)

        lg2k[j] = complex(0, 1) * math.log(Gamm ** 2 + (ek - mu + hOmg * i) ** 2)
        lg2kq[j] = complex(0, 1) * math.log(Gamm ** 2 + (ekq - mu + hOmg * i) ** 2)

        besk[j] = my_Besselv(i, xk)
        beskq[j] = my_Besselv(i, xkq)

        fac1[j] = ek - ekq + hOmg * i
        fac2[j] = fac1[j] + 2 * complex(0, 1) * Gamm
        fac3[j] = fac2[j] - ek + ekq
        j = j + 1
    
    # debug statements
    # print('taninv2k = ', taninv2k)
    # print('taninv1kp = ', taninv1kp)
    # print('lg2k = ', lg2k)
    # print('lg1kp = ', lg1kp)
    # print('fac3 = ', fac3)
    print('modbesk = ', besk)
    print('modbeskq = ', beskq)
    
    for n in range(0, N):
        for alpha in range(0, N):
            for beta in range(0, N):
                for gamma in range(0, N):
                    for s in range(0, N):
                        for l in range(0, N):
                            p1p = fac1[beta - gamma + N - 1] * (
                                    taninv1kp[alpha] - taninv2k[s + alpha] - lg1kp[alpha] + lg2k[s + alpha])
                            p2p = fac2[alpha - gamma + N - 1] * (
                                    taninv1kp[beta] - taninv2k[s + beta] + lg1kp[beta] - lg2k[s + beta])
                            p3p = fac3[alpha - beta + N - 1] * (
                                    -taninv1kqp[gamma] + taninv2kq[s + gamma] - lg1kqp[gamma] + lg2kq[
                                s + gamma])

                            p1m = fac1[beta - gamma + N - 1] * (
                                    taninv1km[alpha] - taninv2k[s + alpha] - lg1km[alpha] + lg2k[s + alpha])

                            p2m = fac2[alpha - gamma + N - 1] * (
                                    taninv1km[beta] - taninv2k[s + beta] + lg1km[beta] - lg2k[s + beta])

                            p3m = fac3[alpha - beta + N - 1] * (
                                    -taninv1kqm[gamma] + taninv2kq[s + gamma] - lg1kqm[gamma] + lg2kq[
                                s + gamma])

                            d1 = -2 * complex(0, 1) * fac1[beta - gamma + N - 1] * fac2[alpha - gamma + N - 1] *                                  fac3[
                                     alpha - beta + N - 1]

                            omint1p = ferp[s] * ((p1p + p2p + p3p) / d1)

                            omint1m = ferm[s] * ((p1m + p2m + p3m) / d1)

                            bess1 = beskq[gamma - n + N - 1] * beskq[gamma - l + N - 1] * besk[beta - l + N - 1] * besk[
                                beta - s + N - 1] * besk[alpha - s + N - 1] * besk[alpha - n + N - 1]

                            grgl = bess1 * (omint1p - omint1m)

                            pp1p = fac1[alpha - beta + N - 1] * (
                                    -taninv1kqp[gamma] + taninv2kq[s + gamma] - lg1kqp[gamma] + lg2kq[
                                s + gamma])

                            pp2p = fac2[alpha - gamma + N - 1] * (
                                    -taninv1kqp[beta] + taninv2kq[s + beta] + lg1kqp[beta] - lg2kq[
                                s + beta])

                            pp3p = fac3[beta - gamma + N - 1] * (
                                    taninv1kp[alpha] - taninv2k[s + alpha] - lg1kp[alpha] + lg2k[s + alpha])

                            pp1m = fac1[alpha - beta + N - 1] * (
                                    -taninv1kqm[gamma] + taninv2kq[s + gamma] - lg1kqm[gamma] + lg2kq[
                                s + gamma])

                            pp2m = fac2[alpha - gamma + N - 1] * (
                                    -taninv1kqm[beta] + taninv2kq[s + beta] + lg1kqm[beta] - lg2kq[
                                s + beta])

                            pp3m = fac3[beta - gamma + N - 1] * (
                                    taninv1km[alpha] - taninv2k[s + alpha] - lg1km[alpha] + lg2k[s + alpha])

                            d2 = -2 * complex(0, 1) * fac1[alpha - beta + N - 1] * fac2[alpha - gamma + N - 1] *                                  fac3[
                                     beta - gamma + N - 1]

                            omint2p = ferp[s] * ((pp1p + pp2p + pp3p) / d2)

                            omint2m = ferm[s] * ((pp1m + pp2m + pp3m) / d2)

                            bess2 = beskq[gamma - n + N - 1] * beskq[gamma - s + N - 1] * beskq[beta - s + N - 1] *                                     beskq[beta - l + N - 1] * besk[alpha - l + N - 1] * besk[alpha - n + N - 1]

                            glga = bess2 * (omint2p - omint2m)

                            dds = dds + 2 * Gamm * (grgl + glga)
                            
                            #DEBUG
                            # print('moddds = ', dds)
                            # print('modbess1=',  bess1)
    return dds.real


# # The modified function
# 
# The function is vectorized again to increase efficiency.
# 
# The custom Bessel function should now handle any-order*, first-kind Bessel function. *(Integer-order!)

# In[36]:


# Make error array.
relerror = np.zeros(1250)

print('Comparing modified version and original')
print('================================================================================================')
print(' kx  | ky  | qx  | qy  | Ds          | modDs  ')
print('================================================================================================')
for i in range(0, 75, 5):
    for j in range(0, 75, 5):
        for k in range(1, 4, 1):
            xin = [i/10, j/10, k/10]
            ds_result = Ds(i/10, j/10, k/10, 0).real
            modds_result = modDsN2(xin)
            print('%2.1f  | %2.1f | %2.1f | 0   |'%(i/10, j/10, k/10), ds_result, ' | ', modds_result)
            relerror[i+j+k-1] = abs((modds_result / ds_result.real) - 1)
            


# In[23]:


errorsum = 0
print('Comparing modified version and original')
print('================================================================================================')
print(' kx  | ky  | qx  | qy  | rel error  ')
print('================================================================================================')
for i in range(0, 75, 5):
    for j in range(0, 75, 5):
        for k in range(1, 4, 1):
            print('%2.1f  | %2.1f | %2.1f | 0   |'%(i/10, j/10, k/10), relerror[i+j+k-1])
            errorsum = errorsum + relerror[i+j+k-1]
           
avgerror = errorsum / (900)            
print('================================================================================================')
print('The average error of modified function is ', avgerror)


# # Ensure that treating the arrays can be implemented with the loops above

# In[8]:


N = 2

sing = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1)
dbl = np.arange(-(N - 1), (N - 1) + 1, 1)

# rng1 = range(-(N - 1) / 2, (N - 1) / 2 + 1, 1)
print('sing:')
print(sing)
print('\nsing as loop:')
k = -(N - 1) / 2    
while(k < ((N - 1) / 2 + 1)):
    print(k)
    k = k + 1
        
print('\ndbl:')
print(dbl)
print('\ndbl as range:')
for i in range(-(N - 1), (N - 1) + 1, 1):
    print(i)


# # Checking the consistency of math and numpy log function

# In[9]:


A = 4  # hbar^2/(2m)=4 evAA^2 (for free electron mass)
eE0 = rati * ((hOmg) ** 2) / (2 * np.sqrt(A * mu))

KT = 1 * 10 ** (-6)
shift = A * (eE0 / hOmg) ** 2

Gamm = 0.005  # Gamma in eV.
hOmg = 0.5  # Photon energy eV

ek = A * (math.sqrt((0.1) ** 2 + (0.1) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2

z = Gamm ** 2 + (ek - hOmg / 2 + hOmg * -0.5) ** 2

mlog = math.log(z)
nplog = np.log(z)

print('math.log result: ', mlog)
print('np.log result: ', nplog)

arrtest0 = [complex(0,mlog)]
arrtest1 = [complex(0,nplog)]

print('math.log result in array: ', arrtest0)
print('np.log result in array: ', arrtest1)

mlog = math.log(Gamm ** 2 + (ek - hOmg / 2 + hOmg * -0.5) ** 2)
nplog = np.log(Gamm ** 2 + (ek - hOmg / 2 + hOmg * -0.5) ** 2)

print('math.log result: ', mlog)
print('np.log result: ', nplog)


# # Testing my Bessel function for various values of v and z
# 
# Ensuring that the number of terms makes the relative error equal to machine epsilon.

# In[33]:


def my_Besselv(v, z):
    # WILL NOT WORK IF v IS NOT AN INTEGER
    # Conditional to handle case of negative v.
    if(v < 0):
        v = abs(v)
        resultsign = (-1) ** v
    else:
        resultsign = 1
    result = 0    
    # Loop to construct Bessel series sum.
    for n in range(0,20):
        sign = (-1)**n
        exp = 2 * n + v
        term = z ** exp
        r = n + v + 1
        if(r == 0):
            r = 1e-15
        denom = math.gamma(r)
        denom = denom * math.factorial(n)
        denom = denom * (2 ** exp)
        term = term / denom * sign
        # print('for ', n, ': ',term)
        result = result + term
        
    return result * resultsign

relerror = np.zeros(78)
l = 0

print('Comparing scipy library Bessel function and my Bessel function:')
print('================================================================================================')
print(' v | z | scipy result |   my result   | rel error')
print('================================================================================================')
for i in range(-3,3):
    for j in range(20,33):
        myres = my_Besselv(i, j/10)
        scires = scipy.special.jv(i, j/10)
        relerror[l] = abs((scires / myres) - 1)
        print('%3d|%3.1f|%14E|%15E|%E' % (i, j/10, scires, myres, relerror[l]))
        l = l + 1
print('================================================================================================')

avgerr = math.fsum(relerror) / l
print('Average relative error: ', avgerr)


# In[ ]:




