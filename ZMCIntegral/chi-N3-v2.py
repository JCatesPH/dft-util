
# coding: utf-8

# # Test to ensure modifications to function do not change evaluated value
# 
# Beginning modification of the function to handle case where N is not equal to 1.


import math
# from numba import cuda
import ZMCIntegral
import time
import numpy as np
import helpers
import numba
# # Define constants in function

mu = 0.1  # Fermi-level
hOmg = 0.5  # Photon energy eV
a = 4  # AA
A = 4  # hbar^2/(2m)=4 evAA^2 (for free electron mass)
rati = 0.01  # ratio
eE0 = rati * ((hOmg) ** 2) / (2 * np.sqrt(A * mu))
# print(eE0)
Gamm = 0.003  # Gamma in eV.
KT = 1 * 10 ** (-6)
shift = A * (eE0 / hOmg) ** 2


@numba.cuda.jit(device=True)        
def modDsN2(x):
    N = 3
    dds = 0
    # ds = 0 # UNUSED
    qx = helpers.getqx()
    ek = A * (math.sqrt((x[0]) ** 2 + (x[1]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((x[0]) ** 2 + (x[1]) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2) / hOmg ** 2

    taninv1kp = numba.cuda.shared.array(N,dtype=numba.types.float64)
    taninv1kqp = numba.cuda.shared.array(N,dtype=numba.types.float64)
    taninv1km = numba.cuda.shared.array(N,dtype=numba.types.float64)
    taninv1kqm = numba.cuda.shared.array(N,dtype=numba.types.float64)

    lg1kp = numba.cuda.shared.array(N,dtype=numba.types.complex128)
    lg1kqp = numba.cuda.shared.array(N,dtype=numba.types.complex128)
    lg1km = numba.cuda.shared.array(N,dtype=numba.types.complex128)
    lg1kqm = numba.cuda.shared.array(N,dtype=numba.types.complex128)

    ferp = numba.cuda.shared.array(N,dtype=numba.types.float64)
    ferm = numba.cuda.shared.array(N,dtype=numba.types.float64)
  
    j = 0
    i = -(N - 1) / 2
    while(i < ((N - 1) / 2 + 1)):
        nu = hOmg * i
        chi = hOmg / 2
        omicron = ek - chi + nu
        phi = ekq - chi + nu
        iota = ek + chi + nu
        kappa = ekq + chi + nu

        taninv1kp[j] = 2 * math.atan2(Gamm, omicron)
        taninv1kqp[j] = 2 * math.atan2(Gamm, phi)
        taninv1km[j] = 2 * math.atan2(Gamm, iota)
        taninv1kqm[j] = 2 * math.atan2(Gamm, kappa)

        lg1kp[j] = complex(0, 1) * math.log(Gamm ** 2 + (omicron) ** 2)
        lg1kqp[j] = complex(0, 1) * math.log(Gamm ** 2 + (phi) ** 2)
        lg1km[j] = complex(0, 1) * math.log(Gamm ** 2 + (iota) ** 2)
        lg1kqm[j] = complex(0, 1) * math.log(Gamm ** 2 + (kappa) ** 2)

        ferp[j] = helpers.my_heaviside(mu - chi - nu)
        ferm[j] = helpers.my_heaviside(mu + chi - nu)
        i = i + 1
        j = j + 1

    size_dbl = 2 * N - 1
    taninv2k = numba.cuda.shared.array(size_dbl,dtype=numba.types.float64)
    taninv2kq = numba.cuda.shared.array(size_dbl,dtype=numba.types.float64)

    lg2k = numba.cuda.shared.array(size_dbl,dtype=numba.types.complex128)
    lg2kq = numba.cuda.shared.array(size_dbl,dtype=numba.types.complex128)
    
    besk = numba.cuda.shared.array(size_dbl,dtype=numba.types.float64)
    beskq = numba.cuda.shared.array(size_dbl,dtype=numba.types.float64)

    fac1 = numba.cuda.shared.array(size_dbl,dtype=numba.types.complex128)
    fac2 = numba.cuda.shared.array(size_dbl,dtype=numba.types.complex128)
    fac3 = numba.cuda.shared.array(size_dbl,dtype=numba.types.complex128)
    
    j = 0
    for i in range(-(N - 1), N, 1):
        xi = hOmg * i
        zeta = ek - mu + xi
        eta = ekq - mu + xi

        taninv2k[j] = 2 * math.atan2(Gamm, zeta)
        taninv2kq[j] = 2 * math.atan2(Gamm, eta)

        lg2k[j] = complex(0, 1) * math.log(Gamm ** 2 + (zeta) ** 2)
        lg2kq[j] = complex(0, 1) * math.log(Gamm ** 2 + (eta) ** 2)

        besk[j] = cudabesselj.besselj(i, xk)
        beskq[j] = cudabesselj.besselj(i, xkq)

        fac1[j] = ek - ekq + xi
        fac2[j] = fac1[j] + 2 * complex(0, 1) * Gamm
        fac3[j] = fac2[j] - ek + ekq
        j = j + 1


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

                            d1 = -2 * complex(0, 1) * fac1[beta - gamma + N - 1] * fac2[alpha - gamma + N - 1] * \
                                 fac3[
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

                            d2 = -2 * complex(0, 1) * fac1[alpha - beta + N - 1] * fac2[alpha - gamma + N - 1] * \
                                 fac3[
                                     beta - gamma + N - 1]

                            omint2p = ferp[s] * ((pp1p + pp2p + pp3p) / d2)

                            omint2m = ferm[s] * ((pp1m + pp2m + pp3m) / d2)

                            bess2 = beskq[gamma - n + N - 1] * beskq[gamma - s + N - 1] * beskq[beta - s + N - 1] * \
                                    beskq[beta - l + N - 1] * besk[alpha - l + N - 1] * besk[alpha - n + N - 1]

                            glga = bess2 * (omint2p - omint2m)

                            dds = dds + 2 * Gamm * (grgl + glga)
    return dds.real



kxi = - math.pi / a
kxf = math.pi / a

kyi = - math.pi / a
kyf = math.pi / a

helpers.setqx(0.01)

MC = ZMCIntegral.MCintegral(modDsN2,[[kxi,kxf],[kyi,kyf]])

# Setting the zmcintegral parameters
MC.depth = 3
MC.sigma_multiplication = 100
MC.num_trials = 10

start = time.time()

result = MC.evaluate()
print('Result for qx = 0.01 :', result[0], ' with error: ', result[1])
print('================================================================')

end = time.time()
print('Computed in ', end-start, ' seconds.')
print('================================================================')