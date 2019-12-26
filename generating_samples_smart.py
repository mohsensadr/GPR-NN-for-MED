import numpy as np
from scipy import integrate
import math
import os
from decimal import *
#np.random.seed(0);

def Z(v,l,dimY):
    getcontext().prec = 30
    f = 0.0;
    for i in range(1,dimY+1):
        f += l[i-1]*v**i
    return np.exp(-f);

def Hi(v,l,dimY,i):
    return Z(v,l,dimY)*v**i

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def Mom(l, dimY, dimX):
    q = [0 for x in range(dimX)]
    D, err = integrate.quad(Z, -1e1, 1e1, args=(l, dimY));
    for i in range(0, dimX):
        intt, err = integrate.quad(Hi, -1e1, 1e1, args=(l, dimY, i + 1));
        q[i] = intt / D;
    return q;

'''
def modify_lamb_varr(l,a, dimY):
    L = []
    for i in range(1,dimY+1):
        L.append( l[i-1]*a**i )
    return np.array(L)
'''
def modify_lamb_varr(l,a,dimY,stat):
    L = []
    if stat == "const":
        for i in range(1,dimY):
            L.append( l[i-1]*a**i )
        L.append(l[3]);
    else:
        for i in range(1,dimY+1):
            L.append( l[i-1]*a**i )
    return np.array(L)

def Za(v, l, a, dimY):
    getcontext().prec = 30
    f = 0.0;
    for i in range(dimY):
        f += -(a*v)**(i+1) * l[i]
    return np.exp(f);

def zh2(v,l,a, dimY):
    return Za(v,l, a, dimY)*v**2

def fix_varr(l, dimY, stat):
    varr = 10.0;
    i = 0;
    a = 1.0

    while abs(varr - 1.0) > 1e-13:
        if i == 0:
            a = 1.0
        d2 = integrate.quad(Za, -1e1, 1e1, args=(l, a, dimY));
        m2 = integrate.quad(zh2, -1e1, 1e1, args=(l, a, dimY));
        varr = (m2[0] / d2[0])
        #print("varr= ", varr, " for i= ", i)
        print("in var i=",i, " the varr=",varr)
        a = a*varr ** (1.0 / 2.0)
        i = i + 1
        if i == 10:
            break;
    L = modify_lamb_varr(l,a,dimY, stat)
    return L, varr

#dimY = 6
#dimX = 6

def sample_new_Nl(dimY, stat, val, la_min, la_max):
    dimX = dimY
    Mo = []
    La = []

    #la_max = np.ones(dimY)*0.1
    #la_min = -np.ones(dimY)*0.1
    #if dimY%2 == 0:
    #    la_max[-1] = 1.0
    #    la_min[-1] = 1e-5

    if stat == "const":
        la_max[-1] = val;
        la_min[-1] = val;
    done = 0
    while done == 0:
        while done == 0:
            l0 = (la_max - la_min) * np.random.rand(dimY) + la_min;
            #l0 = np.array( [1.8996972417185123e-01, 7.0282774880488863e-01, -6.8838471212407354e-02] )
            #l0 = np.array([1.899e-01, 7.028e-01, -6.883e-02])

            D0 = integrate.quad(Z, -1e1, 1e1, args=(l0, dimY));
            #if abs(D0[0]) > 1e-15 and np.isnan(D0[0]) == 0 and np.isinf(D0[0]) == 0 and abs(Z(-10.0, l0, dimY)) < 1e-14 and abs(
            #        Z(10.0, l0, dimY)) < 1e-14:
            if abs(D0[0]) > 1e-15 and np.isnan(D0[0]) == 0 and np.isinf(D0[0]) == 0:
                done = 1;
                print("initilization done")
        iii = 0;
        q = Mom(l0, dimY, dimX)
        varr = q[1]
        ll = [0 for x in range(dimY)]
        #while (abs(q[0]) > 1e-13 or abs(varr-1.0)>1e-13) and (np.isnan(q[0]) == 0 and np.isinf(q[0]) == 0):
        while( 1> 0):
            for ii in range(1, dimY + 1):
                ll[ii-1] = 0.0
                for jj in range(ii, dimY + 1):
                    ll[ii-1] += l0[jj-1]*nCr(jj,ii)*q[1]**ii*q[0]**(jj-ii)
                    #ll[ii - 1] += l0[jj - 1] * nCr(jj, ii) * q[0] ** (jj - ii);
            #q = Mom(ll, dimY, dimX)
            q = Mom(ll, dimY, dimX)
            if stat == "const":
                ll[-1] = val;
            ll, varr = fix_varr(ll, dimY, stat)
            #q = Mom(ll, dimY, dimX)
            #if stat == "const":
            #    ll[-1] = val;
            q = Mom(ll, dimY, dimX)
            l0 = ll;
            print("fix mean iterating")
            iii = iii + 1;
            if iii == 10:
                break;
                done = 0;
        if ((abs(q[0]) < 1e-13 and abs(q[1] - 1.0) < 1e-13)  and np.isnan(q[0]) == 0 and np.isinf(q[0]) == 0):
            Mo.append(q);
            La.append(ll);
        else:
            done = 0
    return La, Mo, ll;

#D0, err = integrate.quad(Z, -1e1, 1e1, args=(l0, dimY));
#if np.isnan(D0) == 0 and np.isinf(D0) == 0:
#    done = 1;
#    print("initilization done, D0[0]= ",D0)

#q = Mom(l0, dimY, dimX)
#l = [0 for x in range(dimY)]
'''
print(q)
while(abs(q[0]) + abs(q[1]-1)  > 1e-14):
    ## fix mean
    while( abs(q[0])>1e-10 ):
        for i in range(1,dimY+1):
            l[i-1] = 0.0
            for j in range(i,dimY+1):
                #l[i-1] += l0[j-1]*nCr(j,i)*q[1]**i*q[0]**(j-i)
                l[i - 1] += l0[j - 1] * nCr(j, i)  * q[0] ** (j - i)
        q = Mom(l, dimY, dimX)
        print(q)
        l0 = l;
        print("fix mean iterating")
    ## fix variance
    varr = q[1]
    while (abs(varr-1) > 1e-10):
        l, varr = fix_varr(l,dimY)
        print("fix var iterating")
    q = Mom(l, dimY, dimX);
    print("abs(q[0]) + abs(q[1]-1) = ", str(abs(q[0]) + abs(q[1]-1)))
print("l0 = "+str(l0))
'''
