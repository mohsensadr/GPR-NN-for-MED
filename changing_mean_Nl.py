from scipy import integrate
import numpy as np
import sympy as sym
import pickle
import dill
from sympy import poly
from sympy import simplify

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def zh3(v,l):
    return Z(v,l, 0.0)*v**3
def zh4(v,l):
    return Z(v,l, 0.0)*v**4
def zh5(v,l):
    return Z(v,l, 0.0)*v**5
def zh6(v,l):
    return Z(v,l, 0.0)*v**6


def Z(v, l, mu):
    dummy = 0.0;
    for i in range(len(l)):
        dummy += (v-mu)**(i+1) * l[i]
    return np.exp(-dummy)

def Za(v, l, a):
    dummy = 0.0;
    for i in range(len(l)):
        dummy += (a*v) ** (i + 1) * l[i]
    return np.exp(-dummy)

def zh1(v,l,mu):
    return Z(v,l, mu)*v

def zh2(v,l,a):
    return Za(v,l, a)*v**2

def sym_new_lamb():
    v = sym.Symbol('v')
    mu = sym.Symbol('mu')
    l1, l2, l3, l4, l5, l6= sym.symbols('l1 l2 l3 l4 l5 l6')
    z = -(v-mu) * l1 - (v-mu)**2 * l2 - (v-mu)**3 * l3 - (v-mu)**4 * l4 - (v-mu)**5 * l5 - (v-mu)**6 * l6
    z = poly(z,v)
    return z.rep.rep

def modify_lamb_varr(l,a):
    L = []
    for i in range(1,len(l)+1):
        L.append( l[i-1]*a**i )
    return np.array(L)

def modify_lamb(l,mu):
    '''-l6, 6*mu*l6 - l5, -15*mu**2*l6 + 5*mu*l5 - l4, 20*mu**3*l6 - 10*mu**2*l5 + 4*mu*l4 - l3,
     -15*mu**4*l6 + 10*mu**3*l5 - 6*mu**2*l4 + 3*mu*l3 - l2, 6*mu**5*l6 - 5*mu**4*l5 + 4*mu**3*l4
      - 3*mu**2*l3 + 2*mu*l2 - l1, -mu**6*l6 + mu**5*l5 - mu**4*l4 + mu**3*l3 - mu**2*l2 + mu*l1]'''
    #l1 = l[0]; l2 = l[1]; l3 = l[2]; l4 = l[3]; l5 = 0; l6 = 0;
    #L = []
    #
    #L.append( 6*mu**5*l6 - 5*mu**4*l5 + 4*mu**3*l4 - 3*mu**2*l3 + 2*mu*l2 - l1);
    #L.append( -15*mu**4*l6 + 10*mu**3*l5 - 6*mu**2*l4 + 3*mu*l3 - l2)
    #L.append(20*mu**3*l6 - 10*mu**2*l5 + 4*mu*l4 - l3)
    #L.append(-15*mu**2*l6 + 5*mu*l5 - l4)

    l1 = l[0];
    l2 = l[1];
    l3 = l[2];
    l4 = l[3];
    if(len(l)==4):
        l5 = 0.0; l6=0.0; l7=0.0;l8=0.0;
    elif(len(l) == 6):
        l5 = l[4];
        l6 = l[5];
        l7 = 0.0;
        l8 = 0.0;
    elif(len(l)==8):
        l5 = l[4];
        l6 = l[5];
        l7 = l[6];
        l8 = l[7];
    L = []
    #
    L.append(
        8 * mu ** 7 * l8 - 7 * mu ** 6 * l7 + 6 * mu ** 5 * l6 - 5 * mu ** 4 * l5 + 4 * mu ** 3 * l4 - 3 * mu ** 2 * l3 + 2 * mu * l2 - l1);
    L.append(
        -28 * mu ** 6 * l8 + 21 * mu ** 5 * l7 - 15 * mu ** 4 * l6 + 10 * mu ** 3 * l5 - 6 * mu ** 2 * l4 + 3 * mu * l3 - l2)
    L.append(56 * mu ** 5 * l8 - 35 * mu ** 4 * l7 + 20 * mu ** 3 * l6 - 10 * mu ** 2 * l5 + 4 * mu * l4 - l3)
    L.append(-70 * mu ** 4 * l8 + 35 * mu ** 3 * l7 - 15 * mu ** 2 * l6 + 5 * mu * l5 - l4)
    if (len(l) > 4):
        L.append(56 * mu ** 3 * l8 - 21 * mu ** 2 * l7 + 6 * mu * l6 - l5)
        L.append(-28 * mu ** 2 * l8 + 7 * mu * l7 - l6)
    if(len(l)>6):
        L.append(8 * mu * l8 - l7)
        L.append(-l8)
    return -np.array(L);


def fix_mean(l):
    mm = 10.0;
    i = 0;
    while abs(mm) > 1e-10:
        if i == 0:
            mm = 0.0
            m0 = 0.0
        den = integrate.quad(Z, -1e1, 1e1, args=(l, -m0));
        m1 = integrate.quad(zh1, -1e1, 1e1, args=(l, -m0));
        mm = m1[0] / den[0]
        m0 += mm

        L = modify_lamb(l, -m0);
        den2 = integrate.quad(Z, -1e1, 1e1, args=(L, 0.0));
        m12 = integrate.quad(zh1, -1e1, 1e1, args=(L, 0.0));
        mm = m12[0] / den2[0]

        print("mm= ", mm, " for i= ", i, m12[0])
        i = i + 1
        if i==100:
            return L, mm
    return L, mm

def fix_varr(l):
    varr = 10.0;
    i = 0;
    a = 1.0

    while abs(varr - 1.0) > 1e-13:
        if i == 0:
            a = 1.0
        d2 = integrate.quad(Za, -1e1, 1e1, args=(l, a));
        m2 = integrate.quad(zh2, -1e1, 1e1, args=(l, a));
        varr = (m2[0] / d2[0])
        #print("varr= ", varr, " for i= ", i)
        print("in var i=",i, " the varr=",varr)
        if i == 100:
            break;
        a = a*varr ** (1.0 / 2.0)
        i = i + 1
    L = modify_lamb_varr(l,a)
    return L, varr


#v = sym.Symbol('v')
#mu = sym.Symbol('mu')
#l1, l2, l3, l4, l5, l6= sym.symbols('l1 l2 l3 l4 l5 l6')

#z = sym_new_lamb();

#l0 = [0.1,  -9.72413568e-01,  2.44762273e-04,  1.93579381e-02, -1.18165578e-04,  4.53634721e-06];
#l0 = [0.0, -1.0e-1, .0, -1.0e-1, 0.0, 1e-2]
#l = [ -1.00000000e-3, 6.89733001e-01,  1.00000000e-4,  1.48769680e-03,  1.00000000e-16,  0.0, 0.0];
#l = [ -1.00000000e-1, 6.89733001e-01,  0.0, 0.0,  0.0,  0.0, 0.0]

def zhi(v,l,i):
    return Z(v,l,0.0)*v**i

def sample_new(dimY):
    #print("\n \n Sample_new()  \n\n")
    Mo = []
    La = []
    #la_max = np.array([1e1, 1e1, 1e1, val])
    #la_min = np.array([-1e1, -1e1, -1e1, val])

    if dimY==4:
        la_max = np.array([ 1.0,    1e0,   1e0, 1e1])
        la_min = np.array([ -1.0,  -1e0,  -1e0, -1e-1])
    elif dimY==6:
        la_max = np.array([ 1e1,   1e1,   1e1,  1e1, 1e1, 1e-4])
        la_min = np.array([-1e1,  -1e1,  -1e1, -1e1,-1e1, 1e-4])
    elif dimY==8:
        la_max = np.array([ 1e1,   1e1,   1e1,  1e1, 1e-2,  1e-2, 1e-3, 1e-7])
        la_min = np.array([-1e1,  -1e1,  -1e1, -1e1,-1e-2, -1e-2, 1e-3, 1e-7])
    done = 0
    while done==0:
        while done == 0:
            l0 = (la_max - la_min) * np.random.rand(dimY) + la_min;
            D0 = integrate.quad(Z, -1e1, 1e1, args=(l0, 0.0));
            if np.isnan(D0[0]) == 0 and np.isinf(D0[0]) == 0 and abs(Z(-10.0,l0,0.0))<1e-14 and abs(Z(10.0,l0,0.0))<1e-14:
                done = 1;
                print("initilization done")


        l=np.array(l0).copy()
        mm = 10.0; varr = 10.0;
        i = 0;
        while (abs(mm)>1e-14 or abs(varr-1.0)>1e-14) and (np.isnan(mm) == 0 and np.isinf(mm) == 0):
            l, mm = fix_mean(l)
            l , varr = fix_varr(l)

            d = integrate.quad(Z, -1e1, 1e1, args=(l, 0.0));
            m = integrate.quad(zh1, -1e1, 1e1, args=(l, 0.0));
            mm = m[0] / d[0]
            print("mm= ",mm," and abs(var - 1.0) =",abs(varr-1.0))
            if i==1000:
                break;
                done = 0;
            i = i+1;

        if ((abs(mm) < 1e-14 and abs(varr - 1.0) < 1e-14)  and np.isnan(mm) == 0 and np.isinf(mm) == 0):
            qall = [];
            qall.append(mm)
            qall.append(varr)
            for kk in range(3,21):
                qdum, dvar = np.array(integrate.quad(zhi, -1e1, 1e1, args=(l, kk))) / d[0];
                qall.append(qdum);
            #Q = [q1, q2, q3[0], q4[0], q5[0], q6[0], q7[0], q8[0], q9[0], q10[0], q11[0], q12[0], q13[0], q6[0], q6[0]];
            Mo.append(qall);
            La.append(l);
        else:
            done = 0

    return La, Mo, l0;

    #fig, ax = plt.subplots();
#    vv = np.linspace(-10.0, 10.0, num=1000)
    #plt.plot(vv,Z(vv,l0,0.0)/D0[0],'-',label='old l')
    #plt.plot(vv,Z(vv,l,0.0)/D[0],'-',label='L cor.')
    #ax.set_ylabel("Z")
    #plt.legend()
    #plt.show()

