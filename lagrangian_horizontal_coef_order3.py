import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
import scipy
import matplotlib.pyplot as plt
import os
from linearwavetheory.settings import stokes_theory_options
import sympy as sp
import validation.terms as terms

one = sp.sympify(1)
zero = sp.sympify(0)

def x31(kd,height):
    import validation.third_order_coef as to
    steepness = 1
    frequency = 0.1

    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = kd / wavenumber

    x11 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_first_harmonic(
        1,wavenumber,depth,height/wavenumber,order=1
    )
    x31 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_first_harmonic(
        1,wavenumber,depth,height/wavenumber,order=3
    ) - x11
    x31 = to.x31(kd[:,None],height)
    return x31

def x33(kd,height):
    import validation.third_order_coef as to
    steepness = 1
    frequency = 0.1

    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency ** 2 / _GRAV / mu
    depth = kd / wavenumber

    x33 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_third_harmonic(
        1, wavenumber, depth, height / wavenumber, order=3
    )
    x33 = to.x33(kd[:, None], height)
    return x33

def shared_3(kd,height=0,analytical=False):
    steepness = 1
    frequency = 0.1

    if analytical:
        mu = sp.symbols('mu')
    else:
        mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / np.tanh(kd)
    depth = kd / wavenumber


    c = angular_frequency / wavenumber

    dim_height = height / wavenumber


    ch1 = np.cosh(kd + wavenumber *dim_height) / np.cosh(kd)
    ch2 = np.cosh(2 * kd + 2 * wavenumber * dim_height) / np.cosh(2 * kd)
    ch3 = np.cosh(3 * kd + 3 * wavenumber * dim_height) / np.cosh(3 * kd)
    ch4 = np.cosh(4 * kd + 4 * wavenumber * dim_height) / np.cosh(4 * kd)
    sh1 = np.sinh(kd + wavenumber *dim_height) / np.cosh(kd)
    sh2 = np.sinh(2 * kd + 2 * wavenumber * dim_height) / np.cosh(2 * kd)
    sh3 = np.sinh(3 * kd + 3 * wavenumber * dim_height) / np.cosh(3 * kd)

    disp = (9 - 10 * mu ** 2 + 9 * mu ** 4) / 16 / mu ** 4
    us = ( 1 + mu**2)/mu**2/2 * ch2

    sh1 = np.sinh(kd+height ) / np.cosh(kd)
    ch1 = np.cosh(kd+height ) / np.cosh(kd)
    ch2 = np.cosh(2 * (kd+height) ) / np.cosh(2*kd)
    sh2 = np.sinh(2 * (kd+height) ) / np.cosh(2*kd)
    sh3 = np.sinh(3 * (kd+height) ) / np.cosh(3*kd)
    ch3 = np.cosh(3 * (kd+height) ) / np.cosh(3*kd)
    ch4 = np.cosh(4 * (kd+height) ) / np.cosh(4*kd)
    sh4 = np.sinh(4 * (kd+height) ) / np.cosh(4*kd)

    a11 = 1 / mu * sh1
    a22 =  (1 + mu ** 2) * (3 - 3 * mu ** 2) / 8 / mu ** 4 * sh2
    a20 = (1+mu**2)/4/mu**2 * sh2
    x11 = - 1 / mu * ch1
    x22 = (1-mu**2)/4/mu**2 - 3 * ( 1  - mu**4) / mu**4/ 8 * ch2
    u11 = 1 / mu * ch1
    u22 = ch2* 3 * ( 1  - mu**4)/ mu**4 /4

    v11 = 1 / mu * sh1
    v22 = sh2* 3 * ( 1  - mu**4)/ mu**4 /4

    u33 = [0, (27 + 15 * mu ** 2 + 117 * mu ** 6 - 159 * mu ** 4) / 64 / mu ** 7]

    Ia = ( 1+3*mu**2)/(1+mu**2)
    Ib = (1-mu**2)/(1+mu**2)

    IIa = ( 1+3*mu**2)
    IIb = (1-mu**2)

    _a11v22 = 3 * ( 1  - mu**4)/ mu**5 /4
    #a11v22 = 2*_a11v22 * sh1 * sh2
    a11v22 = [- _a11v22 *Ib , _a11v22 *Ia ]

    _a20v11 = (1+mu**2)/4/mu**2 * 1 / mu
    a20v11 = [ -_a20v11 *  Ib/2, _a20v11 * Ia/2]

    _a22v11 =  (1 + mu ** 2) * (3 - 3 * mu ** 2) / 8 / mu ** 5
    a22v11 =  [-_a22v11*Ib/2, _a22v11*Ia/2]

    _x11u22 = 3 * ( 1  - mu**4)/ mu**5 /2
    x11u22 = [ _x11u22 *Ib/2, _x11u22 *Ia/2]

    _x22u11a =  -(1-mu**2)/4/mu**3
    _x22u11b = + 3 * (1 - mu ** 4) / mu ** 5 / 8
    x22u11 = [ _x22u11a + _x22u11b*Ib/2, _x22u11b *Ia/2]

    _a11p2u11 = 1 / mu ** 3 / 2
    #a11p2u11 = _a11p2u11 * sh1**2 *ch1
    a11p2u11 = [-_a11p2u11 * IIb/4, _a11p2u11 * IIa/4]

    _x11p2u11 = - 1 / mu ** 3 /2
    x11p2u11 = [+3*_x11p2u11 * IIb/4, _x11p2u11 * IIa/4] #_x11p2u11 * ch1**2 *ch1

    _a11x11v11 = 1 / mu ** 3
    a11x11v11 = [-_a11x11v11 * IIb/4, _a11x11v11 * IIa/4] #_a11x11v11 * ch1 *sh1**2

    _dispx11 = -(9 - 10 * mu ** 2 + 9 * mu ** 4) / 16 / mu ** 5
    dispx11 = [_dispx11,0]

    _usx11 = ( 1 + mu**2)/mu**3/2 #* ch2
    usx11 = [ _usx11 *Ib/2, _usx11 *Ia/2] #_usx11*ch1 * ch2

    _terms = [
        u33,
        a11v22,
        a20v11,
        a22v11,
        x11u22,
        x22u11,
        a11p2u11, #a11**2*u11/2,
        x11p2u11,#-x11**2 * u11/2,
        a11x11v11, #-a11 *x11 * v11,
        dispx11,
        usx11#-us* x11 ,
    ]
    if analytical:
        return _terms

    terms = []
    for term in _terms:
        if isinstance(term,list):
            terms.append(
                term[0]*ch1 + term[1]*ch3
            )
        else:
            terms.append(term)
    return terms


def nx31(kd,height):
    L31 =[
        0,
        1/2,
        1,
        1/2,
        1/2,
        1/2,
        3/4,
        1/4,
        1/4,
        1,
        1,
    ]

    terms = shared_3(kd,height)

    _result = 0
    for fac,term in zip(L31,terms):
        _result += fac * term
    return -_result

def nx33(kd,height):
    L33 = [
        1,
        1 / 2,
        0,
        1 / 2,
        -1 / 2,
        -1 / 2,
        1 / 4,
        -1 / 4,
        -1 / 4,
        0,
        0,
    ]


    terms = shared_3(kd,height)

    _result = 0
    for fac,term in zip(L33,terms):
        _result += fac * term


    return -_result/3

def sym_x31():
    L31 = [
        zero,
        one / 2,
        one,
        one / 2,
        one / 2,
        one / 2,
        sp.sympify(3) / 4,
        one / 4,
        one / 4,
        one,
        one,
    ]

    ch1 = combine(L31, 0)
    ch3 = combine(L31,1)
    print(ch1)
    print(ch3)

def combine(  coef, index ):
    kd = np.ones(2)
    _terms = shared_3(kd,0,True)
    _result = zero
    for _T,_coef in zip(_terms,coef):
        _result += _T[index] * _coef

    _result = sp.together(_result)

    expr = sp.expand(_result.args[-1]) * sp.prod(_result.args[:-1])

    return expr


def sym_x33():

    mu = sp.symbols('mu')
    L33 = [
        one,
        one / 2,
        zero,
        one / 2,
        -one / 2,
        -one / 2,
        one / 4,
        -one / 4,
        -one / 4,
        zero,
        zero,
    ]
    ch3 = combine(L33,1)
    ch1 = combine(L33, 0)


def final_x33(dim_depth,dim_height):
    mu = np.tanh(dim_depth)
    b= (117*mu**6 - 159*mu**4 + 15*mu**2 + 27)/(64*mu**7)
    a= (-38*mu**4 + 68*mu**2 - 30)/(32*mu**5)
    ch1 = np.cosh(dim_depth + dim_height) / np.cosh(dim_depth)
    ch3 = np.cosh(3*dim_depth + 3*dim_height) / np.cosh(3*dim_depth)
    return - (a * ch1 + b*ch3)/3

def final_x31(dim_depth,dim_height):
    mu = np.tanh(dim_depth)
    a=(-10*mu**4 + 12*mu**2 - 18) / (32 * mu ** 5 )
    b=(-42*mu**4 + 76*mu**2 + 30) / (32 * mu ** 5  )
    ch1 = np.cosh(dim_depth + dim_height) / np.cosh(dim_depth)
    ch3 = np.cosh(3*dim_depth + 3*dim_height) / np.cosh(3*dim_depth)
    return - (a * ch1 + b*ch3)



def validate(target,new,name):
    kd = np.linspace(1,5,100)
    height = [0,-0.5]

    _ok = True
    for z in height:
        ref = target(kd,z)
        res = new(kd,z)
        diff = np.max(np.abs(ref-res)/np.abs(ref))
        if diff > 1e-12:
            _ok = False
            print( f"- {name} Height: {z} -> Max diff: {diff}")

        plt.figure()

        mu = np.tanh(kd)
        eps = mu**3 * 0.3
        scale = 1

        plt.plot(mu,ref*scale,label='Ref')
        plt.plot(mu,res*scale,label='Res')
    plt.legend()
    plt.show()
    return _ok


def coefs():
    L31 = [
        0,
        1 / 2,
        1,
        1 / 2,
        1 / 2,
        1 / 2,
        3 / 4,
        1 / 4,
        1 / 4,
        1,
    ]

    L33 = [
        1,
        1 / 2,
        0,
        1 / 2,
        -1 / 2,
        -1 / 2,
        1 / 4,
        -1 / 4,
        -1 / 4,
        0,
    ]
    L31 = np.array(L31)
    L33 = np.array(L33)
    print(L31+L33)

if __name__ == "__main__":

    a = terms.validate(final_x31,nx31,'x31',False)
    a = terms.validate(final_x33, nx33, 'x33',False)

    sym_x31()
    #sym_x31()
    #coefs()
    #coef()
    # #print( a,b,c)



# 0.1
# 3 1.5000000000018257
# 8 0.5000000000008891
# 14 0.5000000000001872
# 17 0.5000000000001872
# 19 -0.5000000000001872
# 20 0.5000000000001872
# 21 0.16666666666666666
# 22 -0.16666666666679145
# 23 1.000000000001279