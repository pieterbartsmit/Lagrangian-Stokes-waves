import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
import scipy
import matplotlib.pyplot as plt
import os
from linearwavetheory.settings import stokes_theory_options
import sympy as sp
import validation.terms as trms
import validation.fourth_order_coef as fo

one = sp.sympify(1)
zero = sp.sympify(0)

def an_stokes_drift(eps, kd, height=0):
    mu = np.tanh(kd)
    frequency = 0.1
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = kd / wavenumber

    c = angular_frequency / wavenumber

    dim_height = height / wavenumber
    an2 = stokes.stokes_drift(eps, wavenumber, depth, dim_height) / c
    #an4 = an_stokes_4(eps, kd, height)
    an4 = x40(kd,height)
    return an4*eps**4 + an2

def an_stokes_4(eps, kd, height=0):

    steepness = 1
    frequency = 0.1

    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = kd / wavenumber


    c = angular_frequency / wavenumber

    dim_height = height / wavenumber


    a11 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic(
        steepness,wavenumber,depth,dim_height,order=1
    )
    a31 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic(
        steepness,wavenumber,depth,dim_height,order=3
    ) - a11
    a22 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_second_harmonic(
        steepness,wavenumber,depth,dim_height,order=2
    )
    a20 = stokes.dimensionless_lagrangian_setup(steepness, wavenumber * depth, wavenumber * dim_height,order=2)
    a33 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_third_harmonic(
        steepness,wavenumber,depth,dim_height,order=3
    )
    x11 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_first_harmonic(
        steepness,wavenumber,depth,dim_height,order=1
    )
    x31 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_first_harmonic(
        steepness,wavenumber,depth,dim_height,order=3
    ) - x11



    x22 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_second_harmonic(
        steepness,wavenumber,depth,dim_height,order=2
    )
    x33 =  stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_third_harmonic(
        steepness,wavenumber,depth,dim_height,order=3
    )
    #x33 = nx33(wavenumber * depth, 0)

    ch1 = np.cosh(kd + dim_height) / np.cosh(kd)
    ch2 = np.cosh(2 * kd + 2 * wavenumber * dim_height) / np.cosh(2 * kd)
    ch3 = np.cosh(3 * kd + 3 * wavenumber * dim_height) / np.cosh(3 * kd)
    sh1 = np.sinh(kd + dim_height) / np.cosh(kd)
    sh2 = np.sinh(2 * kd + 2 * wavenumber * dim_height) / np.cosh(2 * kd)
    sh3 = np.sinh(3 * kd + 3 * wavenumber * dim_height) / np.cosh(3 * kd)

    u11 = ch1*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_first_harmonic(steepness,wavenumber*depth,order=1)
    u22 = ch2*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_second_harmonic(steepness,wavenumber*depth,order=2)
    u33 = ch3*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_third_harmonic(steepness,wavenumber*depth,order=3)

    v11 = sh1*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_first_harmonic(steepness,wavenumber*depth,order=1)
    v22 = sh2*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_second_harmonic(steepness,wavenumber*depth,order=2)
    v33 = sh3*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_third_harmonic(steepness,wavenumber*depth,order=3)

    s1s3 =0 # done
    s2s2 = 0.5 # done
    s1s1 = 0.5 # done
    c1c3 = 0. # done
    c2c2 = 0.5 # done
    c2 = 0. # done
    c1c1 = 0.5 # done
    s1s1c2 = -1/4 #done
    s1s2c1 = 1/4 #done
    c1c1c2 = 1/4
    c1c1s1s1 = 1/8
    c1c1c1c1 = 3/8
    s1s1s1s1 = 3/8

    _stokes = (
        -3 * x11 * u33 * s1s3
        -2 * x22*u22 * s2s2
        -x31*u11*s1s1
        -x33*u11 * s1s3
        + 3 * a11*v33 * c1c3
        + 2 * a22*v22 * c2c2
        + 2 * a20*v22 * c2
        + a31 * v11 * c1c1
        + a33 * v11 * c1c3
        - 2 * x11**2 * u22 * s1s1c2
        -x11 * x22 * u11 * s1s2c1
        +2 * a11**2 * u22 *c1c1c2
        + a11 * a22 * u11 * c1c1c2
        +a11*a20*u11*c1c1 #10b
        - 4* x11*a11*v22 *s1s2c1 #11
        - x11*a22*v11 * s1s1c2 #12a
        - x11 * a20 * v11 * s1s1 #12b
        - x22 *a11*v11 * s1s2c1 #13
        - a11*x11**2 *v11 /2 * c1c1s1s1 #14
        - a11**2*x11 *u11/2 * c1c1s1s1 #15
        + a11**3 * v11 * c1c1c1c1/6 #16
        + x11**3 * u11 * s1s1s1s1/6 #17
    )
    return _stokes #+ an2


def shared_4(kd,height=0,analytical=False):
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


    ch1 = np.cosh(kd + wavenumber*dim_height) / np.cosh(kd)
    ch2 = np.cosh(2 * kd + 2 * wavenumber * dim_height) / np.cosh(2 * kd)
    ch3 = np.cosh(3 * kd + 3 * wavenumber * dim_height) / np.cosh(3 * kd)
    ch4 = np.cosh(4 * kd + 4 * wavenumber * dim_height) / np.cosh(4 * kd)
    sh1 = np.sinh(kd + wavenumber*dim_height) / np.cosh(kd)
    sh2 = np.sinh(2 * kd + 2 * wavenumber * dim_height) / np.cosh(2 * kd)
    sh3 = np.sinh(3 * kd + 3 * wavenumber * dim_height) / np.cosh(3 * kd)

    disp = (9 - 10 * mu ** 2 + 9 * mu ** 4) / 16 / mu ** 4
    us = ( 1 + mu**2)/mu**2/2 * ch2

    _x22a = (1-mu**2)/4/mu**2
    _x22b = - 3 * ( 1  - mu**4) / mu**4/ 8
    x22 =  (1-mu**2)/4/mu**2 - 3 * ( 1  - mu**4) / mu**4/ 8 * ch2



    _a11 = 1 / mu
    a11 = _a11 * sh1

    # A31
    a31a = (
            +(-21 + 19 * mu ** 2 - 14 * mu ** 4) / 32 / mu ** 5
    ) +(1 - mu ** 2) * (3 - 4 * mu ** 2) / mu ** 5 / 32

    a31b = (
            (9 + 23 * mu ** 2 - 12 * mu ** 4) / 32 / mu ** 5
    ) +1 / mu ** 5 / 32 * (21 * mu ** 2 - 18 * mu ** 4 + 9)
    a31 = a31a * sh1 + a31b * sh3


    _a22 = - (1+mu**2)/4/mu**2  +(1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 4
    a22 = _a22 * sh2
    _a20 = (1+mu**2)/4/mu**2
    a20 = _a20 * sh2


    a33a = (1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 5-(1 - mu ** 2) * (3 - 4 * mu ** 2) / mu ** 5 / 32
    a33b = (27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 7 / 192-1 / mu ** 5 / 32 * (21 * mu ** 2 - 18 * mu ** 4 + 9)
    a33 = a33a*sh1 + a33b * sh3

    _x11 = -1/mu
    x11 = _x11 * ch1
    x31a = (10 * mu ** 4 - 12 * mu ** 2 + 18) / (32 * mu ** 5)
    x31b = (42 * mu ** 4 - 76 * mu ** 2 - 30) / (32 * mu ** 5)
    x31 =  x31a * ch1 + x31b * ch3

    x33b = (-39 * mu ** 6 + 53 * mu ** 4 - 5 * mu ** 2 - 9) / (64 * mu ** 7)
    x33a = (38 * mu ** 4 - 68 * mu ** 2 + 30) / (96 * mu ** 5)
    x33 = x33a * ch1 + x33b * ch3

    _u11 = 1 / mu
    u11 = ch1*_u11
    _u22 = 3 * ( 1  -mu**4)/mu**4/4
    u22 = ch2*_u22

    _u33 = 3 * ( 9  + 5 * mu**2 + 39 * mu**6 - 53 * mu**4)/64 / mu**7
    u33 = ch3*_u33

    u44 = [0,0,4 * (1 - mu ** 2) * (
                (
                405 + 1161 * mu ** 2 - 6462 * mu ** 4 + 3410 * mu ** 6 + 1929 * mu ** 8 + 197 * mu ** 10
                )
                / (1536 * mu ** 10 * (5 + mu ** 2))
        )]
    u42 = [0,2 * (
                ( -81 -135 * mu ** 2 +810*mu**4 +182* mu**6 - 537*mu**8 + 145*mu**10)
              / (768 * mu**10)
        ),0]

    v11 = sh1*_u11
    v22 = sh2*_u22
    v33 = sh3*_u33

    Ia = (1 + 6*mu**2 + mu**4 ) / (1+3*mu**2)
    Ib = (1 - mu**4) / (1+3*mu**2)
    I0 = (1 - mu**2)**2 / (1+3*mu**2)
    IIa = (1 + 6*mu**2 + mu**4 )
    IIb = (1 - mu**4)
    IIc = (1-mu**2)**2
    IIIa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)**2
    IIIb = (1 - mu**2)/(1+mu**2)
    Oa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)
    Ob = (1 - mu**2)
    Oc = (1-mu**2)**2/(1+mu**2)


    _T0a = 2*x22*(disp-us)
    _us = (1 + mu ** 2) / mu ** 2 / 2
    _T0b =  (1-mu**2)/4/mu**2 - 3 * ( 1  - mu**4) / mu**4/ 8 * ch2
    _x22a = (1-mu**2)/4/mu**2
    _x22b = - 3 * ( 1  - mu**4) / mu**4/ 8
    _T0 = 2*(
        _x22a * disp - _x22a*_us *ch2 + _x22b*disp*ch2 - _us * _x22b * ch2*ch2
    )
    _T0 = [
        2 * _x22a * disp - _us * _x22b * IIIb**2,
        - 2*_x22a * _us + 2*_x22b * disp,
        - IIIa * _us * _x22b
    ]

    #_x11*u33

    _x22u22a = -2 * _x22a * _u22
    _x22u22b = -2 * _x22b * _u22
    x22u22 = [
        IIIb**2*_x22u22b/2,
        _x22u22a,
        IIIa*_x22u22b/2
        ]
    _x11u33 = -3 * _x11 * _u33
    x11u33 = [
        0,
        Ib/2 * _x11u33,
        Ia/2 * _x11u33
    ]


    _x31u11a = -x31a * _u11
    _x31u11b = -x31b * _u11
    x31u11 = [
        (1-mu**2)/2 * _x31u11a,
        (1+mu**2)/2 * _x31u11a + Ib/2*_x31u11b,
        Ia/2 *_x31u11b
    ]

    _x33u11a = -x33a * _u11
    _x33u11b = -x33b * _u11
    x33u11 = [
        (1-mu**2)/2 * _x33u11a,
        (1+mu**2)/2 * _x33u11a + Ib/2*_x33u11b,
        Ia/2 *_x33u11b
    ]

    _a11v33 = 3 * _a11 * _u33
    a11v33 = _a11v33*sh1*sh3
    a11v33 = [
        0,
        -Ib*_a11v33/2,
        Ia * _a11v33/2
    ]

    _a22v22 = 2 *_a22*_u22
    a22v22 = [
        - IIIb**2/2 * _a22v22,
        0,
        IIIa/2 * _a22v22
    ]

    _a20v22 = 2*_a20 * _u22
    a20v22 = [
        - IIIb**2/2 * _a20v22,
        0,
        IIIa/2 * _a20v22
    ]

    _a31v11a = a31a*_u11
    _a31v11b = a31b*_u11

    a31v11 = [
        - (1-mu**2) *_a31v11a/2,
        (1+mu**2) * _a31v11a/2 - Ib*_a31v11b/2,
        Ia *_a31v11b/2
    ]

    _a33v11a = a33a*_u11
    _a33v11b = a33b*_u11

    a33v11 = [
        - (1-mu**2)/2 *_a33v11a,
        (1+mu**2)/2 * _a33v11a - Ib/2*_a33v11b,
        Ia/2 *_a33v11b
    ]

    _x11p2u22 = -2 *_x11**2 * _u22
    x11p2u22 = [
        Oc/4 * _x11p2u22,
        Ob/2 * _x11p2u22,
        Oa/4 * _x11p2u22
    ]

    _x11x22u11a = -_x11 * _x22a * _u11
    _x11x22u11b = -_x11 * _x22b * _u11
    x11x22u11 = [
        + (1-mu**2)/2 * _x11x22u11a+ Oc/4 * _x11x22u11b,
        (1+mu**2)/2 * _x11x22u11a + Ob/2*_x11x22u11b,
        Oa/4 *_x11x22u11b
    ]

    _a11p2u22 = 2*_a11 **2 * _u22
    a11p2u22 = [
        Oc/4 * _a11p2u22,
        -Ob/2 * _a11p2u22,
        Oa/4 * _a11p2u22
    ]

    _a11a22u11 = _a11 * _a22 * _u11
    a11a22u11 = [
        -Oc/4 * _a11a22u11,
        0,
        Oa/4 * _a11a22u11
    ]

    _a11a20u11 = _a11 * _a20 * _u11
    a11a20u11 = [
        -Oc/4 * _a11a20u11,
        0,
        Oa/4 * _a11a20u11
    ]

    _x11a11v22 = -4* _x11 * _a11 * _u22
    x11a11v22 = [
        -Oc/4 * _x11a11v22,
        0,
        Oa/4 * _x11a11v22
    ]

    _x11a22v11 = -_x11 * _a22 * _u11
    x11a22v11 = [
        -Oc/4 * _x11a22v11,
        0,
        Oa/4 * _x11a22v11
    ]

    _x11a20v11 = -_x11 * _a20 * _u11
    x11a20v11 = [
        -Oc/4 * _x11a20v11,
        0,
        Oa/4 * _x11a20v11
    ]

    _x22a11v11a = -_x22a * _a11 * _u11
    _x22a11v11b = -_x22b * _a11 * _u11
    x22a11v11 = _x22a11v11a *sh1 * sh1 + _x22a11v11b * sh1 * sh1 * ch2
    x22a11v11 = (
            _x22a11v11a * ((1+mu**2)/2 *ch2 - (1-mu**2)/2 )
            + _x22a11v11b * ( Oa/4*ch4 - Ob/2*ch2 + Oc/4  )  #sh1 * sh1 * ch2
    )

    x22a11v11 = [
        - (1-mu**2)/2 * _x22a11v11a + Oc * _x22a11v11b/4,
        (1+mu**2)/2 * _x22a11v11a - Ob * _x22a11v11b/2,
        +Oa * _x22a11v11b/4
        ]

    _a11x11p2v11 = -_a11 * _x11 **2 * _u11 /2
    a11x11p2v11 = [
        - IIc * _a11x11p2v11 / 8,
        0,
        IIa * _a11x11p2v11 / 8
    ]


    _a11p2x11u11 = -_a11 **2 * _x11 * _u11 /2
    a11p2x11u11 = [
        - IIc * _a11p2x11u11 / 8,
        0,
        IIa * _a11p2x11u11 / 8
    ]

    _a11p3v11 = _a11 **3 * _u11 /6
    a11p3v11 = [
        3 * IIc * _a11p3v11/8,
        -IIb * _a11p3v11/2,
        IIa * _a11p3v11/8
    ]


    _x11p3u11 = _x11 **3 * _u11 /6
    x11p3u11 = [
        3 * IIc * _x11p3u11/8,
        IIb * _x11p3u11/2,
        IIa * _x11p3u11/8
    ]

    _terms = [
        _T0, #0
        x11u33, #-3 * x11 * u33, #1
        x22u22, #2
        x31u11,  # -x31*u11, #3a
        x33u11, #3b
        a11v33, #4
        a22v22, #5a
        a20v22, #5b
        a31v11, #6a
        a33v11, #6b
        x11p2u22, #7
        x11x22u11, #8
        a11p2u22, #9
        a11a22u11, #10a
        a11a20u11, #10b
        x11a11v22, #11
        x11a22v11, #12a
        x11a20v11, #12b
        x22a11v11, #13
        a11x11p2v11, #14
        a11p2x11u11, #15
        a11p3v11, #16
        x11p3u11, #17
        u42,
        u44
    ]

    if analytical:
        return _terms

    terms = []
    for term in _terms:
        if isinstance(term,list):
            terms.append(
                term[0] + term[1]*ch2 + term[2]*ch4
            )
        else:
            terms.append(term)

    return terms

def combine(  coef, index ):
    kd = np.ones(2)
    _terms = shared_4(kd,0,True)
    _result = zero
    for _T,_coef in zip(_terms,coef):
        _result += _T[index] * _coef

    _result = sp.together(_result)

    expr = sp.expand(_result.args[-1]) * sp.prod(_result.args[:-1])

    return expr

def sym_x40():
    s1s3 = zero # done
    s2s2 = one/2 # done
    s1s1 = one/2 # done
    c1c3 = zero # done
    c2c2 = one/2 # done
    c2 = zero # done
    c1c1 = one/2 # done
    s1s1c2 = -one/4 #done
    s1s2c1 = one/4 #done
    c1c1c2 = one/4
    c1c1s1s1 = one/8
    c1c1c1c1 = sp.sympify(3)/8
    s1s1s1s1 = sp.sympify(3)/8


    L =[
        c2,
        s1s3,
        s2s2,
        s1s1,
        s1s3,
        c1c3,
        c2c2,
        c2,
        c1c1,
        c1c3,
        s1s1c2,
        s1s2c1,
        c1c1c2,
        c1c1c2,
        c1c1, #10b
        s1s2c1, #11
        s1s1c2, #12a
        s1s1, #12b
        s1s2c1, #13
        c1c1s1s1, #14
        c1c1s1s1, #15
        c1c1c1c1, #16
        s1s1s1s1, #17
        zero,
        zero
    ]

    ch0 = combine(L, 0)
    ch2 = combine(L,1)
    ch4 = combine(L, 2)
    print(ch0)
    print(ch2)
    print(ch4)


def x40(kd,height):
    s1s3 =0 # done
    s2s2 = 0.5 # done
    s1s1 = 0.5 # done
    c1c3 = 0. # done
    c2c2 = 0.5 # done
    c2 = 0. # done
    c1c1 = 0.5 # done
    s1s1c2 = -1/4 #done
    s1s2c1 = 1/4 #done
    c1c1c2 = 1/4
    c1c1s1s1 = 1/8
    c1c1c1c1 = 3/8
    s1s1s1s1 = 3/8


    L40 =[
        c2,
        s1s3,
        s2s2,
        s1s1,
        s1s3,
        c1c3,
        c2c2,
        c2,
        c1c1,
        c1c3,
        s1s1c2,
        s1s2c1,
        c1c1c2,
        c1c1c2,
        c1c1, #10b
        s1s2c1, #11
        s1s1c2, #12a
        s1s1, #12b
        s1s2c1, #13
        c1c1s1s1, #14
        c1c1s1s1, #15
        c1c1c1c1, #16
        s1s1s1s1, #17
        0,
        0
    ]

    terms = shared_4(kd,height)

    _result = 0
    for fac,term in zip(L40,terms):
        _result += fac * term

    return _result

def sym_x42():
    s1s3 = one/2 # done
    s2s2 = zero # done
    s1s1 = -one/2 # done
    c1c3 = one/2 # done
    c2c2 = zero # done
    c2 = one # done
    c1c1 = one/2 # done
    s1s1c2 = one/2 #done
    s1s2c1 = zero #done
    c1c1c2 = one/2
    c1c1s1s1 = zero
    c1c1c1c1 = one/2
    s1s1s1s1 = -one/2



    L =[
        c2,
        s1s3,
        s2s2,
        s1s1,
        s1s3,
        c1c3,
        c2c2,
        c2,
        c1c1,
        c1c3,
        s1s1c2,
        s1s2c1,
        c1c1c2,
        c1c1c2,
        c1c1, #10b
        s1s2c1, #11
        s1s1c2, #12a
        s1s1, #12b
        s1s2c1, #13
        c1c1s1s1, #14
        c1c1s1s1, #15
        c1c1c1c1, #16
        s1s1s1s1, #17
        one,
        zero
    ]

    ch0 = combine(L, 0)
    ch2 = combine(L,1)
    ch4 = combine(L, 2)
    print(-ch0/2)
    print(-ch2/2)
    print(-ch4/2)

def x42(kd,height):
    s1s3 =1/2 # done
    s2s2 = 0 # done
    s1s1 = -1/2 # done
    c1c3 = 1/2 # done
    c2c2 = 0 # done
    c2 = 1 # done
    c1c1 = 1/2 # done
    s1s1c2 = 1/2 #done
    s1s2c1 = 0 #done
    c1c1c2 = 1/2
    c1c1s1s1 = 0
    c1c1c1c1 = 1/2
    s1s1s1s1 = -1/2


    L42 =[
        c2,
        s1s3,
        s2s2,
        s1s1,
        s1s3,
        c1c3,
        c2c2,
        c2,
        c1c1,
        c1c3,
        s1s1c2,
        s1s2c1,
        c1c1c2,
        c1c1c2,
        c1c1, #10b
        s1s2c1, #11
        s1s1c2, #12a
        s1s1, #12b
        s1s2c1, #13
        c1c1s1s1, #14
        c1c1s1s1, #15
        c1c1c1c1, #16
        s1s1s1s1, #17
        1,
        0,
    ]

    terms = shared_4(kd,height)

    _result = 0
    for fac,term in zip(L42,terms):
        _result += fac * term


    return -_result / 2


def sym_x44():
    s1s3 = -one/2 # done
    s2s2 = -one/2 # done
    s1s1 = zero # done
    c1c3 = one/2 # done
    c2c2 = one/2 # done
    c2 = zero # done
    c1c1 = zero # done
    s1s1c2 = -one/4 #done
    s1s2c1 = -one/4 #done
    c1c1c2 = one/4
    c1c1s1s1 = -one/8
    c1c1c1c1 = one/8
    s1s1s1s1 = one/8




    L =[
        c2,
        s1s3,
        s2s2,
        s1s1,
        s1s3,
        c1c3,
        c2c2,
        c2,
        c1c1,
        c1c3,
        s1s1c2,
        s1s2c1,
        c1c1c2,
        c1c1c2,
        c1c1, #10b
        s1s2c1, #11
        s1s1c2, #12a
        s1s1, #12b
        s1s2c1, #13
        c1c1s1s1, #14
        c1c1s1s1, #15
        c1c1c1c1, #16
        s1s1s1s1, #17
        zero,
        one
    ]

    ch0 = combine(L, 0)
    ch2 = combine(L,1)
    ch4 = combine(L, 2)
    print(-ch0/4)
    print(-ch2/4)
    print(-ch4/4)


def x44(kd,height):
    s1s3 =-1/2 # done
    s2s2 = -1/2 # done
    s1s1 = 0
    c1c3 = 1/2
    c2c2 = 1/2
    c2 = 0.
    c1c1 = 0
    s1s1c2 = -1/4
    s1s2c1 = -1/4 #done
    c1c1c2 = 1/4 #done
    c1c1s1s1 = -1/8
    c1c1c1c1 = 1/8
    s1s1s1s1 = 1/8


    L44 =[
        c2,
        s1s3,
        s2s2,
        s1s1,
        s1s3,
        c1c3,
        c2c2,
        c2,
        c1c1,
        c1c3,
        s1s1c2,
        s1s2c1,
        c1c1c2,
        c1c1c2,
        c1c1, #10b
        s1s2c1, #11
        s1s1c2, #12a
        s1s1, #12b
        s1s2c1, #13
        c1c1s1s1, #14
        c1c1s1s1, #15
        c1c1c1c1, #16
        s1s1s1s1, #17
        0,
        1,
    ]

    terms = shared_4(kd,height)

    _result = 0
    for fac,term in zip(L44,terms):
        _result += fac * term

    return -_result / 4






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
    return _ok


def coef():
    s1s3 =0 # done
    s2s2 = 0.5 # done
    s1s1 = 0.5 # done
    c1c3 = 0. # done
    c2c2 = 0.5 # done
    c2 = 0. # done
    c1c1 = 0.5 # done
    s1s1c2 = -1/4 #done
    s1s2c1 = 1/4 #done
    c1c1c2 = 1/4
    c1c1s1s1 = 1/8
    c1c1c1c1 = 3/8
    s1s1s1s1 = 3/8


    L40 =[
        c2,
        s1s3,
        s2s2,
        s1s1,
        s1s3,
        c1c3,
        c2c2,
        c2,
        c1c1,
        c1c3,
        s1s1c2,
        s1s2c1,
        c1c1c2,
        c1c1c2,
        c1c1, #10b
        s1s2c1, #11
        s1s1c2, #12a
        s1s1, #12b
        s1s2c1, #13
        c1c1s1s1, #14
        c1c1s1s1, #15
        c1c1c1c1, #16
        s1s1s1s1, #17
        0,
        0
    ]

    s1s3 =1/2 # done
    s2s2 = 0 # done
    s1s1 = -1/2 # done
    c1c3 = 1/2 # done
    c2c2 = 0 # done
    c2 = 1 # done
    c1c1 = 1/2 # done
    s1s1c2 = 1/2 #done
    s1s2c1 = 0 #done
    c1c1c2 = 1/2
    c1c1s1s1 = 0
    c1c1c1c1 = 1/2
    s1s1s1s1 = -1/2

    L42 =[
        c2,
        s1s3,
        s2s2,
        s1s1,
        s1s3,
        c1c3,
        c2c2,
        c2,
        c1c1,
        c1c3,
        s1s1c2,
        s1s2c1,
        c1c1c2,
        c1c1c2,
        c1c1, #10b
        s1s2c1, #11
        s1s1c2, #12a
        s1s1, #12b
        s1s2c1, #13
        c1c1s1s1, #14
        c1c1s1s1, #15
        c1c1c1c1, #16
        s1s1s1s1, #17
        1,
        0,
    ]

    s1s3 =-1/2 # done
    s2s2 = -1/2 # done
    s1s1 = 0
    c1c3 = 1/2
    c2c2 = 1/2
    c2 = 0.
    c1c1 = 0
    s1s1c2 = -1/4
    s1s2c1 = -1/4 #done
    c1c1c2 = 1/4 #done
    c1c1s1s1 = -1/8
    c1c1c1c1 = 1/8
    s1s1s1s1 = 1/8

    L44 =[
        c2,
        s1s3,
        s2s2,
        s1s1,
        s1s3,
        c1c3,
        c2c2,
        c2,
        c1c1,
        c1c3,
        s1s1c2,
        s1s2c1,
        c1c1c2,
        c1c1c2,
        c1c1, #10b
        s1s2c1, #11
        s1s1c2, #12a
        s1s1, #12b
        s1s2c1, #13
        c1c1s1s1, #14
        c1c1s1s1, #15
        c1c1c1c1, #16
        s1s1s1s1, #17
        0,
        1,
    ]


    a =(np.array(L40)+np.array(L42)+np.array(L44))
    for i,t in enumerate(a):
        print(i,t)

def final_x40(dim_depth,dim_height):
    mu = np.tanh(dim_depth)

    c= 0
    b = (-132*mu**8 + 40*mu**6 + 64*mu**4 - 168*mu**2 - 60)/(128*mu**6*(3*mu**2 + 1))
    a = (-21*mu**10 - 115*mu**8 + 78*mu**6 + 218*mu**4 + 87*mu**2 + 9)/(32*mu**8*(3*mu**2 + 1))
    ch2 = np.cosh(2*dim_depth + 2*dim_height) / np.cosh(2*dim_depth)
    ch4 = np.cosh(4*dim_depth + 4*dim_height) / np.cosh(4*dim_depth)
    return (b * ch2 + a*ch4)

def final_x42(dim_depth,dim_height):
    mu = np.tanh(dim_depth)


    c = (11*mu**6 - 19*mu**4 + 29*mu**2 - 21) / (96 * mu ** 6)
    b = (39*mu**12 + 724*mu**10 - 1269*mu**8 - 2032*mu**6 + 381*mu**4 + 540*mu**2 + 81)/(768*mu**10*(3*mu**2 + 1))
    a = (-345*mu**10 - 1411*mu**8 + 3462*mu**6 - 358*mu**4 - 957*mu**2 - 135)/(384*mu**8*(3*mu**2 + 1))
    ch2 = np.cosh(2*dim_depth + 2*dim_height) / np.cosh(2*dim_depth)
    ch4 = np.cosh(4*dim_depth + 4*dim_height) / np.cosh(4*dim_depth)
    return (c+ b * ch2 + a*ch4)


def final_x44(dim_depth,dim_height):
    mu = np.tanh(dim_depth)

    c = (83*mu**8 - 268*mu**6 + 314*mu**4 - 156*mu**2 + 27) / (
                384 * mu ** 8 )

    b = (-789*mu**10 + 907*mu**8 + 774*mu**6 - 1042*mu**4 + 15*mu**2 + 135)/(768*mu**8*(3*mu**2 + 1))
    a = ((591*mu**14 + 5393*mu**12 + 6175*mu**10 - 28135*mu**8 + 12997*mu**6 + 5355*mu**4 - 1971*mu**2 - 405)
         /(1536*mu**10*(mu**2 + 5)*(3*mu**2 + 1)))
    ch2 = np.cosh(2*dim_depth + 2*dim_height) / np.cosh(2*dim_depth)
    ch4 = np.cosh(4*dim_depth + 4*dim_height) / np.cosh(4*dim_depth)
    return (c+ b * ch2 + a*ch4)


if __name__ == "__main__":
    a = trms.validate( final_x40,fo.x40,'x40',False)
    b = trms.validate(final_x42, fo.x42, 'x42', False)
    b = trms.validate(final_x44, fo.x44, 'x44', False)

    sym_x44()
    # b = validate(a42, na42, 'a42')
    # c = validate(a44, na44, 'a44')
    #coef()
    # #print( a,b,c)
    # ur = 0.1
    # kd = np.tanh(100)
    # eps = ur*kd**3
    # kd=2
    #
    # a = shared_4(kd,0)
    # for ii,t in enumerate(a):
    #     if abs(t) > 5e-2:
    #         print(ii,t, np.tanh(kd))

    #coef()
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