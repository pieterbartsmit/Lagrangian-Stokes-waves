import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
import scipy
import matplotlib.pyplot as plt
import os
from linearwavetheory.settings import stokes_theory_options
import sympy as sp

def shared_4(kd,height=0):
    steepness = 1
    frequency = 0.1
    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency ** 2 / _GRAV / mu
    depth = kd / wavenumber

    height = height / wavenumber
    a11 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_first_harmonic(
        steepness, wavenumber, depth, height, order=1
    )
    a31 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_first_harmonic(
        steepness, wavenumber, depth, height, order=3
    ) - a11
    a22 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_second_harmonic(
        steepness, wavenumber, depth, height, order=2
    )
    a33 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_third_harmonic(
        steepness, wavenumber, depth, height, order=3
    )

    a44e = stokes.eulerian_elevation_amplitudes.dimensionless_material_surface_amplitude_fourth_harmonic(
        steepness, wavenumber, depth, height, order=4
    )
    a22e = stokes.eulerian_elevation_amplitudes.dimensionless_material_surface_amplitude_second_harmonic(
        steepness, wavenumber, depth, height, order=2
    )
    a42e = stokes.eulerian_elevation_amplitudes.dimensionless_material_surface_amplitude_second_harmonic(
        steepness, wavenumber, depth, height, order=4
    ) - a22e


    x11 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_first_harmonic(
        steepness, wavenumber, depth, height, order=1
    )
    x31 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_first_harmonic(
        steepness, wavenumber, depth, height, order=3
    ) - x11
    x22 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_second_harmonic(
        steepness, wavenumber, depth, height, order=2
    )
    x33 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_third_harmonic(
        steepness, wavenumber, depth, height, order=3
    )


    terms = [
        a44e, #T1a
        a42e, #T1b
        - 3 * x11 * a33, # T2a
        - x11 * a31, #T2b
        - 2 * x22 * a22, #T3
        - x33 * a11, #T4a
        - x31 * a11,  #T4b
        - 2 * x11 ** 2 * a22, #T5
        - x11 * x22 * a11, #T6
        + x11 ** 3 / 6 * a11 #T7
    ]

    return terms

def lag_fac():
    L40 = [
        0, #T1a
        0, #T1b
        0, #T2a
        0.5, #T2b
        0.5, #T3
        0, #T4a
        0.5, #T4b
        -1/4, #T5
        1/4, #T6
        3/8 #T7
    ]
    L42 = [
        0, #T1a
        1, #T1b
        0.5, #T2a
        -0.5, #T2b
        0., #T3
        0.5, #T4a
        -0.5, #T4b
        0.5, #T5
        0, #T6
        -0.5 #T7
    ]
    L44 = [
        1, #T1a
        0, #T1b
        -0.5, #T2a
        0, #T2b
        -0.5, #T3
        -0.5, #T4a
        0, #T4b
        -0.25, #T5
        -0.25, #T6
        0.125 #T7
    ]
    L40 = np.array(L40)
    L42 = np.array(L42)
    L44 = np.array(L44)



def a40(kd,height):
    L40 = [
        0, #T1a
        0, #T1b
        0, #T2a
        0.5, #T2b
        0.5, #T3
        0, #T4a
        0.5, #T4b
        -1/4, #T5
        1/4, #T6
        3/8 #T7
    ]

    terms = shared_4(kd,height)

    _result = 0
    for fac,term in zip(L40,terms):
        _result += fac * term

    return _result

def a42(kd,height):
    L42 = [
        0, #T1a
        1, #T1b
        0.5, #T2a
        -0.5, #T2b
        0., #T3
        0.5, #T4a
        -0.5, #T4b
        0.5, #T5
        0, #T6
        -0.5 #T7
    ]

    terms = shared_4(kd,height)

    _result = 0
    for fac,term in zip(L42,terms):
        _result += fac * term
    _result = _result

    frequency = 0.1
    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency ** 2 / _GRAV / mu
    depth = kd / wavenumber

    a22 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_second_harmonic(
        1,wavenumber,depth,height/wavenumber,order=2
    )

    a42 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_second_harmonic(
        1,wavenumber,depth,height/wavenumber,order=4
    ) - a22

    return _result


def a44(kd,height):
    L44 = [
        1, #T1a
        0, #T1b
        -0.5, #T2a
        0, #T2b
        -0.5, #T3
        -0.5, #T4a
        0, #T4b
        -0.25, #T5
        -0.25, #T6
        0.125 #T7
    ]

    terms = shared_4(kd,height)

    _result = 0
    for fac,term in zip(L44,terms):
        _result += fac * term

    frequency = 0.1
    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency ** 2 / _GRAV / mu
    depth = kd / wavenumber

    a44 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_fourth_harmonic(
        1,wavenumber,depth,height/wavenumber,order=4
    )

    return _result



def nshared_4(kd,height=0):
    steepness = 1
    frequency = 0.1
    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency ** 2 / _GRAV / mu
    depth = kd / wavenumber



    sh1 = np.sinh(kd+height ) / np.cosh(kd)
    ch1 = np.cosh(kd+height ) / np.cosh(kd)
    ch2 = np.cosh(2 * (kd+height) ) / np.cosh(2*kd)
    sh2 = np.sinh(2 * (kd+height) ) / np.cosh(2*kd)
    sh3 = np.sinh(3 * (kd+height) ) / np.cosh(3*kd)
    ch3 = np.cosh(3 * (kd+height) ) / np.cosh(3*kd)
    ch4 = np.cosh(4 * (kd+height) ) / np.cosh(4*kd)
    sh4 = np.sinh(4 * (kd+height) ) / np.cosh(4*kd)

    a11 = sh1/mu
    a31 =  (
            +(-21 + 19 * mu ** 2 - 14 * mu ** 4) / 32 / mu ** 5 * sh1
            +  (9 + 23 * mu ** 2 - 12 * mu ** 4) / 32 / mu ** 5 * sh3
    )
    x22 = - 3 / 8 * (1 / mu ** 4 - 1) * ch2
    a22 = (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 4 * sh2

    a33 =  (
            (1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 5 * sh1
            + (27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 7 / 192 * sh3
            )


    a44e = [
            (1 - mu ** 4) * (-27 + 30 * mu ** 2 - 11 * mu ** 4) / mu ** 8 / 384,
            + (-mu ** 12 - 32 * mu ** 10 - 97 * mu ** 8 + 280 * mu ** 6 + 141 * mu ** 4 + 2376 * mu ** 2 + 405) / mu ** 10 / 1536 / (
                    5 + mu ** 2)
    ]


    a42e= [
            (89 * mu ** 10 - 681 * mu ** 8 + 262 * mu ** 6 + 762 * mu ** 4 - 351 * mu ** 2 - 81) / (768 * mu ** 10),
            + (1 + 6 * mu ** 2 + mu ** 4) * (27 - 12 * mu ** 2 + 5 * mu ** 4) / mu ** 8 / 192
    ]

    x11 = - ch1/mu
    x31 = (16 - 12 * mu ** 2 + 15 * mu ** 4) / mu ** 5 / 32 * ch1 + (- 28 - 72 * mu ** 2 + 33 * mu ** 4) / 32 / mu ** 5* ch3

    x33 =  (
            (28 - 60 * mu ** 2 + 29 * mu ** 4) / mu ** 5 / 96* ch1
            +  (-27 - 11 * mu ** 2 - 123 * mu ** 6 + 167 * mu ** 4) / 192 / mu ** 7 * ch3
    )

    a33 =  (
            (1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 5 * sh1
            + (27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 7 / 192 * sh3
            )

    _x11 = 1 #ch1
    a33 =  (
            3*(1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 6 * sh1 * ch1
            + 3*(27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 8 / 192 * sh3 * ch1
            )

    Ia = (1 + 6*mu**2 + mu**4 ) / (1+3*mu**2)
    Ib = (1 - mu**4) / (1+3*mu**2)
    IIa = (1 + 6*mu**2 + mu**4 )
    IIb = (1 - mu**4)
    IIIa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)**2
    IIIb = (1 - mu**2)/(1+mu**2)
    Oa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)
    Ob = (1 - mu**2)



    x11a33 = [
        (
                (1 + mu ** 2) / 2 * 3*(1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 6
                +(1 - mu**4) / (1+3*mu**2)/2 * 3*(27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 8 / 192
         ),
        (1 + 6*mu**2 + mu**4 ) / (1+3*mu**2)/2 *  3*(27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 8 / 192
    ]


    x11a31=  (
            (
            +(-21 + 19 * mu ** 2 - 14 * mu ** 4) / 32 / mu ** 6 * (1+mu**2)/2
            + (9 + 23 * mu ** 2 - 12 * mu ** 4) / 32 / mu ** 6 * Ib/2),
            +  (9 + 23 * mu ** 2 - 12 * mu ** 4) / 32 / mu ** 6 *Ia/2
    )


    x22a22 = [ -2*(1-mu**2)/4/mu**2 * (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 4,
            + 6 / 8 * (1 / mu ** 4 - 1)*(1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 4 * IIIa/2
        ]

    x33a11 =  (
        (
                -(38 * mu ** 4 - 68 * mu ** 2 + 30) / (96 * mu ** 6)* (1+mu**2)/2
                + Ib/2 *  (-39 * mu ** 6 + 53 * mu ** 4 - 5 * mu ** 2 - 9) / (64 * mu ** 8)
         ),
        -  (-39 * mu ** 6 + 53 * mu ** 4 - 5 * mu ** 2 - 9) / (64 * mu ** 8) * Ia/2
    )

    alpha = (10 * mu ** 4 - 12 * mu ** 2 + 18) / (32 * mu ** 5)
    beta = (42 * mu ** 4 - 76 * mu ** 2 - 30) / (32 * mu ** 5)

    x31a11 = (
        (
                -(10 * mu ** 4 - 12 * mu ** 2 + 18) / (32 * mu ** 6) * (1+mu**2)/2
                + Ib / 2*(42 * mu ** 4 - 76 * mu ** 2 - 30) / (32 * mu ** 6)
        )
        ,
        -(42 * mu ** 4 - 76 * mu ** 2 - 30) / (32 * mu ** 6)* Ia/2
    )

    a22 = (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 4 * sh2


    x11a22 = -2*(1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 6 * (Oa * sh4/4 + Ob*sh2/2)

    x11a22 = [-2 * (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 6 * Ob  / 2,
              -2 * (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 6 * Oa  / 4]

    _x11x22a11 = - (1 - mu ** 2) / 8 / mu ** 4 *sh1 * ch1

    x11x22a11 =  [ (1 - mu ** 2) * (1+mu**2) / 8 / mu ** 4,- 3 * (1  -  mu ** 4) /mu**6/ 8  * Oa/4]
    #x11x11x11a11 = - 1/mu**4  / 6 * (1/8 * IIa *sh4 + 1/4 * IIb *sh2 )
    x11x11x11a11 = [
        - 1 / mu ** 4 / 6 * 1 / 4 * IIb,
        - 1 / mu ** 4 / 6 * 1 / 8 * IIa
        ]

    T =[
        a44e, #T1a
        a42e, #T1b
        x11a33,
        x11a31,
        x22a22,
        x33a11,
        x31a11,
        x11a22,
        x11x22a11,
        x11x11x11a11
    ]

    terms = []
    for t in range(len(T)):
        _T = T[t]
        terms.append(_T[0] * sh2+_T[1]*sh4)





    return terms



def na40(kd,height):
    L40 = [
        0, #T1a
        0, #T1b
        0, #T2a
        1/2, #T2b
        1/2, #T3
        0, #T4a
        1/2, #T4b
        -1/4, #T5
        1/4, #T6
        3/8 #T7
    ]

    terms = nshared_4(kd,height)

    _result = 0
    for fac,term in zip(L40,terms):
        _result += fac * term

    mu = np.tanh(kd)
    a = (-90*mu**8 - 4*mu**6 - 6*mu**4 - 136*mu**2 - 44)/(128*mu**6*(3*mu**2 + 1))
    #b = (mu**4 + 6*mu**2 + 1)*(-36.0*mu**8 + 35.0*mu**6 + 126.0*mu**4 + 64.0*mu**2 + 9.0)/(128*mu**8*(mu**2 + 1)*(3*mu**2 + 1))
    b = (-36.0 * mu ** 10 - 145.0 * mu ** 8 + 445.0 * mu ** 6 + 410.0 * mu ** 4 + 109.0 * mu ** 2 + 9.0) / (
                128 * mu ** 8 * (3 * mu ** 2 + 1))

    sh2 = np.sinh(2*(kd+height))/np.cosh(2*kd)
    sh4 = np.sinh(4*(kd+height))/np.cosh(4*kd)
    _res = a*sh2 + b*sh4

    return _result

def na42(kd,height):
    L42 = [
        0, #T1a
        1, #T1b
        1/2, #T2a
        -1/2, #T2b
        0., #T3
        1/2, #T4a
        -1/2, #T4b
        1/2, #T5
        0, #T6
        -1/2 #T7
    ]

    terms = nshared_4(kd,height)

    _result = 0
    for fac,term in zip(L42,terms):
        _result += fac * term
    _result = _result

    mu = np.tanh(kd)
    a = (537*mu**12 - 1556*mu**10 + 421*mu**8 + 2960*mu**6 - 13*mu**4 - 540*mu**2 - 81)/(768*mu**10*(3*mu**2 + 1))
    b = (mu**4 + 6*mu**2 + 1)*(576*mu**8 - 568*mu**6 - 944*mu**4 + 416*mu**2 + 216)/(768*mu**8*(mu**2 + 1)*(3*mu**2 + 1))

    sh2 = np.sinh(2*(kd+height))/np.cosh(2*kd)
    sh4 = np.sinh(4*(kd+height))/np.cosh(4*kd)
    _res = a*sh2 + b*sh4

    return _result


def na44(kd,height):
    L44 = [
        1, #T1a
        0, #T1b
        -0.5, #T2a
        0, #T2b
        -0.5, #T3
        -0.5, #T4a
        0, #T4b
        -0.25, #T5
        -0.25, #T6
        0.125 #T7
    ]

    terms = nshared_4(kd,height)

    _result = 0
    for fac,term in zip(L44,terms):
        _result += fac * term

    mu = np.tanh(kd)
    #a = (336*mu**10 - 532*mu**8 - 244*mu**6 + 616*mu**4 - 116*mu**2 - 108)/(768*mu**8*(3*mu**2 + 1))
    #b = (-603.0*mu**16 - 6112.0*mu**14 - 11872.0*mu**12 + 22296.0*mu**10 + 15990.0*mu**8 - 17984.0*mu**6 - 3344.0*mu**4 + 2376.0*mu**2 + 405)/(1536*mu**10*(mu**2 + 1)*(mu**2 + 5)*(3*mu**2 + 1))

    a = (336*mu**10 - 532*mu**8 - 244*mu**6 + 616*mu**4 - 116*mu**2 - 108)/(768*mu**8*(3*mu**2 + 1))
    b = (-603.0*mu**14 - 5509.0*mu**12 - 6363.0*mu**10 + 28659.0*mu**8 - 12669.0*mu**6 - 5315.0*mu**4 + 1971.0*mu**2 + 405.0)/(1536*mu**10*(mu**2 + 5)*(3*mu**2 + 1))

    sh2 = np.sinh(2*(kd+height))/np.cosh(2*kd)
    sh4 = np.sinh(4*(kd+height))/np.cosh(4*kd)
    _res = a*sh2 + b*sh4

    return _result


def validate(target,new,name):
    kd = np.linspace(1,5,100)
    height = [0,-0.5]

    _ok = True
    for z in height:
        ref = np.squeeze(target(kd,z))
        res = new(kd,z)
        diff = np.max(np.abs(ref-res)/np.abs(ref))
        if diff > 1e-12:
            _ok = False
            print( f"- {name} Height: {z} -> Max diff: {diff}")

        plt.figure()
        plt.plot(kd,ref,'ko')
        plt.plot(kd,res)
    #plt.show()
    return _ok

if __name__ == "__main__":

    import validation.fourth_order_coef as terms
    # a = validate(a40,na40,'a40')
    # b = validate(a42, na42, 'a42')
    # c = validate(a44, na44, 'a44')

    a = validate(terms.eta40,na40,'a40')
    b = validate(terms.eta42, na42, 'a42')
    c = validate(terms.eta44, na44, 'a44')
    #print( a,b,c)
