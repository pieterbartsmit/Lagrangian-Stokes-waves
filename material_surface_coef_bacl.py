import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
import scipy
import matplotlib.pyplot as plt
import os
from linearwavetheory.settings import stokes_theory_options
import sympy as sp


def target_42(kd):
    depth = 10
    wavenumber = kd/depth
    steepness = 1

    a22 = stokes.eulerian_elevation_amplitudes.dimensionless_surface_amplitude_second_harmonic(steepness,wavenumber,depth,order=2)
    a42 = stokes.eulerian_elevation_amplitudes.dimensionless_surface_amplitude_second_harmonic(steepness, wavenumber,
                                                                                               depth, order=4) - a22
    return a42


def target_44(kd):
    depth = 10
    wavenumber = kd/depth
    steepness = 1
    a44 = stokes.eulerian_elevation_amplitudes.dimensionless_surface_amplitude_fourth_harmonic(steepness, wavenumber,
                                                                                               depth, order=4)
    return a44


def est_42(kd,height=0):

    mu = np.tanh(kd)
    depth = 10
    wavenumber = kd/depth
    steepness = 1


    #disp =  (81-603*mu**2 + 3618*mu**4 -3662* mu**6 + 1869*mu**8 -663 *mu**10)/ (1024 * mu ** 10)
    disp =  (9 - 10 * mu ** 2 + 9 * mu ** 4) / 16 / mu ** 4

    a11 = stokes.eulerian_elevation_amplitudes.dimensionless_material_surface_amplitude_first_harmonic(
        steepness,wavenumber,depth,height/wavenumber,order=1
    )
    a31 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_first_harmonic(
        steepness,wavenumber,depth,height/wavenumber,order=3
    ) - a11
    a22 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_second_harmonic(
        steepness,wavenumber,depth,height/wavenumber,order=2
    )

    a33 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_third_harmonic(
        steepness,wavenumber,depth,height/wavenumber,order=3
    )
    ch1 = np.cosh(kd + height) / np.cosh(kd)
    ch2 = np.cosh(2 * kd + 2 * height) / np.cosh(2 * kd)
    ch3 = np.cosh(3 * kd + 3 * height) / np.cosh(3 * kd)
    ch4 = np.cosh(4 * kd + 4 * height) / np.cosh(4 * kd)
    sh1 = np.sinh(kd + height) / np.cosh(kd)
    sh2 = np.sinh(2 * kd + 2 * height) / np.cosh(2 * kd)
    sh3 = np.sinh(3 * kd + 3 *  height) / np.cosh(3 * kd)
    sh4 = np.sinh(4 * kd + 4 *  height) / np.cosh(4 * kd)

    u11 = ch1*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_first_harmonic(steepness,wavenumber*depth,order=1)
    u22 = ch2*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_second_harmonic(steepness,wavenumber*depth,order=2)
    u33 = ch3*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_third_harmonic(steepness,wavenumber*depth,order=3)
    u44 = ch4 * stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_fourth_harmonic(steepness,
                                                                                                    wavenumber * depth,
                                                                                                    order=4)

    v11 = sh1*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_first_harmonic(steepness,wavenumber*depth,order=1)
    v22 = sh2*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_second_harmonic(steepness,wavenumber*depth,order=2)
    v33 = sh3*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_third_harmonic(steepness,wavenumber*depth,order=3)
    v44 = sh4 * stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_fourth_harmonic(steepness,
                                                                                                    wavenumber * depth,
                                                                                                        order=4)
    v42 = sh2 * stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_second_harmonic(steepness,
                                                                                                     wavenumber * depth,
                                                                                                     order=4) - v22

    _result = (
        - 2*disp * a22 #1
        -1/2 *a11 * u33 #2
        + 0 * a11**2 * v22 #3
        + 0*a22*a11*v11 #4
        +1/8 * a11**3 *u11 #5
        + 0 * 2* u22*a22 #6
        + a11 * v11 * a22 #7
        + 3/2 * u11 * a33 #8a
        + 1/2 * u11 * a31 #8b
        + 0 * v44 #9
        + v42
        + 3*a11*u33/2 #10
        + 2*a11**2*v22/2 #11
        + 0 * 2 * a22 * u22 #12
        - a33 * u11 / 2 #13a
        + a31 * u11 / 2 #13b
        + a11 * a22 * v11 * 0 #14
        + 1/24 * a11**3 * u11 #15
    )

    facs = (
        - 2, #1
        1, #2 + #10
        + 1, #3 + #11
        + 1, #4 + #14 + #7
        +1/6, #5 + #15
        + 0, #6 + #12
        + 1, #8a + #13a
        + 1, #8b + #13b
        + 0, #9a
        + 1, #9b
    )

    terms = (
        a22*disp, #1
        u33*a11, #T2*T2v, #2
        a11**2 * v22, #3
        a22*a11*v11, #4
        a11**3*u11, #ok  #a11**3*u11, #T5*T5v, # **3 *u11, #5
        u22*a22, #6
        u11 * a33, #8a
        u11 * a31, #8b
        v44, #9
        v42,
    )


    _result2 = 0
    for fac,term in zip(facs,terms):
        _result2 += fac*term

    #return _result/2

    return _result2/2


def est_42_subs(kd,height=0):

    mu = np.tanh(kd)
    depth = 10
    wavenumber = kd/depth
    steepness = 1


    #disp =  (81-603*mu**2 + 3618*mu**4 -3662* mu**6 + 1869*mu**8 -663 *mu**10)/ (1024 * mu ** 10)
    disp =  (9 - 10 * mu ** 2 + 9 * mu ** 4) / 16 / mu ** 4

    ch1 = np.cosh(kd + height) / np.cosh(kd)
    ch2 = np.cosh(2 * kd + 2 * height) / np.cosh(2 * kd)
    ch3 = np.cosh(3 * kd + 3 * height) / np.cosh(3 * kd)
    ch4 = np.cosh(4 * kd + 4 * height) / np.cosh(4 * kd)
    sh1 = np.sinh(kd + height) / np.cosh(kd)
    sh2 = np.sinh(2 * kd + 2 * height) / np.cosh(2 * kd)
    sh3 = np.sinh(3 * kd + 3 *  height) / np.cosh(3 * kd)
    sh4 = np.sinh(4 * kd + 4 *  height) / np.cosh(4 * kd)



    Ia = (1 + 6*mu**2 + mu**4 ) / (1+3*mu**2)
    Ib = (1 - mu**4) / (1+3*mu**2)
    IIa = (1 + 6*mu**2 + mu**4 )
    IIb = (1 - mu**4)
    IIIa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)**2
    IIIb = (1 - mu**2)/(1+mu**2)
    Oa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)
    Ob = (1 - mu**2)

    a11 = 1/mu * sh1
    a31 =(
            +(-21 + 19 * mu ** 2 - 14 * mu ** 4) / 32 / mu ** 5 *sh1
            + (9 + 23 * mu ** 2 - 12 * mu ** 4) / 32 / mu ** 5*sh3
    )

    a22 = (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 4 * sh2

    a33 = (
            (1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 5 * sh1
            +  (27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 7 / 192 * sh3
    )

    u11 = ch1/mu
    u22 = ch2* 3 * ( 1  -mu**4) / 4/ mu**4
    u33 = ch3*3/64 / mu * ( 9 / mu**6 + 5 / mu**4 + 39 - 53 / mu**2)


    v11 = sh1/mu
    v22 = sh2* 3 * ( 1  -mu**4) / 4/ mu**4
    v33 = sh3*3/64 / mu * ( 9 / mu**6 + 5 / mu**4 + 39 - 53 / mu**2)
    v44 = sh4 * 4 * (1 - mu ** 2) * (
                (
                405 + 1161 * mu ** 2 - 6462 * mu ** 4 + 3410 * mu ** 6 + 1929 * mu ** 8 + 197 * mu ** 10
                )
                / (1536 * mu ** 10 * (5 + mu ** 2))
        )
    v42 = sh2 * (
                ( -81 -135 * mu ** 2 +810*mu**4 +182* mu**6 - 537*mu**8 + 145*mu**10)
              / (384 * mu**10)
        )

    facs = (
        - 2, #1
        1, #2 + #10
        + 1, #3 + #11
        + 1, #4 + #14 + #7
        +1/6, #5 + #15
        + 0, #6 + #12
        + 1, #8a + #13a
        + 1, #8b + #13b
        + 0, #9a
        + 1, #9b
    )

    T1 = (1 + mu ** 2) * (3 - mu ** 2) * (9 - 10 * mu ** 2 + 9 * mu ** 4) / 128 / mu ** 8
    T1v = sh2

    T2 = ( 27 + 15 *mu**2 + 117 * mu**6 - 159 * mu**4) /mu**8/64
    T2v = Ia *sh4 /2 - Ib*sh2 /2

    T3 = 3 * ( 1  -mu**4) / 4/ mu**6
    T3v = Oa * sh4/4 - Ob* sh2/2

    T4 = (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 6
    T4v = Oa * sh4/4 - Ob* sh2/2

    T5 =1/mu**4
    T5v = IIa *sh4 /8 - IIb*sh2 /4

    T6 =3 * ( 1  -mu**4) * (1 + mu ** 2) * (3 - mu ** 2) / 32 / mu ** 8
    T6v = IIIa *sh4 /2

    T8aa =  (27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 8 / 192
    T8ab = (1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 6
    T8aav = Ia*sh4/2 + Ib*sh2/2
    T8abv = (1+mu**2)/2 *sh2

    T8ba = (9 + 23 * mu ** 2 - 12 * mu ** 4) / 32 / mu ** 6
    T8bb = (-21 + 19 * mu ** 2 - 14 * mu ** 4) / 32 / mu ** 6
    T8bav = Ia*sh4/2 + Ib*sh2/2
    T8bbv = (1+mu**2)/2 *sh2

    T9a =4 * (1 - mu ** 2) * (
                (
                405 + 1161 * mu ** 2 - 6462 * mu ** 4 + 3410 * mu ** 6 + 1929 * mu ** 8 + 197 * mu ** 10
                )
                / (1536 * mu ** 10 * (5 + mu ** 2))
        )
    T9b =  ( -81 -135 * mu ** 2 +810*mu**4 +182* mu**6 - 537*mu**8 + 145*mu**10) / (384 * mu**10)
    T9av = sh4
    T9bv = sh2

    terms = (
        T1*T1v, #1
        T2*T2v, #2
        T3*T3v, #a11**2 * v22, #3
        T4*T4v, #a22*a11*v11, #4
        T5*T5v, # **3 *u11, #5
        T6*T6v, #u22*a22, #6
        T8aa*T8aav + T8ab*T8abv, #u11 * a33, #8a
        T8ba*T8bav + T8bb*T8bbv, #u11 * a31, #8b
        T9a*T9av,#v44, #9
        T9b*T9bv #v42,
    )


    _result = 0
    for fac,term in zip(facs,terms):
        _result += fac*term

    return _result/2


def est_44(kd,height=0):

    mu = np.tanh(kd)
    depth = 10
    wavenumber = kd/depth
    steepness = 1


    #disp =  (81-603*mu**2 + 3618*mu**4 -3662* mu**6 + 1869*mu**8 -663 *mu**10)/ (1024 * mu ** 10)
    disp =  (9 - 10 * mu ** 2 + 9 * mu ** 4) / 16 / mu ** 4

    a11 = stokes.eulerian_elevation_amplitudes.dimensionless_material_surface_amplitude_first_harmonic(
        steepness,wavenumber,depth,height,order=1
    )
    a31 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_first_harmonic(
        steepness,wavenumber,depth,height,order=3
    ) - a11
    a22 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_second_harmonic(
        steepness,wavenumber,depth,height,order=2
    )

    a33 = stokes.lagrangian_displacement_amplitudes.dimensionless_material_surface_amplitude_third_harmonic(
        steepness,wavenumber,depth,height,order=3
    )

    ch1 = np.cosh(kd + height) / np.cosh(kd)
    ch2 = np.cosh(2 * kd + 2 * wavenumber * height) / np.cosh(2 * kd)
    ch3 = np.cosh(3 * kd + 3 * wavenumber * height) / np.cosh(3 * kd)
    ch4 = np.cosh(4 * kd + 4 * wavenumber * height) / np.cosh(4 * kd)
    sh1 = np.sinh(kd + height) / np.cosh(kd)
    sh2 = np.sinh(2 * kd + 2 * wavenumber * height) / np.cosh(2 * kd)
    sh3 = np.sinh(3 * kd + 3 * wavenumber * height) / np.cosh(3 * kd)
    sh4 = np.sinh(4 * kd + 4 * wavenumber * height) / np.cosh(4 * kd)

    u11 = ch1*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_first_harmonic(steepness,wavenumber*depth,order=1)
    u22 = ch2*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_second_harmonic(steepness,wavenumber*depth,order=2)
    u33 = ch3*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_third_harmonic(steepness,wavenumber*depth,order=3)
    u44 = ch4 * stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_fourth_harmonic(steepness,
                                                                                                    wavenumber * depth,
                                                                                                    order=4)

    v11 = sh1*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_first_harmonic(steepness,wavenumber*depth,order=1)
    v22 = sh2*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_second_harmonic(steepness,wavenumber*depth,order=2)
    v33 = sh3*stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_third_harmonic(steepness,wavenumber*depth,order=3)
    v44 = sh4 * stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_fourth_harmonic(steepness,
                                                                                                    wavenumber * depth,
                                                                                                        order=4)
    v42 = sh2 * stokes.eularian_velocity_amplitudes.dimensionless_velocity_amplitude_second_harmonic(steepness,
                                                                                                     wavenumber * depth,
                                                                                                     order=4) - v22

    Ia = (1 + 6*mu**2 + mu**4 ) / (1+3*mu**2)
    Ib = (1 - mu**4) / (1+3*mu**2)
    IIa = (1 + 6*mu**2 + mu**4 )
    IIb = (1 - mu**4)
    IIIa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)**2
    IIIb = (1 - mu**2)/(1+mu**2)
    Oa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)
    Ob = (1 - mu**2)

    _ch3sh1 = Ia *sh4 /2 - Ib*sh2 /2
    _sh1p2sh2 = Oa * sh4/4 - Ob* sh2/2
    _sh3ch1 = IIa *sh4 /8 - IIb*sh2 /4
    _sh2ch2 = IIIa *sh4 /2
    _ch1sh3 = Ia*sh4/2 + Ib*sh2/2
    _ch1sh1 = (1+mu**2)/2 *sh2

    # facs = (
    #     0,
    #     +1/2,
    #     + 1/2,
    #     + 1/4, #4
    #     +1/16,
    #     + 1,
    #     + 1/2, #7
    #     + 3/2,
    #     + 0, #8b
    #     + 1,
    #     + 0, #9b
    #     + 3/2, #10
    #     + 1/2,
    #     + 1,
    #     + 1 / 2, #13a
    #     + 0, #13b
    #     + 1/4,
    #     + 1/48,
    # )

    facs = (
        0, #1
        +2 , #2 + #10
        + 1, #3 + #11
        + 1 , #4 +14 + #7
        +1/12, #5 + 15
        + 2, #6 + #12
        + 2, #8a + #13a
        + 0, #8b + #13b
        + 1, #9a
        + 0, #9b
    )

    T1 = (1 + mu ** 2) * (3 - mu ** 2) * (9 - 10 * mu ** 2 + 9 * mu ** 4) / 128 / mu ** 8
    T1v = sh2

    T2 = (27 + 15 * mu ** 2 + 117 * mu ** 6 - 159 * mu ** 4) / mu ** 8 / 64
    T2v = Ia * sh4 / 2 - Ib * sh2 / 2

    T3 = 3 * (1 - mu ** 4) / 4 / mu ** 6
    T3v = Oa * sh4 / 4 - Ob * sh2 / 2

    T4 = (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 6
    T4v = Oa * sh4 / 4 - Ob * sh2 / 2

    T5 = 1 / mu ** 4
    T5v = IIa * sh4 / 8 - IIb * sh2 / 4

    T6 = 3 * (1 - mu ** 4) * (1 + mu ** 2) * (3 - mu ** 2) / 32 / mu ** 8
    T6v = IIIa * sh4 / 2

    T8aa = (27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 8 / 192
    T8ab = (1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 6
    T8aav = Ia * sh4 / 2 + Ib * sh2 / 2
    T8abv = (1 + mu ** 2) / 2 * sh2

    T8ba = (9 + 23 * mu ** 2 - 12 * mu ** 4) / 32 / mu ** 6
    T8bb = (-21 + 19 * mu ** 2 - 14 * mu ** 4) / 32 / mu ** 6
    T8bav = Ia*sh4/2 + Ib*sh2/2
    T8bbv = (1+mu**2)/2 *sh2

    T9a = 4 * (1 - mu ** 2) * (
            (
                    405 + 1161 * mu ** 2 - 6462 * mu ** 4 + 3410 * mu ** 6 + 1929 * mu ** 8 + 197 * mu ** 10
            )
            / (1536 * mu ** 10 * (5 + mu ** 2))
    )
    T9b = (-81 - 135 * mu ** 2 + 810 * mu ** 4 + 182 * mu ** 6 - 537 * mu ** 8 + 145 * mu ** 10) / (384 * mu ** 10)
    T9av = sh4
    T9bv = sh2

    terms = (
        T1 * T1v,  # 1
        T2 * T2v,  # 2
        T3 * T3v,  # a11**2 * v22, #3
        T4 * T4v,  # a22*a11*v11, #4
        T5 * T5v,  # **3 *u11, #5
        T6 * T6v,  # u22*a22, #6
        T8aa * T8aav + T8ab * T8abv,  # u11 * a33, #8a
        T8ba * T8bav + T8bb * T8bbv,  # u11 * a31, #8b
        T9a * T9av,  # v44, #9
        T9b * T9bv  # v42,
    )


    _result = 0
    for fac,term in zip(facs,terms):
        _result += fac*term
    #

    return _result/4




def comp42():
    kd = np.linspace(0.5,5,1000)

    a42 = target_42(kd)
    a42_est = est_42_subs(kd)
    a42_est_z = est_42_subs(kd,0.5)
    a42_est_old_z = est_42(kd,0.5)
    # plt.plot(kd,a42 * 6/5 )
    # plt.plot(kd,a42_est * 6/5 )

    plt.plot(kd,a42_est_z * 6/5 )
    plt.plot(kd,a42_est_old_z * 6/5 )

    #plt.ylim(([0, 5]))
    plt.grid()

    d = np.max(np.abs(a42 - a42_est)/a42)
    d2 = np.max(np.abs(a42_est_z - a42_est_old_z)/a42_est_z)


    print('42',d,d2)



    return

def comp44():
    kd = np.linspace(0.5,5,1000)

    a44 = target_44(kd)
    a44_est = est_44(kd)
    plt.plot(kd,a44 *3)
    plt.plot(kd,a44_est *3)

    d = np.max(np.abs(a44 - a44_est) / a44)
    print('44',d)

    plt.grid()


    return


if __name__ == "__main__":

    plt.figure()
    comp42()
    #plt.figure()
    #comp44()
    plt.show()