import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
import matplotlib.pyplot as plt
from integrate import get_numerical_amplitudes



params = {
    "axes.labelsize": 12,
    "axes.labelcolor": "grey",
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 10,
    "xtick.color": "grey",
    "ytick.color": "grey",
    "ytick.labelsize": 10,
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.grid": False,
}

plt.rcParams.update(params)



def plot_amplitudes():
    fig = plt.figure()
    steepness =  [0.1,0.2,0.3,0.4] #[0.05,0.1,0.15,0.2,0.25]
    colors = ['c','r','b','g','k']
    for i,eps in enumerate(steepness):
        plot_hor_amplitude(eps,colors[i])



def plot_hor_amplitude(generalized_ursell,color):
    heigt = 0
    kds, _,numerical,us = get_numerical_amplitudes(generalized_ursell, heigt)
    numerical[:,0] = us

    frequency = 0.1
    mu = numpy.tanh(kds)
    epsilon = generalized_ursell * mu ** 3
    angular_frequency = 2 * numpy.pi * frequency
    wavenumber = angular_frequency ** 2 / _GRAV / mu
    depth  = kds / wavenumber

    a0 = an_stokes_drift(epsilon, kds, heigt)

    a11 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_first_harmonic(
        epsilon, wavenumber, depth, heigt/wavenumber, order=1
    )

    a1 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_first_harmonic(
        epsilon, wavenumber, depth, heigt/wavenumber, order=3
    )
    a2 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_second_harmonic(
        epsilon, wavenumber, depth, heigt/wavenumber, order=2
    )

    a3 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_third_harmonic(
        epsilon, wavenumber, depth, heigt/wavenumber, order=3
    )

    # x31 = nx31(wavenumber * depth, 0)
    # x33 = nx33(wavenumber * depth, 0)
    # a3 = x33 * epsilon**3
    # a1 = a11 + x31*epsilon**3


    _a42 = x42(kds, heigt) * epsilon**4
    _a44 = x44(kds, heigt) * epsilon**4
    a2 = a2 + _a42
    a4 = _a44

    # a4 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_horizontal_displacement_fourth_harmonic(
    #     epsilon, wavenumber, depth, heigt/wavenumber, order=4
    # )

    amps = [a0,a1,a2,a3,a4,0*a3]

    jj = 0
    epsilon0 = 0.2
    ylims_max = [ 4,0,0.5,0.25,0.5,0.5]
    ylims_min = [0,-2,-1.5, -1.5,-2,-2]
    scale = np.array([epsilon0**2,epsilon0,epsilon0**2,epsilon0**3,epsilon0**4,epsilon0**5])
    kds = np.tanh(kds)
    for amp in amps:
        plt.subplot(2, 3, jj+1)
        _scale = scale[jj]*mu**2
        plt.plot(kds,amp/_scale,color)

        plt.plot( kds, numerical[:,jj]/_scale,color,linestyle='--',marker='x')
        jj=jj+1

        plt.xlim([0.1,1])


        plt.ylim([ylims_min[jj-1], ylims_max[jj-1]])

def plot_ver_amplitude(generalized_ursell,color):

    heigt = 0
    kds, numerical,_,_ = get_numerical_amplitudes(generalized_ursell, heigt)


    frequency = 0.1
    mu = numpy.tanh(kds)
    epsilon = generalized_ursell * mu ** 3
    angular_frequency = 2 * numpy.pi * frequency
    wavenumber = angular_frequency ** 2 / _GRAV / mu
    depth  = kds / wavenumber

    a0 = stokes.mean_properties.dimensionless_lagrangian_setup(epsilon, kds, heigt,  order=4)

    a1 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic(
        epsilon, wavenumber, depth, heigt/wavenumber, order=4
    )
    a2 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_second_harmonic(
        epsilon, wavenumber, depth, heigt/wavenumber, order=4
    )

    a3 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_third_harmonic(
        epsilon, wavenumber, depth, heigt/wavenumber, order=3
    )

    a4 = stokes.lagrangian_displacement_amplitudes.lagrangian_dimensionless_vertical_displacement_amplitude_fourth_harmonic(
        epsilon, wavenumber, depth, heigt/wavenumber, order=4
    )

    amps = [a0,a1,a2,a3,a4,0*a3]

    jj = 0
    epsilon0 = 0.2
    ylims_max = [ 4,4,4,4,4,0.5]
    ylims_min = [0,0,0, -.5,-0.5,-.5]
    scale = np.array([epsilon0**2,epsilon0,epsilon0**2,epsilon0**3,epsilon0**4,epsilon0**5])
    kds = np.tanh(kds)
    for amp in amps:
        plt.subplot(2, 3, jj+1)
        _scale = scale[jj]*mu**3
        plt.plot(kds,amp/_scale,color)

        plt.plot( kds, numerical[:,jj]/_scale,color,linestyle='--',marker='x')
        jj=jj+1

        plt.xlim([0.1,1])


        plt.ylim([ylims_min[jj-1], ylims_max[jj-1]])




if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 12), dpi=600)

    plot_amplitudes()
    plt.tight_layout()
    plt.show()