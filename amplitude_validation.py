import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
import linearwavetheory.stokes_theory.regular_waves.lagrangian_displacement_amplitudes as lda
import linearwavetheory.stokes_theory.regular_waves.mean_properties as mean
import scipy
import matplotlib.pyplot as plt
import os
from linearwavetheory.settings import stokes_theory_options
from integrate import integrate_stokes_solution, get_numerical_setup_stokes,get_numerical_amplitudes
from lagrangian_horizontal_coef import an_stokes_drift, x42, x44



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
_nonlinear_options = stokes_theory_options(reference_frame="lagrangian")


def plot_amplitude(generalized_ursell):
    heigt = 0
    kds, num_ver,num_hor,us = get_numerical_amplitudes(generalized_ursell, heigt)
    num_hor[:,0] = us

    frequency = 0.1
    mu = numpy.tanh(kds)
    epsilon = generalized_ursell * mu ** 3
    angular_frequency = 2 * numpy.pi * frequency
    wavenumber = angular_frequency ** 2 / _GRAV / mu
    depth  = kds / wavenumber

    a0 = an_stokes_drift(epsilon, kds, heigt)

    eta20 = mean.dimensionless_lagrangian_setup(epsilon, kds, heigt, order=2)
    eta40 = mean.dimensionless_lagrangian_setup(epsilon, kds, heigt, order=4) - eta20
    eta11 = lda.lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic(epsilon,wavenumber,depth,heigt/wavenumber,order=1)
    eta22 = lda.lagrangian_dimensionless_vertical_displacement_amplitude_second_harmonic(epsilon, wavenumber, depth,
                                                                                        heigt / wavenumber, order=2)
    eta31 = lda.lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic(epsilon, wavenumber, depth,
                                                                                        heigt / wavenumber, order=3)-eta11
    eta33 = lda.lagrangian_dimensionless_vertical_displacement_amplitude_third_harmonic(epsilon, wavenumber, depth,
                                                                                        heigt / wavenumber, order=3)
    eta42 = lda.lagrangian_dimensionless_vertical_displacement_amplitude_second_harmonic(epsilon, wavenumber, depth,
                                                                                        heigt / wavenumber, order=4)-eta22
    eta44 = lda.lagrangian_dimensionless_vertical_displacement_amplitude_fourth_harmonic(epsilon, wavenumber, depth,
                                                                                            heigt / wavenumber, order=4)




    x11 = lda.x11(kds,heigt) * epsilon
    x22 = lda.x22(kds, heigt) * epsilon**2
    x31 = lda.x31(kds, heigt) * epsilon**3
    x33 = lda.x33(kds, heigt) * epsilon**3
    x42 = lda.x42(kds, heigt) * epsilon**4
    x44 = lda.x44(kds, heigt) * epsilon**4
    ul20 = lda.ul20(kds, heigt) * epsilon**2
    ul40 = lda.ul40(kds, heigt) * epsilon**4


    an_ver = [eta40,eta31,eta42,eta33,eta44]
    num_ver = [num_ver[:,0]-eta20,num_ver[:,1]-eta11,num_ver[:,2]-eta22,num_ver[:,3],num_ver[:,4]]

    an_hor = [ul40,x31,x42,x33,x44]
    num_hor = [us-ul20,num_hor[:,1]-x11,num_hor[:,2]-x22,num_hor[:,3],num_hor[:,4]]

    jj = 0
    epsilon0 = generalized_ursell
    ylims_max = [ 4,0,0.5,0.25,0.5,0.5]
    ylims_min = [0,-2.5,-1.5, -1.5,-2,-2]
    panel = [1,2,3,2,3]

    lab = [None,'n=1','n=2','n=3','n=4']
    colors = ['k','k','k','r','r']

    scale = np.array([epsilon0**4,epsilon0**3,epsilon0**4,epsilon0**3,epsilon0**4,epsilon0**5])
    kds = np.tanh(kds)
    for an_hor_amp,an_ver_amp,num_hor_amp,num_ver_amp in zip(an_hor,an_ver,num_hor,num_ver):
        if jj == 0:
            ver_scale = scale[jj] * mu ** 5
            hor_scale = scale[jj] * mu ** 4
        else:
            ver_scale = scale[jj] * mu ** 3
            hor_scale = scale[jj] * mu ** 2

        color = colors[jj]

        plt.subplot(2, 3, panel[jj])
        plt.plot(kds, num_ver_amp / ver_scale, color, linestyle='--', marker='x')
        plt.plot(kds, an_ver_amp / ver_scale, color,label=lab[jj])

        if jj == 0:
            plt.ylabel('$\\eta_l^{(4,0)} \\mu^{7}$')
            plt.yticks([0.1, 0.4, 0.7, 1])
            plt.ylim([0.1, 1])
        else:
            plt.legend()

        if jj == 1:
            plt.ylabel('$\\eta^{(3,n)}_l \\mu^{6}$')
            plt.yticks([-0.1, 0.2, 0.5, 0.8])
            plt.ylim([-0.1, 0.8])

        if jj == 2:
            plt.ylabel('$\\eta^{(4,n)}_l \\mu^{9}$')
            plt.yticks([-0.3, -0.1, 0.1, 0.3])
            plt.ylim([-0.3, 0.3])


        plt.xticks([0.1, 0.4, 0.7, 1])
        plt.xlim([0.1, 1])
        plt.grid(True)

        plt.subplot(2, 3, panel[jj]+3)
        plt.plot(kds,an_hor_amp/hor_scale,color,label=lab[jj])
        plt.plot(kds, num_hor_amp / hor_scale, color, linestyle='--', marker='x')
        if jj == 0:
            plt.ylabel('$u_s^{(4)} \\mu^{8}$')
            plt.yticks([0.2, 0.7, 1.2, 1.7])
            plt.ylim([0.2, 1.7])
        else:
            plt.legend()

        if jj == 1:
            plt.ylabel('$x^{(3,n)}_l \\mu^{7}$')
            plt.yticks([-1.7, -1.1, -0.5, 0.1])
            plt.ylim([-1.7, 0.1])

        if jj == 2:
            plt.ylabel('$x^{(4,n)}_l \\mu^{10}$')
            plt.yticks([-0.4, -0.2, 0., 0.2])
            plt.ylim([-0.4, 0.2])

        plt.xlabel('$\\mu$')
        plt.grid(True)
        jj=jj+1
        plt.xticks([0.1,0.4,0.7,1])
        plt.xlim([0.1,1])



if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 6), dpi=600)

    plot_amplitude(0.1)
    plt.tight_layout()

    os.makedirs('./figures',exist_ok=True)
    fig.savefig('./figures/amplitude_validation.png')
    plt.show()