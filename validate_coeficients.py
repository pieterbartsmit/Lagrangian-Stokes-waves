import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves.lagrangian_displacement_amplitudes as lda
import matplotlib.pyplot as plt
import os
from integrate import get_numerical_amplitudes
import figure_params


def plot_amplitude(generalized_ursell):
    relative_z = 0
    kds, num_ver,num_hor,us = get_numerical_amplitudes(generalized_ursell, relative_z)
    num_hor[:,0] = us

    frequency = 0.1
    mu = numpy.tanh(kds)
    epsilon = generalized_ursell * mu ** 3
    angular_frequency = 2 * numpy.pi * frequency

    eta20 = lda.eta20(kds,relative_z) * epsilon**2
    eta40 = lda.eta40(kds,relative_z) * epsilon**4
    eta11 = lda.eta11(kds,relative_z) * epsilon
    eta22 = lda.eta22(kds,relative_z) * epsilon**2
    eta31 = lda.eta31(kds,relative_z) * epsilon**3
    eta33 = lda.eta33(kds,relative_z) * epsilon**3
    eta42 = lda.eta42(kds,relative_z) * epsilon**4
    eta44 = lda.eta44(kds,relative_z) * epsilon**4

    x11 = lda.x11(kds,relative_z) * epsilon
    x22 = lda.x22(kds, relative_z) * epsilon**2
    x31 = lda.x31(kds, relative_z) * epsilon**3
    x33 = lda.x33(kds, relative_z) * epsilon**3
    x42 = lda.x42(kds, relative_z) * epsilon**4
    x44 = lda.x44(kds, relative_z) * epsilon**4
    ul20 = lda.ul20(kds, relative_z) * epsilon**2
    ul40 = lda.ul40(kds, relative_z) * epsilon**4


    an_ver = [eta40,eta31,eta42,eta33,eta44]
    num_ver = [num_ver[:,0]-eta20,num_ver[:,1]-eta11,num_ver[:,2]-eta22,num_ver[:,3],num_ver[:,4]]

    an_hor = [ul40,x31,x42,x33,x44]
    num_hor = [us-ul20,num_hor[:,1]-x11,num_hor[:,2]-x22,num_hor[:,3],num_hor[:,4]]

    jj = 0
    epsilon0 = generalized_ursell
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
            #plt.ylim([0.1, 1])
        else:
            plt.legend()

        if jj == 1:
            plt.ylabel('$\\eta^{(3,n)}_l \\mu^{6}$')
            plt.yticks([-0.1, 0.2, 0.5, 0.8])
            #plt.ylim([-0.1, 0.8])

        if jj == 2:
            plt.ylabel('$\\eta^{(4,n)}_l \\mu^{9}$')
            plt.yticks([-0.3, -0.1, 0.1, 0.3])
            #plt.ylim([-0.3, 0.3])


        plt.xticks([0.1, 0.4, 0.7, 1])
        plt.xlim([0.1, 1])
        plt.grid(True)

        plt.subplot(2, 3, panel[jj]+3)
        plt.plot(kds,an_hor_amp/hor_scale,color,label=lab[jj])
        plt.plot(kds, num_hor_amp / hor_scale, color, linestyle='--', marker='x')
        if jj == 0:
            plt.ylabel('$u_s^{(4)} \\mu^{8}$')
            plt.yticks([0.2, 0.7, 1.2, 1.7])
            #plt.ylim([0.2, 1.7])
        else:
            plt.legend()

        if jj == 1:
            plt.ylabel('$x^{(3,n)}_l \\mu^{7}$')
            plt.yticks([-1.7, -1.1, -0.5, 0.1])
            #plt.ylim([-1.7, 0.1])

        if jj == 2:
            plt.ylabel('$x^{(4,n)}_l \\mu^{10}$')
            plt.yticks([-0.4, -0.2, 0., 0.2])
            #plt.ylim([-0.4, 0.2])

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
    fig.savefig('./figures/figure_amplitude_validation.png')
    plt.show()