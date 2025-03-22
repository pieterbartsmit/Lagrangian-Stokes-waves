import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
import scipy
import matplotlib.pyplot as plt
import os
from linearwavetheory.settings import stokes_theory_options
import integrate



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

def plot_orbital(steepness, frequency, kd, height ):

    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = kd / wavenumber


    time, numerical_eta, numerical_x, numerical_stokes,_ = integrate.integrate_stokes_solution(
        steepness, frequency, kd, number_of_waves=10, height=height/wavenumber
    )

    nonlinear_frequency = stokes.dimensionless_nonlinear_dispersion_relation(
        steepness, kd, height, nonlinear_options=_nonlinear_options) * frequency


    # Note because the lower order soluttions have different periods, but we want to show complete orbits, we do some
    # custom filtering here. Really this is to avoid drawing more markers than necessary for the numerical solution -
    # which looks messy. This will otherwise not ifluence the comparison.
    msk = time * nonlinear_frequency <= 1.0
    numerical_eta = numerical_eta[msk]
    numerical_x = numerical_x[msk]
    numerical_time = time[msk]

    msk = time * nonlinear_frequency <= 1.03
    time = time[msk]

    numerical_eta = numerical_eta - height/wavenumber
    analytical_z = stokes.vertical_particle_location(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=4)-height/wavenumber

    analytical_x = stokes.horizontal_particle_displacement(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=4)

    analytical_z2 = stokes.vertical_particle_location(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=2)-height/wavenumber

    analytical_x2 = stokes.horizontal_particle_displacement(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=2)

    analytical_z1 = stokes.vertical_particle_location(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=1)-height/wavenumber

    analytical_x1 = stokes.horizontal_particle_displacement(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=1)


    local_steepness = steepness * np.cosh(kd + height) / np.cosh(kd)
    scaling = wavenumber/local_steepness

    # Plot numerical solution, but decimate by factor 5 to avoid too many markers.
    thin = 5
    plt.plot(scaling*numerical_x[0::thin], scaling*numerical_eta[0::thin],'kx', label='Num.')

    # Plot third order solution
    plt.plot(scaling*analytical_x, scaling*analytical_z,'k',linewidth=2, label='$O(\\epsilon^4)$')

    # plot second and first order solutions for reference
    plt.plot(scaling * analytical_x2, scaling * analytical_z2, 'b', linewidth=1,linestyle='-',
             label='$O(\\epsilon^2)$')
    plt.plot(scaling * analytical_x1, scaling * analytical_z1, 'grey', linewidth=1, linestyle='--',label='$O(\\epsilon^1)$')

    # Set the limits of the plot
    limit = 1.5 #((np.max( analytical_x )*scaling +1e-12) // 0.25 + 2) *0.25
    xlimit = [ -limit,limit]
    ylimit = xlimit

    plt.xlim( xlimit )
    plt.ylim( ylimit)
    plt.gca().set_aspect('equal')

    plt.grid()

def plot_orbitals():
    heights = [ 0 , -.5]

    kds = [ 10,  1.0]

    #steepness = 0.3
    ursell = 0.3
    frequency = 0.1

    figure = plt.figure(figsize=(8, 8),dpi=600)

    nkd = len(kds)
    nheights = len(heights)
    panel = ['a','b','c','d']
    for ii,height in enumerate(heights):
        for jj,kd in enumerate(kds):
            z = kd * height
            mu = np.tanh(kd)
            steepness = mu**3 * ursell
            plt.subplot(nkd,nheights, ii*nheights+jj+1  )
            plot_orbital(steepness, frequency, kd, z)
            ax = plt.gca()
            if jj == 0:
                plt.ylabel('$\\hat{\\eta}_l / \\hat{x}_l^{(1,1)}$')
            else:
                ax.set_yticklabels([''])

            if ii == 1:
                plt.xlabel('$x_l/ \\hat{x}_l^{(1,1)}$ \n a')
            else:
                ax.set_xticklabels([''])



            plt.title(f'{panel[ii*nheights+jj]}: $\\overline{{z}} = {height*kd:.1f}$, $\\mu = {mu:.1f}$')


    figure.tight_layout()
    plt.legend(bbox_to_anchor=(-0.86, -.28), loc='lower left', ncol=4)
    return figure


if __name__ == '__main__':
    fig = plot_orbitals()



    # plot_setup(steepness)
    #
    # #os.makedirs('./figures', exist_ok=True)
    fig.savefig('./figures/orbitals.png')
    plt.show()
    #rekenregels()