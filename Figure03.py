"""
Script to generate Figure 3 of the paper "Lagrangian Surface Properties in Regular Stokes Waves".
"""

import numpy as np
from linearwavetheory.stokes_theory.regular_waves.mean_properties import dimensionless_variance, dimensionless_skewness, dimensionless_kurtosis
import matplotlib.pyplot as plt
from linearwavetheory.stokes_theory.regular_waves.mean_properties import dimensionless_stokes_drift,dimensionless_lagrangian_setup
import os

def plot_at_constant_steepness(ax, epsilon, color):
    relative_depth = np.linspace(0.1,4,100)
    z = 0 #np.linspace(-relative_depth, 0, 100)
    stokes_drift_2 = dimensionless_stokes_drift(epsilon, relative_depth, z, order=2)
    stokes_drift_4 = dimensionless_stokes_drift(epsilon, relative_depth, z, order=4)

    # Factor 2 to account for stochastic estimate
    delta = 2*(stokes_drift_4 - stokes_drift_2)/stokes_drift_2
    ax.plot(relative_depth, delta, color=color, linestyle='-',linewidth=2,label=f'$\\epsilon={epsilon:.1f}$')

def plot_at_constant_ursell(ax, ursell, color):
    z = 0 #np.linspace(-relative_depth, 0, 100)
    steepness = 10**np.linspace(-5,np.log10(0.5),10000)
    kd_ur = np.atanh((steepness/ursell)**(1/3))
    stokes_drift_2 = dimensionless_stokes_drift(steepness, kd_ur, z, order=2)
    stokes_drift_4 = dimensionless_stokes_drift(steepness, kd_ur, z, order=4)
    ur_03 = 2*(stokes_drift_4 - stokes_drift_2)/stokes_drift_2
    ax.plot( kd_ur, ur_03, '--', color=color,linewidth=1)

def plot_fourth_order_stokes_drift(ax):
    #fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    plot_at_constant_ursell(ax, 0.3, 'r')
    plot_at_constant_ursell(ax, 0.2, 'k')
    plot_at_constant_ursell(ax, 0.1, 'b')
    plot_at_constant_ursell(ax, 0.4, 'grey')

    plot_at_constant_steepness(ax, 0.1, 'b')
    plot_at_constant_steepness(ax, 0.2, 'k')
    plot_at_constant_steepness(ax, 0.3, 'r')
    plot_at_constant_steepness(ax, 0.4, 'grey')
    ax.legend()
    ax2 = ax.secondary_xaxis('top', functions=(lambda x: np.tanh(x), lambda x: np.atanh(x)))
    ax2.set_xlabel('$\\mu$')
    ax2.set_xticks([0.2,0.6,0.8,0.9])

    plt.sca(ax)
    plt.grid(True)
    plt.ylim([0., 1])
    plt.xlim([0.1,4])
    plt.xlabel( 'Relative depth $kd$')
    plt.ylabel('Excess Drift [-]')
    return fig


def plot_skewness_at_constant_steepness(ax,steepness,color):
    kd = np.linspace(0.1,4,100)
    linear = dimensionless_skewness(steepness, kd,0,reference_frame='eulerian',stochastic=True,order=4)
    lagrangian_variance =dimensionless_skewness(steepness, kd,0,reference_frame='lagrangian',stochastic=True)/linear-1
    ax.plot(kd,lagrangian_variance,color,linewidth=2,label='Lagrangian')


def plot_skewness_at_constant_ursell(ax,ursell,color):
    kd = np.linspace(0.1,4,100)
    steepness = ursell * np.tanh(kd)**3

    linear = dimensionless_skewness(steepness, kd,0,reference_frame='eulerian',stochastic=True,order=4)
    lagrangian_variance =dimensionless_skewness(steepness, kd,0,reference_frame='lagrangian',stochastic=True)/linear-1
    ax.plot(kd,lagrangian_variance,color,linewidth=1,linestyle='--',label='Lagrangian')


def plot_kurtosis_at_constant_steepness(ax,steepness,color):
    kd = np.linspace(0.1,4,100)
    linear = dimensionless_kurtosis(steepness, kd,0,reference_frame='eulerian',stochastic=True,order=4)
    lagrangian_variance =dimensionless_kurtosis(steepness, kd,0,reference_frame='lagrangian',stochastic=True)/linear-1
    ax.plot(kd,lagrangian_variance,color,linewidth=2,label='Lagrangian')


def plot_kurtosis_at_constant_ursell(ax,ursell,color):
    kd = np.linspace(0.1,4,100)
    steepness = ursell * np.tanh(kd)**3
    linear = dimensionless_kurtosis(steepness, kd,0,reference_frame='eulerian',stochastic=True,order=4)
    lagrangian_variance =dimensionless_kurtosis(steepness, kd,0,reference_frame='lagrangian',stochastic=True)/linear-1
    ax.plot(kd,lagrangian_variance,color,linewidth=1,linestyle='--',label='Lagrangian')


def plot_skewness(ax):
    plot_skewness_at_constant_steepness(ax, 0.1, 'b')
    plot_skewness_at_constant_steepness(ax, 0.2, 'k')
    plot_skewness_at_constant_steepness(ax, 0.3, 'r')
    plot_skewness_at_constant_steepness(ax, 0.4, 'grey')

    plot_skewness_at_constant_ursell(ax, 0.1, 'b')
    plot_skewness_at_constant_ursell(ax, 0.2, 'k')
    plot_skewness_at_constant_ursell(ax, 0.3, 'r')
    plot_skewness_at_constant_ursell(ax, 0.4, 'grey')

    ax2 = ax.secondary_xaxis('top', functions=(lambda x: np.tanh(x), lambda x: np.atanh(x)))
    ax2.set_xlabel('$\\mu$')
    ax2.set_xticks([0.2,0.6,0.8,0.9])

    plt.sca(ax)
    plt.grid(True)
    plt.ylim([0., 1])
    plt.xlim([0.1,4])
    plt.xlabel( 'Relative depth $kd$')
    plt.ylabel('Excess Skewness [-]')


def plot_kurtosis(ax):
    plot_kurtosis_at_constant_steepness(ax, 0.05, 'b')
    plot_kurtosis_at_constant_steepness(ax, 0.1, 'k')
    plot_kurtosis_at_constant_steepness(ax, 0.15, 'r')
    #plot_kurtosis_at_constant_steepness(ax, 0.4, 'grey')

    plot_kurtosis_at_constant_ursell(ax, 0.1, 'b')
    plot_kurtosis_at_constant_ursell(ax, 0.2, 'k')
    plot_kurtosis_at_constant_ursell(ax, 0.3, 'r')
    plot_kurtosis_at_constant_ursell(ax, 0.4, 'grey')

    ax2 = ax.secondary_xaxis('top', functions=(lambda x: np.tanh(x), lambda x: np.atanh(x)))
    ax2.set_xlabel('$\\mu$')
    ax2.set_xticks([0.2,0.6,0.8,0.9])

    plt.sca(ax)
    plt.grid(True)
    plt.ylim([0., 1])
    plt.xlim([0.1,4])
    plt.xlabel( 'Relative depth $kd$')
    plt.ylabel('Excess Kurtosis [-]')


def plot_variance_at_constant_steepness(ax,steepness,color):
    kd = np.linspace(0.1,4,100)

    linear = steepness**2 /2
    lagrangian_variance =dimensionless_variance(steepness, kd,0,reference_frame='lagrangian',stochastic=True)/linear-1
    ax.plot(kd,lagrangian_variance,color,linewidth=2)


def plot_variance_at_constant_ursell(ax,ursell,color):
    kd = np.linspace(0.1,4,100)
    steepness = ursell * np.tanh(kd)**3
    linear = steepness**2 /2
    lagrangian_variance =dimensionless_variance(steepness, kd,0,reference_frame='lagrangian',stochastic=True)/linear-1
    ax.plot(kd,lagrangian_variance,color,linewidth=1,linestyle='--',label=f'Ur={ursell:.1f}')


def plot_variance(ax):
    plot_variance_at_constant_steepness(ax, 0.1, 'b')
    plot_variance_at_constant_steepness(ax, 0.2, 'k')
    plot_variance_at_constant_steepness(ax, 0.3, 'r')
    plot_variance_at_constant_steepness(ax, 0.4, 'grey')

    plot_variance_at_constant_ursell(ax, 0.1, 'b')
    plot_variance_at_constant_ursell(ax, 0.2, 'k')
    plot_variance_at_constant_ursell(ax, 0.3, 'r')
    plot_variance_at_constant_ursell(ax, 0.4, 'grey')
    ax.legend()

    ax2 = ax.secondary_xaxis('top', functions=(lambda x: np.tanh(x), lambda x: np.atanh(x)))
    ax2.set_xlabel('$\\mu$')
    ax2.set_xticks([0.2,0.6,0.8,0.9])

    plt.sca(ax)
    plt.grid(True)
    plt.ylim([0., 1])
    plt.xlim([0.1,4])
    plt.xlabel( 'Relative depth $kd$')
    plt.ylabel('Excess Variance [-]')



if __name__ == "__main__":
    fig, ax = plt.subplots(2,2,figsize=(10, 8), dpi=600,sharex=True)
    plot_fourth_order_stokes_drift(ax[0,0])

    #fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    plot_variance(ax[0,1])

    #fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    plot_skewness(ax[1,0])

    #fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    plot_kurtosis(ax[1,1])
    fig.tight_layout()
    os.makedirs('./figures',exist_ok=True)
    fig.savefig('./figures/figure03.png')
    plt.show()
