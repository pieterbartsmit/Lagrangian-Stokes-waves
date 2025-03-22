import numpy as np
from linearwavetheory.stokes_theory.regular_waves.mean_properties import dimensionless_variance, dimensionless_skewness
import matplotlib.pyplot as plt


def plot_variance(ax):
    mu = np.linspace(0.7,1,100)
    epsilon = 0.2
    kd = np.atanh(mu)
    kd[-1] = 20

    linear = epsilon**2 /2
    ur = epsilon/ mu**3


    eulerian_variance =dimensionless_variance(epsilon, kd,0,reference_frame='eulerian',stochastic=True)/linear
    lagrangian_variance =dimensionless_variance(epsilon, kd,0,reference_frame='lagrangian',stochastic=True)/linear

    eulerian_variance_deterministic =dimensionless_variance(epsilon, kd,0,reference_frame='eulerian',stochastic=False)/linear
    lagrangian_variance_deterministic =dimensionless_variance(epsilon, kd,0,reference_frame='lagrangian',stochastic=False)/linear


    ax.plot(mu,eulerian_variance,'k',label='Eulerian')
    ax.plot(mu,lagrangian_variance,'r',label='Lagrangian')
    ax.plot(mu,eulerian_variance_deterministic,'k--',label='Eulerian deterministic')
    ax.plot(mu,lagrangian_variance_deterministic,'r--',label='Lagrangian deterministic')

    plt.legend()

def plot_skewness(ax):
    mu = np.linspace(0.7,1,100)
    epsilon = 0.2
    kd = np.atanh(mu)
    kd[-1] = 30

    linear = 3*epsilon**4 /4
    ur = epsilon/ mu**3

    eulerian_skewness =dimensionless_skewness(epsilon, kd,0,reference_frame='eulerian',stochastic=True)/linear
    lagrangian_skewness =dimensionless_skewness(epsilon, kd,0,reference_frame='lagrangian',stochastic=True)/linear

    #eulerian_variance_deterministic =dimensionless_variance(epsilon, kd,0,reference_frame='eulerian',stochastic=False)/linear
    #lagrangian_variance_deterministic =dimensionless_variance(epsilon, kd,0,reference_frame='lagrangian',stochastic=False)/linear


    ax.plot(mu,eulerian_skewness,'k',label='Eulerian')
    ax.plot(mu,lagrangian_skewness,'r',label='Lagrangian')
    #ax.plot(mu,eulerian_variance_deterministic,'k--',label='Eulerian deterministic')
    #ax.plot(mu,lagrangian_variance_deterministic,'r--',label='Lagrangian deterministic')

    plt.legend()




if __name__ == "__main__":
    fig, ax = plt.subplots()
    #plot_variance(ax)
    plot_skewness(ax)
    plt.show()
