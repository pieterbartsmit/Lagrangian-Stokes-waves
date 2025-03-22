import numpy as np
from linearwavetheory.stokes_theory.regular_waves.mean_properties import dimensionless_stokes_drift,dimensionless_lagrangian_setup
import matplotlib.pyplot as plt

def plot_stokes(ax, relative_depth,color):
    mu = np.tanh(relative_depth)
    epsilon = 0.3
    print(epsilon/mu**3)
    z = np.linspace(-relative_depth,0,100)

    stokes_drift_2 = dimensionless_stokes_drift(epsilon, relative_depth, z, order=2)
    stokes_drift_4 = dimensionless_stokes_drift(epsilon, relative_depth, z,order=4)
    delta = (stokes_drift_4-stokes_drift_2)/stokes_drift_2
    ax.plot(delta,z,color=color,linestyle='-')
    ax.plot(delta*2, z,color=color,linestyle='--')

    ax.fill_betweenx(z, delta, delta*2,  color=color, alpha=0.05)
    plt.grid(True)

    plt.ylim([-2,0])



def plot_stokes2(ax, epsilon, color):
    relative_depth = np.linspace(0.5,3,100)
    mu = np.tanh(relative_depth)
    #epsilon = 0.3
    #print(epsilon / mu ** 3)
    z = 0 #np.linspace(-relative_depth, 0, 100)
    ur = epsilon/mu**3

    stokes_drift_2 = dimensionless_stokes_drift(epsilon, relative_depth, z, order=2)
    stokes_drift_4 = dimensionless_stokes_drift(epsilon, relative_depth, z, order=4)

    delta = (stokes_drift_4 - stokes_drift_2)

    delta_1 =  delta/stokes_drift_2
    delta_2 =  2*delta / stokes_drift_2


    #delta = (stokes_drift_4 - 0*stokes_drift_2) / stokes_drift_2
    ax.plot(relative_depth, delta_1, color=color, linestyle='-')
    ax.plot(relative_depth,delta_2, color=color, linestyle='--')
    ax.plot(relative_depth, ur**2, color=color, linestyle=':')
    plt.grid(True)

    plt.ylim([0., .5])
    plt.xlim([0.5,2.5])
    plt.xlabel( 'Relative depth $kd$')
    plt.ylabel('$\\epsilon^2 u_s^{(4)}/ u_s^{(2)}$')


if __name__ == "__main__":
    fig, ax = plt.subplots()
    plot_stokes2(ax, 0.05, 'b')
    plot_stokes2(ax, 0.1,'k')
    plot_stokes2(ax, 0.2,'r')

    plt.show()