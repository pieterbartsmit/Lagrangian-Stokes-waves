import roguewavespectrum
import numpy as np
import matplotlib.pyplot as plt
import os

from linearwavetheory.stokes_theory.regular_waves.mean_properties import dimensionless_variance, dimensionless_skewness, dimensionless_kurtosis
from linearwavetheory.stokes_theory.regular_waves.mean_properties import dimensionless_stokes_drift,dimensionless_lagrangian_setup

def get_data()-> roguewavespectrum.Spectrum:
    return roguewavespectrum.Spectrum.from_netcdf('./data/SPOT-31849C.nc')

def plot_stokes_drift(ax):
    spotter = get_data()
    time = spotter.time
    steepness = np.sqrt(spotter.mean_squared_slope())
    kd = np.atanh(spotter.relative_depth())

    plt.sca(ax)
    stokes_drift_2 = dimensionless_stokes_drift(steepness, kd, 0, order=2)

    # Factor 2 to account for stochastic estimate
    stokes_drift_4 = 2*(dimensionless_stokes_drift(steepness, kd, 0, order=4) - stokes_drift_2) + stokes_drift_2
    stokes_drift_4 = (stokes_drift_4 - stokes_drift_2)/stokes_drift_2

    #plt.plot(time, stokes_drift_2,'k', label='Second-order')
    plt.plot(time, stokes_drift_4,'k',linewidth=2,label='Fourth-order')

    plt.xticks(rotation=45)
    plt.grid()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%H'))
    #plt.xlabel('Time (day-hour)')
    #plt.legend(fontsize=8)
    plt.ylabel('Excess Stokes Drift [-]')
    plt.ylim([0,0.25])

# plot variance
def plot_variance(ax):
    spotter = get_data()
    time = spotter.time
    steepness = np.sqrt(spotter.mean_squared_slope())
    kd = np.atanh(spotter.relative_depth())

    plt.sca(ax)
    variance_2 = dimensionless_variance(steepness, kd,0,reference_frame='lagrangian',stochastic=True,order=2)
    variance_4 = dimensionless_variance(steepness, kd,0,reference_frame='lagrangian',stochastic=True,order=4)
    variance_4e = dimensionless_variance(steepness, kd, 0, reference_frame='eulerian', stochastic=True, order=4)

    variance_4 = (variance_4 - variance_2) / variance_2
    variance_4e = (variance_4e - variance_2) / variance_2

    #plt.plot(time, variance_2,'k', label='Second-order')
    plt.plot(time, variance_4,'k',linewidth=2,label='Fourth-order Lagrangian')
    plt.plot(time, variance_4e,'b',label='Fourth-order Eulerian')

    plt.xticks(rotation=45)
    plt.grid()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%H'))
    #plt.xlabel('Time (day-hour)')
    #plt.legend(fontsize=8)
    plt.ylabel('Excess Variance [-]')
    plt.ylim([0,0.25])


# plot the kurtosis
def plot_kurtosis(ax):
    spotter = get_data()
    time = spotter.time
    steepness = np.sqrt(spotter.mean_squared_slope())
    kd = np.atanh(spotter.relative_depth())

    plt.sca(ax)
    kurtosis_2 = dimensionless_kurtosis(steepness, kd,0,reference_frame='lagrangian',stochastic=True,order=4)
    kurtosis_4 = dimensionless_kurtosis(steepness, kd,0,reference_frame='lagrangian',stochastic=True,order=6)
    kurtosis_4e = dimensionless_kurtosis(steepness, kd, 0, reference_frame='eulerian', stochastic=True, order=6)

    kurtosis_4 = (kurtosis_4 - kurtosis_2) / kurtosis_2
    kurtosis_4e = (kurtosis_4e - kurtosis_2) / kurtosis_2

    #plt.plot(time, kurtosis_2,'k', label='Fourth-order')
    plt.plot(time, kurtosis_4,'k',linewidth=2,label='Lagrangian')
    plt.plot(time, kurtosis_4e,'b',label='Eulerian')


    plt.xticks(rotation=45)
    plt.grid()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%H'))
    plt.xlabel('Time (day-hour)')
    plt.ylabel('Excess Kurtosis [-]')
    plt.legend(fontsize=8)
    plt.ylim([0,0.6])


# plot the skewness
def plot_skewness(ax):
    spotter = get_data()
    time = spotter.time
    steepness = np.sqrt(spotter.mean_squared_slope())
    kd = np.atanh(spotter.relative_depth())

    plt.sca(ax)
    skewness_2 = dimensionless_skewness(steepness, kd,0,reference_frame='lagrangian',stochastic=True,order=4)
    skewness_4 = dimensionless_skewness(steepness, kd,0,reference_frame='lagrangian',stochastic=True,order=6)
    skewness_4e = dimensionless_skewness(steepness, kd, 0, reference_frame='eulerian', stochastic=True, order=6)

    skewness_4 = (skewness_4 - skewness_2) / skewness_2
    skewness_4e = (skewness_4e - skewness_2) / skewness_2


    #plt.plot(time, skewness_2,'k', label='Fourth-order')
    plt.plot(time, skewness_4,'k',linewidth=2,label='Lagrangian')
    plt.plot(time, skewness_4e,'b',label='Eulerian')

    plt.xticks(rotation=45)
    plt.grid()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%H'))
    plt.xlabel('Time (day-hour)')
    plt.ylabel('Excess Skewness [-]')
    #plt.legend(fontsize=8)
    plt.ylim([0,0.6])



if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), dpi=600, sharex=True)
    plot_stokes_drift(ax[0,0])
    plt.title('(a)')

    plot_variance(ax[0,1])
    plt.title('(b)')

    plot_skewness(ax[1,0])
    plt.title('(c)')

    plot_kurtosis(ax[1,1])
    plt.title('(b)')

    plt.tight_layout()
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/Figure04.png', dpi=600, transparent=True)
    plt.show()