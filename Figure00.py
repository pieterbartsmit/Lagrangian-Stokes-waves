import roguewavespectrum
import numpy as np
import matplotlib.pyplot as plt
import os

def get_data()-> roguewavespectrum.Spectrum:
    return roguewavespectrum.Spectrum.from_netcdf('./data/SPOT-31849C.nc')

def plot_storm():


    plt.figure(figsize=(8,4),dpi=600)
    spotter = get_data()

    time = spotter.time
    hm0 = spotter.hm0()
    steepness = np.sqrt(spotter.mean_squared_slope())

    # panel 1
    ursell = spotter.ursell_number()
    plt.subplot(1,2,1)
    ax = plt.gca()
    plt.plot(time, hm0,'k')
    plt.ylim([0,6])
    plt.grid()
    plt.xticks(rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Reformat the time axis to day and hour in matplotlib
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%H'))
    plt.xlabel('Time (day-hour)')
    plt.ylabel('Significant wave height (m)')



    # panel 2
    plt.subplot(1, 2, 2)
    plt.plot(time,ursell, 'k',label='Ursell Number')
    plt.plot(time,steepness,'b', label='Steepness')
    plt.legend(fontsize=8)

    plt.ylabel('Nonlinearity measure [-]')
    plt.ylim([0,0.3])
    plt.xticks(rotation=45)
    plt.grid()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%H'))
    plt.xlabel('Time (day-hour)')

    plt.tight_layout()
    os.makedirs('./figures',exist_ok=True)
    plt.savefig('./figures/Figure00.png',dpi=600,transparent=True)

if __name__ == "__main__":
    plot_storm()