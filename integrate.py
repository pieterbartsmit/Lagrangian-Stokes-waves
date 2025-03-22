import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
import scipy
import matplotlib.pyplot as plt
import os
from linearwavetheory.settings import stokes_theory_options

from scipy.signal import argrelextrema








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

def integrate_stokes_solution( steepness , frequency, kd, number_of_waves=1, height=0, **kwargs):

    if 'relative_dt' in kwargs:
        relative_timestep = kwargs.get('relative_dt', 0.01)
        filename = f'./data/b{steepness}+{frequency}+{kd}+{number_of_waves}+{height}+{relative_timestep}.npz'
    else:
        filename = f'./data/{steepness}+{frequency}+{kd}+{number_of_waves}+{height}.npz'

    relative_timestep = kwargs.get('relative_dt', 0.01)

    cache = kwargs.get('cache',True)

    try:
        if os.path.exists(filename) and cache:
            data = np.load(filename)
            time = data['time']
            eta = data['eta']
            xl = data['xl']
            us = data['us']
            x = data['x']

            return time, eta, xl, us, x
    except KeyError:
        pass

    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = kd / wavenumber

    nonlinear_frequency = stokes.dimensionless_nonlinear_dispersion_relation(
        steepness, wavenumber*depth,nonlinear_options=_nonlinear_options) * frequency

    # Determine the time step
    relative_timestep = kwargs.get('relative_dt',0.01)
    absolute_timestep = relative_timestep / nonlinear_frequency

    # Create the time vector
    nt = int(number_of_waves // relative_timestep) + 1

    time = np.linspace(0,nt-1,nt, endpoint=True) * absolute_timestep

    def differential_equation(t, y):
        x = y[0]
        z = y[1]
        u = stokes.horizontal_velocity(steepness,wavenumber,depth,t,x, z, order=5)[0]
        w = stokes.vertical_velocity(steepness,wavenumber,depth,t,x, z, order=5)[0]

        return np.array( [u,w] )

    if height < 0:
        initial_condition = np.array(
            [
                0.0,
                stokes.material_surface_vertical_elevation(steepness,wavenumber,depth,0,0,height,order=5)[0]
            ]
        )
    else:
        print('ja')
        initial_condition = np.array(
            [
                0.0,
                stokes.free_surface_elevation(steepness,wavenumber,depth,0,0,order=5)[0]
            ]
        )

        if initial_condition[1]*wavenumber > steepness*2:
            raise ValueError('Initial condition is not correct')

    sol = scipy.integrate.solve_ivp(
        fun=differential_equation,
        t_span=[0,time[-1]],
        y0=initial_condition,
        t_eval=time,
        max_step=absolute_timestep
    )

    x = sol.y[0,:]
    eta = sol.y[1,:]
    t= time
    time = sol.t

    jj = argrelextrema(eta, np.greater)[0]

    eta = eta[0:jj[-1]]
    x = x[0:jj[-1]]
    time = time[0:jj[-1]]

    us = 2 * np.mean(x) / (time[-1] - time[0])
    xl = x - us * time

    # xl = scipy.signal.detrend(x)
    #
    # xl = xl + x[1] - xl[1]
    #
    # xlm = x - xl
    # us = (xlm[-1]-xlm[0]) / (time[-1] - time[0])

    os.makedirs('./data',exist_ok=True)
    np.savez( filename , time=time, eta=eta, xl=xl, us=us,x=x)

    return time, eta, xl, us,x

def get_numerical_setup_stokes(steepness, height):
    frequency = 0.1
    kds = np.array([0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,4.5,5])
    nkd = len(kds)

    stokes_numerical = numpy.zeros_like(kds)
    setup_numerical = numpy.zeros_like(kds)

    filename = f'./data/setup_stokes_{nkd}_{steepness}+{frequency}+{height}.npz'
    if os.path.exists(filename):
        data = np.load(filename)
        kds = data['kds']
        stokes_numerical =data['stokes_numerical']
        setup_numerical = data['setup_numerical']
        return kds, stokes_numerical, setup_numerical

    for i, kd in enumerate(kds):

        mu = numpy.tanh(kd)
        angular_frequency = 2 * numpy.pi * frequency
        wavenumber = angular_frequency**2 / _GRAV / mu

        c = angular_frequency / wavenumber
        try:
            time, numerical_eta, numerical_x, us = integrate_stokes_solution(
                steepness, frequency, kd, number_of_waves=100, height=height/wavenumber
            )
            setup_numerical[i] = np.mean(numerical_eta) * wavenumber
            stokes_numerical[i] = us / c
        except ValueError:
            stokes_numerical[i] = np.nan
            setup_numerical[i] = np.nan

    os.makedirs('./data',exist_ok=True)
    np.savez(filename, kds=kds, stokes_numerical=stokes_numerical, setup_numerical=setup_numerical)

    return kds, stokes_numerical, setup_numerical


def get_numerical_amplitudes(generalized_ursell, dimensionless_height):
    frequency = 0.1
    mu = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.9999]

    #kds = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,4.5,5])
    kds = np.atanh(mu)
    nkd = len(kds)

    mu = numpy.tanh(kds)
    steepness = generalized_ursell* mu**3

    vertical = numpy.zeros((len(kds),6))
    horizontal = numpy.zeros((len(kds), 6))
    stokes_drift = numpy.zeros(len(kds))

    filename = f'./data/amplitudes_ursell_combined_{nkd}_{generalized_ursell}+{frequency}+{dimensionless_height}.npz'
    if os.path.exists(filename):
        data = np.load(filename)
        kds = data['kds']
        vertical =data['vertical']
        horizontal = data['horizontal']
        stokes_drift = data['stokes_drift']
        return kds, vertical, horizontal, stokes_drift

    for i, kd in enumerate(kds):
        print(i,kd,generalized_ursell)
        mu = numpy.tanh(kd)
        angular_frequency = 2 * numpy.pi * frequency
        wavenumber = angular_frequency**2 / _GRAV / mu

        nonlinear_frequency = stokes.dimensionless_nonlinear_dispersion_relation(
            steepness[i], kd, nonlinear_options=_nonlinear_options) * frequency

        c = angular_frequency / wavenumber
        try:
            time, numerical_eta, numerical_x, us,_ = integrate_stokes_solution(
                steepness[i], frequency, kd, number_of_waves=100, height=dimensionless_height/wavenumber
            )
            vertical[i,:] = get_amplitudes(
                time, numerical_eta, dimensionless_height, wavenumber, nonlinear_frequency,'real')

            horizontal[i,:] = get_amplitudes(
                time, numerical_x, 0, wavenumber, nonlinear_frequency,'imag')

            stokes_drift[i] = us / c


        except ValueError:
            vertical[i] = np.nan


    os.makedirs('./data',exist_ok=True)
    np.savez(filename, kds=kds, vertical=vertical,horizontal=horizontal,stokes_drift=stokes_drift)

    return kds, vertical, horizontal,stokes_drift


def get_amplitudes(time, signal,offset,scaling,nonlinear_frequency,sign='real'):
    N = len(signal) * 10
    freq = np.fft.rfftfreq(N, d=time[1] - time[0])
    window = np.hanning(len(signal))
    amplitude_correction = 2

    signal = signal - offset / scaling
    signal = signal * scaling * window * amplitude_correction

    jj = argrelextrema(signal, np.greater)[0]
    signal = signal[0:jj[-1]]

    _fft = np.fft.rfft(signal, N)

    # ensure we (almost) fit an exact amount

    if sign == 'real':
        _signs = np.sign(np.real(_fft))
    else:
        _signs = np.sign(np.imag(_fft))
    _fft = 2 * np.abs(_fft) / N * 10
    # plot_amp(steepness,nonlinear_frequency,freq,_fft)

    out = np.zeros(6)
    for jj in range(6):

        if jj == 0:
            fac = 0.5
            jjf = 0
        else:
            f = (jj - 1 + 0.75) * nonlinear_frequency
            jf = np.argmin(np.abs(freq - f))

            ii = np.argmax(_fft[jf:])
            jjf = jf + ii
            fac = 1

        out[jj] = _fft[jjf] * fac * _signs[jjf]
    return out


def plot_amp(steepness,nonlinear_freq,freq,_fft):
    N = len(freq)
    plt.plot(freq/nonlinear_freq,_fft,'o')

    for jj in range(5):
        plt.axvline(jj,linestyle='--',color='grey')

    plt.xlim([0,6])
    plt.grid()
    plt.show()


