"""
Functions to estimate Lagrangian drift from numerical integration of the Stokes solution.
"""

import numpy
import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
from linearwavetheory.stokes_theory.regular_waves.nonlinear_dispersion import nonlinear_dispersion_relation
import scipy
import os
from scipy.signal import argrelextrema


def integrate_stokes_solution(steepness, frequency, relative_depth, number_of_waves=1, relative_z=0, **kwargs):
    """
    Integrate the Stokes solution for a given steepness, frequency, and relative depth. The function caches results
    by default to avoid recomputing the solution for the same parameters. The solution is saved in a .npz file.

    :param steepness:
    :param frequency:
    :param relative_depth: relative depth of the wave (kd)
    :param number_of_waves: number of waves to compute the solution for. (approximate)
    :param relative_z: relative height where we want to compute the solution (kz)
    :param kwargs:
    :return:
    """

    relative_timestep = kwargs.get('relative_dt', 0.01)
    cache = kwargs.get('cache',True)
    force_recompute = kwargs.get('force_recompute', False)
    filename = f'solution_{steepness}+{frequency}+{relative_depth}+{number_of_waves}+{relative_z}+{relative_timestep}.npz'
    filename= kwargs.get('filename',filename)
    filename = f'./data/{filename}'

    if os.path.exists(filename) and cache and not force_recompute:
        data = np.load(filename)
        time = data['time']
        eta = data['eta']
        xl = data['xl']
        us = data['us']
        x = data['x']

        return time, eta, xl, us, x

    # Calculate the wavenumber/depth associated with the given frequency and relative depth
    mu = np.tanh(relative_depth)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = relative_depth / wavenumber
    z = relative_z / wavenumber

    nonlinear_frequency = nonlinear_dispersion_relation(
        steepness, wavenumber,depth,z,'lagrangian') / np.pi/2

    # Determine the time step
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

    if relative_z < 0:
        initial_condition = np.array(
            [
                0.0,
                stokes.material_surface_vertical_elevation(steepness,wavenumber,depth,0,0,z,order=4)[0]
            ]
        )
    else:
        # At the surface we can use a 5th order approximation - this helps in particular getting the 4th order setup
        # correct - which otherwise is really finicky if otherwise a 5th order velocity field is used.
        # (try setting order=4 and see what happens). Note that the other orders are not as sensitive to this.
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
        max_step=absolute_timestep[0]
    )

    x = sol.y[0,:]
    eta = sol.y[1,:]
    time = sol.t

    # We try to only get a whole number of waves- since we start at an extrema of elevation, we merely need to
    # find the last extrema of the elevation signal to get a whole number of waves.
    jj = argrelextrema(eta, np.greater)[0]

    eta = eta[0:jj[-1]]
    x = x[0:jj[-1]]
    time = time[0:jj[-1]]

    # Stokes drift is defined as the average horizontal velocity over one wave period. Approximating it from the
    # mean horizontal location as done here proved to be numerically the most stable.
    us = 2 * np.mean(x) / (time[-1] - time[0])

    # Lagrangian displacement is merely the horizontal location minus the stokes drift times time
    xl = x - us * time

    # Save the results to a .npz file for later use
    os.makedirs('./data',exist_ok=True)
    np.savez( filename , time=time, eta=eta, xl=xl, us=us,x=x)

    return time, eta, xl, us,x


def get_numerical_amplitudes(generalized_ursell, relative_z,force_recompute=False):
    frequency = 0.1
    mu = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.9999]

    kds = np.atanh(mu)
    nkd = len(kds)

    mu = numpy.tanh(kds)
    steepness = generalized_ursell* mu**3

    vertical = numpy.zeros((len(kds),6))
    horizontal = numpy.zeros((len(kds), 6))
    stokes_drift = numpy.zeros(len(kds))

    filename = f'./data/amplitudes_{generalized_ursell:0.2f}+{frequency:0.2f}+{relative_z:0.2f}.npz'
    if os.path.exists(filename) and not force_recompute:
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
        depth = kd / wavenumber

        nonlinear_frequency = (
            nonlinear_dispersion_relation(steepness[i],wavenumber,depth,relative_z / wavenumber,
                                          reference_frame='lagrangian')/2/np.pi
        )

        c = angular_frequency / wavenumber
        try:
            time, numerical_eta, numerical_x, us,_ = integrate_stokes_solution(
                steepness[i], frequency, kd, number_of_waves=100, relative_z=relative_z, force_recompute=force_recompute
            )
            vertical[i,:] = get_amplitudes(
                time, numerical_eta, relative_z, wavenumber, nonlinear_frequency, 'real')


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


