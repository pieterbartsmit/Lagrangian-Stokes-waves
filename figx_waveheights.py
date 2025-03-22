from linearwavetheory.stokes_theory.regular_waves.eulerian_elevation_amplitudes import (
    dimensionless_material_surface_amplitude_first_harmonic, dimensionless_material_surface_amplitude_third_harmonic)


from linearwavetheory.stokes_theory.regular_waves.lagrangian_displacement_amplitudes import (
    lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic,
    lagrangian_dimensionless_vertical_displacement_amplitude_third_harmonic,
)

from linearwavetheory.settings import _GRAV

import numpy as np
import matplotlib.pyplot as plt

def plot_amplitudes():
    steepness = 0.2
    frequency = 0.1

    kd = np.linspace(1, 10, 100)

    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = kd / wavenumber

    a0 = 2*dimensionless_material_surface_amplitude_first_harmonic(steepness, wavenumber, depth,0,order=1)

    a1z = 2*dimensionless_material_surface_amplitude_first_harmonic(steepness, wavenumber, depth,0)-a0
    a3z = 2*dimensionless_material_surface_amplitude_third_harmonic(steepness, wavenumber, depth,0)
    a13z = a1z + a3z

    a1l = 2*lagrangian_dimensionless_vertical_displacement_amplitude_first_harmonic(steepness, wavenumber, depth,0)-a0
    a3l = 2*lagrangian_dimensionless_vertical_displacement_amplitude_third_harmonic(steepness, wavenumber, depth,0)
    a13l = a1l + a3l

    plt.plot( kd, a1z, label="First harmonic (Eulerian)")
    plt.plot( kd, a3z, label="Third harmonic (Eulerian)")
    plt.plot( kd, a1l, label="First harmonic (Lagrangian)")
    plt.plot( kd, a3l, label="Third harmonic (Lagrangian)")
    plt.plot( kd, a13z, label="First + Third harmonic (Eulerian)")
    plt.plot( kd, a13l, label="First + Third harmonic (Lagrangian)")
    plt.legend()
    plt.xlabel("kd")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_amplitudes()