import numpy as np
from linearwavetheory.settings import _GRAV
import linearwavetheory.stokes_theory.regular_waves as stokes
from linearwavetheory.settings import stokes_theory_options
import scipy
import matplotlib.pyplot as plt

# For convinience set the default to lagrangian reference frame - otherwise raise an error

_nonlinear_options = stokes_theory_options(reference_frame="lagrangian")


def integrate_stokes_solution( steepness , frequency, kd, number_of_waves=1, height=0, **kwargs):

    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = kd / wavenumber

    nonlinear_dispersion = stokes.nonlinear_dispersion_relation(
        steepness, wavenumber, depth, nonlinear_options=_nonlinear_options) * angular_frequency
    nonlinear_frequency = nonlinear_dispersion/2/np.pi

    # Determine the time step
    relative_timestep = kwargs.get('relative_dt',0.01)
    absolute_timestep = relative_timestep / nonlinear_frequency

    # Create the time vector
    nt = int(number_of_waves // relative_timestep) + 1

    time = np.linspace(0,nt-1,nt, endpoint=True) * absolute_timestep

    def differential_equation(t, y):
        x = y[0]
        z = y[1]
        u = stokes.horizontal_velocity(steepness,wavenumber,depth,t,x, z)[0]
        w = stokes.vertical_velocity(steepness,wavenumber,depth,t,x, z)[0]
        return np.array( [u,w] )

    initial_condition = np.array(
        [
            0.0,
            stokes.material_surface_vertical_elevation(steepness,wavenumber,depth,0,0,height)[0]
        ]
    )

    sol = scipy.integrate.solve_ivp(
        fun=differential_equation,
        t_span=[0,time[-1]],
        y0=initial_condition,
        t_eval=time,
        max_step=absolute_timestep
    )


    usa = stokes.stokes_drift(steepness, wavenumber, depth,height)

    x = sol.y[0,:]
    eta = sol.y[1,:]
    time = sol.t




    #us = x[100]/time[100]

    # Determine the oscillatory part - make sure the first point is the same (0)
    xl = scipy.signal.detrend(x)
    xl = xl + x[1] - xl[1]

    #xl = x - us * time


    # determine the stokes drift
    xlm = x - xl
    us = (xlm[-1]-xlm[0]) / (time[-1] - time[0])
    print('stokes ratio:',us/usa)


    return time, eta, xl, us

def plot_material_surfaces(steepness, frequency, kd, heights=( 0 , -0.1,-0.2, -0.3, -0.4,-0.5, -0.6, -0.7,-0.8 ) ):

    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = kd / wavenumber

    for height in heights:
        time, numerical_eta, numerical_x, numerical_stokes = integrate_stokes_solution(
            steepness, frequency, kd, number_of_waves=10, height=height/wavenumber
        )
        analytical_z = stokes.vertical_particle_displacement(
            steepness, wavenumber, depth, time, 0, height/wavenumber, order=3)

        analytical_z2 = stokes.vertical_particle_displacement(
            steepness, wavenumber, depth, time, 0, height/wavenumber, order=2)

        # plt.plot(time * frequency,time*0 + height, 'k--', linewidth=2, label='Analytical')
        # plt.plot(time * frequency, wavenumber * analytical_z2, 'grey', linewidth=2, label='Analytical')

        if height == 0:
            color = 'k'
            linewidth = 2
            linestyle = '-'
        else:
            color = 'k'
            linewidth = 1
            linestyle = '--'


        thin = 5
        plt.plot(time[0::thin] * frequency, wavenumber*numerical_eta[0::thin],'k.', label='Numerical')
        plt.plot(time * frequency, wavenumber * analytical_z,color,linewidth=linewidth,linestyle=linestyle, label='Analytical')
        plt.ylabel('$kz^\\prime$')
        plt.xlabel('$\\omega t^\\prime / (2\\pi)$')



    plt.xlim([0, 1])
    plt.ylim( [-0.5, 0.25] )
    #plt.legend()
    plt.grid()
    plt.show()

def plot_orbital(steepness, frequency, kd, height ):

    mu = np.tanh(kd)
    angular_frequency = 2 * np.pi * frequency
    wavenumber = angular_frequency**2 / _GRAV / mu
    depth = kd / wavenumber


    time, numerical_eta, numerical_x, numerical_stokes = integrate_stokes_solution(
        steepness, frequency, kd, number_of_waves=10, height=height/wavenumber
    )

    nonlinear_dispersion = stokes.lagrangian_dimensionless_nonlinear_dispersion_relation(
        steepness, wavenumber, depth,height/wavenumber) * angular_frequency
    nonlinear_frequency = nonlinear_dispersion/2/np.pi

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
    analytical_z = stokes.lagrangian_vertical_displacement(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=3)-height/wavenumber

    analytical_x = stokes.lagrangian_horizontal_displacement(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=3)

    analytical_z2 = stokes.lagrangian_vertical_displacement(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=2)-height/wavenumber

    analytical_x2 = stokes.lagrangian_horizontal_displacement(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=2)

    analytical_z1 = stokes.lagrangian_vertical_displacement(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=1)-height/wavenumber

    analytical_x1 = stokes.lagrangian_horizontal_displacement(
        steepness, wavenumber, depth, time, 0, height/wavenumber, order=1)


    setup = stokes.lagrangian_setup(steepness, wavenumber, depth, height/wavenumber) / steepness

    local_steepness = steepness * np.sinh(kd + height) / np.cosh(kd)
    scaling = wavenumber/local_steepness

    thin = 5

    plt.plot(scaling*numerical_x[0::thin], scaling*numerical_eta[0::thin],'kx', label='Num.')
    plt.plot(scaling*analytical_x, scaling*analytical_z,'k',linewidth=2, label='$O(\\epsilon^3)$')

    plt.plot(scaling * analytical_x2, scaling * analytical_z2, 'b', linewidth=1,linestyle='-',
             label='$O(\\epsilon^2)$')
    plt.plot(scaling * analytical_x1, scaling * analytical_z1, 'grey', linewidth=1, linestyle='--',label='$O(\\epsilon^1)$')


    local_steepness = steepness * np.sinh(kd + height) / np.cosh(kd)


    xlimit = ((np.max( analytical_x )*scaling +1e-12) // 0.1 + 2) *0.1
    ylimit = ((np.max(analytical_z) * scaling + 1e-12) // 0.1 + 2) * 0.1

    xlimit = [ -xlimit,xlimit]

    ylimit = xlimit



    plt.plot(0, np.mean(scaling*numerical_eta), 'kx', markersize=6)
    plt.plot(0,(setup),'ko',markersize=4)
    plt.plot( xlimit, [0,0], 'k--', linewidth=2)
    plt.xlim( xlimit )
    plt.ylim( ylimit)

    plt.gca().set_aspect('equal')
    plt.grid()

def plot_orbitals():
    heights = [ 0 , 1/3]
    kds = [ 6,  1.2]
    steepness = 0.2
    frequency = 0.1

    figure = plt.figure(figsize=(8, 8),dpi=600)

    nkd = len(kds)
    nheights = len(heights)
    for ii,height in enumerate(heights):
        for jj,kd in enumerate(kds):
            z = -kd * ii * height


            plt.subplot(nkd,nheights, ii*nheights+jj+1  )
            plot_orbital(steepness, frequency, kd, z)
            if jj == 0:
                plt.ylabel('$\\overline{\\eta}_L + \\eta_l$')

            if ii == 1:
                plt.xlabel('$x_l$')

    plt.legend(mode = "expand", ncol = 4)
    figure.tight_layout()


if __name__ == '__main__':
    plot_orbitals()