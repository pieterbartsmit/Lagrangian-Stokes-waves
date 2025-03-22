
import validation.terms as terms
import linearwavetheory.stokes_theory.regular_waves.lagrangian_displacement_amplitudes as lda
from validation.second_order_coef import x22

def x3n(dimensionless_depth, dimensionless_height,n):
    c2 = terms.cos(2)
    c3 = terms.cos(3)
    c1 = terms.cos(1)
    s2 = terms.sin(2)
    s1 = terms.sin(1)

    _disp = terms.dispersion(dimensionless_depth, dimensionless_height)

    _eta11 = terms.al11(dimensionless_depth, dimensionless_height) * c1
    _eta22 = terms.al22(dimensionless_depth, dimensionless_height) * c2
    _eta20 = terms.al20(dimensionless_depth, dimensionless_height)

    _x11 = terms.x11(dimensionless_depth, dimensionless_height) * s1
    _x11_disp = terms.x11(dimensionless_depth, dimensionless_height) * c1
    _x22 = x22(dimensionless_depth, dimensionless_height)[:,None] * s2

    depth = 10
    wavenumber = dimensionless_depth/depth
    _x22 = lda.x22(dimensionless_depth,dimensionless_height) * s2

    _d_dx_u11 = - terms.u11(dimensionless_depth, dimensionless_height) * s1
    _d_dz_u11 = terms.w11(dimensionless_depth, dimensionless_height) * c1

    _d2_dx2_u11 = - terms.u11(dimensionless_depth, dimensionless_height) * c1
    _d2_dz2_u11 = terms.u11(dimensionless_depth, dimensionless_height) * c1
    _d2_dzdx_u11 = - terms.w11(dimensionless_depth, dimensionless_height) * s1

    _d_dx_u22 = - 2 * terms.u22(dimensionless_depth, dimensionless_height) * s2
    _d_dz_u22 = 2 * terms.w22(dimensionless_depth, dimensionless_height) * c2

    _u33 = terms.u33(dimensionless_depth, dimensionless_height) * c3


    mu = terms.mu(dimensionless_depth)
    ch2 = terms.ch(2,dimensionless_depth,dimensionless_height)
    disp = (9 - 10 * mu ** 2 + 9 * mu ** 4) / 16 / mu ** 4
    us = ( 1 + mu**2)/mu**2/2 * ch2


    _u3nl = (
         _disp * _x11_disp
        + _u33
        + _eta11 * _d_dz_u22
        + _x11 * _d_dx_u22
        + _x22 * _d_dx_u11
        + _eta22 * _d_dz_u11
        + _eta20 * _d_dz_u11
        + _eta11**2 * _d2_dz2_u11 / 2
        + _x11**2 * _d2_dx2_u11 / 2
        + _eta11 * _x11 * _d2_dzdx_u11
    )

    amp = terms.fourier_amplitude(_u3nl, n, 'cos')
    return - amp / n

def target_x33(dimensionless_depth, dimensionless_height):
    return terms.x33(dimensionless_depth, dimensionless_height)

def target_x31(dimensionless_depth, dimensionless_height):
    return terms.x31(dimensionless_depth, dimensionless_height)


def x33(dimensionless_depth, dimensionless_height):
    return x3n(dimensionless_depth, dimensionless_height, 3)

def eta_3n(dimensionless_depth, dimensionless_height, n):
    c2 = terms.cos(2)
    c3 = terms.cos(3)
    c1 = terms.cos(1)
    s2 = terms.sin(2)
    s1 = terms.sin(1)

    _disp = terms.dispersion(dimensionless_depth, dimensionless_height)

    _a11 = terms.a11(dimensionless_depth, dimensionless_height)
    _a22 = terms.a22(dimensionless_depth, dimensionless_height)
    _a33 = terms.a33(dimensionless_depth, dimensionless_height) * c3
    _a31 = terms.a31(dimensionless_depth, dimensionless_height) * c1

    _x11 = terms.x11(dimensionless_depth, dimensionless_height) * s1
    _x22 = terms.x22(dimensionless_depth, dimensionless_height) * s2

    _d_dx_a11 = - _a11 * s1
    _d_dx_a22 = - 2*_a22 * s2

    _d2_dx2_a11 = - _a11 * c1

    _a3nl = (
        + _a33
        + _a31
        + _x11 * _d_dx_a22
        + _x22 * _d_dx_a11
        + _x11**2 * _d2_dx2_a11 / 2
    )
    a3nl = terms.fourier_amplitude(_a3nl, n, 'cos')
    return a3nl


def eta31(dimensionless_depth, dimensionless_height):
    return eta_3n(dimensionless_depth, dimensionless_height, 1)

def eta33(dimensionless_depth, dimensionless_height):
    return eta_3n(dimensionless_depth, dimensionless_height, 3)

def target_eta31(dimensionless_depth, dimensionless_height):
    return terms.al31(dimensionless_depth, dimensionless_height)

def target_eta33(dimensionless_depth, dimensionless_height):
    return terms.al33(dimensionless_depth, dimensionless_height)

def x31(dimensionless_depth, dimensionless_height):
    return x3n(dimensionless_depth, dimensionless_height, 1)

def validation():
    terms.validate(terms.x33,x33,'x33',False)
    terms.validate(target_x31, x31, 'x31', False)

    terms.validate(target_eta31, eta31, 'eta31', False)
    terms.validate(target_eta33, eta33, 'eta33', False)


if __name__ == '__main__':
    validation()