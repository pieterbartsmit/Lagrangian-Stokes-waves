import sympy as sp

mu = sp.symbols('mu')
one = sp.sympify(1)
zero = sp.sympify(0)
def terms():


    Ia = (1 + 6*mu**2 + mu**4 ) / (1+3*mu**2)
    Ib = (1 - mu**4) / (1+3*mu**2)
    IIa = (1 + 6*mu**2 + mu**4 )
    IIb = (1 - mu**4)
    IIIa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)**2
    IIIb = (1 - mu**2)/(1+mu**2)
    Oa = (1 + 6*mu**2 + mu**4 ) / (1+mu**2)
    Ob = (1 - mu**2)


    a44e = [
            (1 - mu ** 4) * (-27 + 30 * mu ** 2 - 11 * mu ** 4) / mu ** 8 / 384,
            + (-mu ** 12 - 32 * mu ** 10 - 97 * mu ** 8 + 280 * mu ** 6 + 141 * mu ** 4 + 2376 * mu ** 2 + 405) / mu ** 10 / 1536 / (
                    5 + mu ** 2)
    ]


    a42e= [
            (89 * mu ** 10 - 681 * mu ** 8 + 262 * mu ** 6 + 762 * mu ** 4 - 351 * mu ** 2 - 81) / (768 * mu ** 10),
            + (1 + 6 * mu ** 2 + mu ** 4) * (27 - 12 * mu ** 2 + 5 * mu ** 4) / mu ** 8 / 192
    ]


    x11a33 = [
        (
                (1 + mu ** 2) / 2 * 3*(1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 6
                +(1 - mu**4) / (1+3*mu**2)/2 * 3*(27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 8 / 192
         ),
        (1 + 6*mu**2 + mu**4 ) / (1+3*mu**2)/2 *  3*(27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 8 / 192
    ]


    x11a31=  (
            (
            +(-21 + 19 * mu ** 2 - 14 * mu ** 4) / 32 / mu ** 6 * (1+mu**2)/2
            + (9 + 23 * mu ** 2 - 12 * mu ** 4) / 32 / mu ** 6 * Ib/2),
            +  (9 + 23 * mu ** 2 - 12 * mu ** 4) / 32 / mu ** 6 *Ia/2
    )


    x22a22 = [ -2*(1-mu**2)/4/mu**2 * (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 4,
            + 6 / 8 * (1 / mu ** 4 - 1)*(1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 4 * IIIa/2
        ]

    x33a11 =  (
        (
                -(38 * mu ** 4 - 68 * mu ** 2 + 30) / (96 * mu ** 6)* (1+mu**2)/2
                + Ib/2 *  (-39 * mu ** 6 + 53 * mu ** 4 - 5 * mu ** 2 - 9) / (64 * mu ** 8)
         ),
        -  (-39 * mu ** 6 + 53 * mu ** 4 - 5 * mu ** 2 - 9) / (64 * mu ** 8) * Ia/2
    )


    x31a11 = (
        (
                -(10 * mu ** 4 - 12 * mu ** 2 + 18) / (32 * mu ** 6) * (1+mu**2)/2
                + Ib / 2*(42 * mu ** 4 - 76 * mu ** 2 - 30) / (32 * mu ** 6)
        )
        ,
        -(42 * mu ** 4 - 76 * mu ** 2 - 30) / (32 * mu ** 6)* Ia/2
    )

    x11a22 = [-2 * (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 6 * Ob  / 2,
              -2 * (1 + mu ** 2) * (3 - mu ** 2) / 8 / mu ** 6 * Oa  / 4]


    x11x22a11 =  [ (1 - mu ** 2) * (1+mu**2) / 8 / mu ** 4,- 3 * (1  -  mu ** 4) /mu**6/ 8  * Oa/4]
    x11x11x11a11 = [
        - 1 / mu ** 4 / 6 * 1 / 4 * IIb,
        - 1 / mu ** 4 / 6 * 1 / 8 * IIa
        ]

    T =[
        a44e, #T1a
        a42e, #T1b
        x11a33,
        x11a31,
        x22a22,
        x33a11,
        x31a11,
        x11a22,
        x11x22a11,
        x11x11x11a11
        # [3*(1 - mu ** 2) * (-9 + 6 * mu ** 2) / 96 / mu ** 6,
        #  3*(27 + 69 * mu ** 2 + 9 * mu ** 6 - 33 * mu ** 4) / mu ** 8 / 192]
    ]

    return T
    # _sh2terms = 0*mu
    # _sh4terms = 0 * mu
    # for _T in T:
    #     _sh2terms += _T[0]
    #     _sh4terms += _T[1]
    #
    #
    # sh2terms = sp.together(_sh2terms)
    # sh4terms = sp.together(_sh4terms)
    # print(sh2terms.args)

def na40():
    L40 = [
        zero, #Tonea
        zero, #Toneb
        zero, #T2a
        one/2, #T2b
        one/2, #T3
        zero, #T4a
        one/2, #T4b
        -one/4, #T5
        one/4, #T6
        sp.sympify(3)/8 #T7
    ]

    T40 = L40

    t_res_sh2 = combine(T40,0)
    t_res_sh4 = combine(T40, 1)
    l_res_sh2 = combine(L40,0)
    l_res_sh4 = combine(L40, 1)

    return t_res_sh2,t_res_sh4,l_res_sh2,l_res_sh4

def na42():
    T42 = [
        zero, #Tonea
        one, #Toneb
        one/2, #T2a
        -one/2, #T2b
        zero, #T3
        one/2, #T4a
        -one/2, #T4b
        one/2, #T5
        zero, #T6
        -one/2 #T7
    ]

    L42 = [
        zero, #Tonea
        zero, #Toneb
        one/2, #T2a
        -one/2, #T2b
        zero, #T3
        one/2, #T4a
        -one/2, #T4b
        one/2, #T5
        zero, #T6
        -one/2 #T7
    ]

    t_res_sh2 = combine(T42,0)
    t_res_sh4 = combine(T42, 1)
    l_res_sh2 = combine(L42,0)
    l_res_sh4 = combine(L42, 1)

    return t_res_sh2,t_res_sh4,l_res_sh2,l_res_sh4


def na44():
    L44 = [
        zero, #T1a
        zero, #T1b
        -one / 2,  # T2a
        zero,  # T2b
        -one / 2,  # T3
        -one / 2,  # T4a
        zero,  # T4b
        -one / 4,  # T5
        -one / 4,  # T6
        one / 8  # T7
    ]

    T44 = [
        one, #T1a
        zero, #T1b
        -one/2, #T2a
        zero, #T2b
        -one/2, #T3
        -one/2, #T4a
        zero, #T4b
        -one/4, #T5
        -one/4, #T6
        one/8 #T7
    ]

    t_res_sh2 = combine(T44, 0)
    t_res_sh4 = combine(T44, 1)
    l_res_sh2 = combine(L44, 0)
    l_res_sh4 = combine(L44, 1)

    return t_res_sh2, t_res_sh4, l_res_sh2, l_res_sh4

def combine(  coef, index ):
    _terms = terms()
    _result = 0*mu
    for _T,_coef in zip(_terms,coef):
        _result += _T[index] * _coef

    _result = sp.together(_result)

    expr = sp.expand(_result.args[-1]) * sp.prod(_result.args[:-1])

    return expr



if __name__ == '__main__':
    T40_sh2,T40_sh4,L40_sh2,L40_sh4 = na40()
    T42_sh2, T42_sh4, L42_sh2, L42_sh4 = na42()
    T44_sh2, T44_sh4, L44_sh2, L44_sh4 = na44()

    print(sp.div((mu**4 + 6*mu**2 + 1)*(-294.0*mu**8 + 292.0*mu**6 + 328.0*mu**4 - 420.0*mu**2 - 162.0),(mu**2 + 1)))
    #print((-108*mu**8 + 48*mu**6 + 40*mu**4 - 176*mu**2 - 60)/2)
    #print((-108 * mu ** 8 + 48 * mu ** 6 + 40 * mu ** 4 - 176 * mu ** 2 - 60) / 4)
    #print(768/4)
    print((474*mu**10 - 526*mu**8 - 540*mu**6 + 580*mu**4 + 66*mu**2 - 54)/2)
    #print(L42_sh2)
    #print(L42_sh4 )

