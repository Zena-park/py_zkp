from poly_utils import (
    multiply_polys,
    add_polys,
    subtract_polys,
    div_polys,
    lagrange_interp
)

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def r1cs_to_qap(A, B, C):
    A, B, C = transpose(A), transpose(B), transpose(C)
    new_A = [lagrange_interp([a for a in row]) for row in A]
    new_B = [lagrange_interp([b for b in row]) for row in B]
    new_C = [lagrange_interp([c for c in row]) for row in C]
    Z = [1]
    for i in range(1, len(A[0]) + 1):
        Z = multiply_polys(Z, [-i, 1])
    return (new_A, new_B, new_C, Z)

def create_solution_polynomials(r, new_A, new_B, new_C):
    Apoly = []
    for rval, a in zip(r, new_A):
        Apoly = add_polys(Apoly, multiply_polys([rval], a))

    Bpoly = []
    for rval, b in zip(r, new_B):
        Bpoly = add_polys(Bpoly, multiply_polys([rval], b))

    Cpoly = []
    for rval, c in zip(r, new_C):
        Cpoly = add_polys(Cpoly, multiply_polys([rval], c))

    sol = subtract_polys(multiply_polys(Apoly, Bpoly), Cpoly)
    #return Apoly, Bpoly, Cpoly, o
    return sol

def create_divisor_polynomial(sol, Z):
    quot, rem = div_polys(sol, Z)
    return quot