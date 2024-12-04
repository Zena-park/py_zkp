from py_zkp.plonk.setup import get_setup_values, to_poly, to_vanishing_poly
# import numpy as np
import galois

#----------------------------
# public
input = [2, 3]
p = 241
#----------------------------
# witness
code = """
def qeval(x, y):
    g0 = x * x
    g1 = x * x
    g2 = y * y
    g3 = g0 * 2
    g4 = g1 * g2
    g5 = g3 - g4
    return g5 + 3
"""

(ql, qr, qm, qo, qc, a, b, c, sigma1, sigma2, sigma3, omega, roots) = get_setup_values(code, input)
print('ql', ql)
print('qr', qr)
print('qm', qm)
print('qo', qo)
print('qc', qc)
print('a', a)
print('b', b)
print('c', c)
print('sigma1', sigma1)
print('sigma2', sigma2)
print('sigma3', sigma3)
print('omega', omega)
print('roots', roots)

#----------------
Fp = galois.GF(p)
# n = len(ql)
# n = 2**int(np.ceil(np.log2(n)))
# assert n & n - 1 == 0, "n must be a power of 2"

print(' --- Gate Polynomials  ---')
QL = to_poly(roots, ql, Fp)
QR = to_poly(roots, qr, Fp)
QM = to_poly(roots, qm, Fp)
QC = to_poly(roots, qc, Fp)
QO = to_poly(roots, qo, Fp)

print('QL', QL)
print('QR', QR)
print('QM', QM)
print('QC', QC)
print('QO', QO)

print('QL(8) ', QL(8) )

S1 = to_poly(roots, sigma1, Fp)
S2 = to_poly(roots, sigma2, Fp)
S3 = to_poly(roots, sigma3, Fp)

k1 = 2
k2 = 4
c1_roots = roots
c2_roots = c1_roots * k1
c3_roots = c1_roots * k2
I1 = to_poly(roots, c1_roots, Fp)
I2 = to_poly(roots, c2_roots, Fp)
I3 = to_poly(roots, c3_roots, Fp)

print(' --- Permutation Polynomials ---')
print('S1 ', S1 )
print('S2 ', S2 )
print('S3 ', S3 )
print('I1 ', I1 )
print('I2 ', I2 )
print('I3 ', I3 )

padding = 3
for i in range(0, len(roots)):
    s = f"i = {i:{padding}} --> {roots[i]:{padding}} "
    s += f"  I1({roots[i]:{padding}}) = {I1(roots[i]):{padding}} "
    s += f"  I2({roots[i]:{padding}}) = {I2(roots[i]):{padding}} "
    s += f"  I3({roots[i]:{padding}}) = {I3(roots[i]):{padding}} "
    s += f"  S1({roots[i]:{padding}}) = {S1(roots[i]):{padding}} "
    s += f"  S2({roots[i]:{padding}}) = {S2(roots[i]):{padding}} "
    s += f"  S3({roots[i]:{padding}}) = {S3(roots[i]):{padding}} "
    print(s)

    assert I1(roots[i]) == roots[i], f"I1({roots[i]}) != {roots[i]}"
    assert I2(roots[i]) == k1 * roots[i], f"I2({roots[i]}) != {k1 * roots[i]}"
    assert I3(roots[i]) == k2 * roots[i], f"I3({roots[i]}) != {k2 * roots[i]}"

    assert S1(roots[i]) == sigma1[i], f"S1({roots[i]}) != {sigma1[i]}"
    assert S2(roots[i]) == sigma2[i], f"S2({roots[i]}) != {sigma2[i]}"
    assert S3(roots[i]) == sigma3[i], f"S3({roots[i]}) != {sigma3[i]}"

Zh = to_vanishing_poly(roots, Fp)
for x in roots:
    assert Zh(x) == 0

print(' --- Vanishing Polynomial ---')
# Zh = x^8 + 240 which translates to x^8 - 1 (240 == -1 in Prime Field p=241)
print('Zh ', Zh )

