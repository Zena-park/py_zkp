from py_ecc.fields import bn128_FQ as FQ
from py_ecc import bn128

g1 = bn128.G1
g2 = bn128.G2

mult = bn128.multiply
pairing = bn128.pairing
add = bn128.add
neg = bn128.neg

class FR(FQ):
    field_modulus = bn128.curve_order


# Multiply two polynomials
def multiply_polys(a, b):
    o = [FR(0)] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            o[i + j] += a[i] * b[j]
    return o

# Add two polynomials
def add_polys(a, b, subtract=False):
    o = [FR(0)] * max(len(a), len(b))
    for i in range(len(a)):
        o[i] += a[i]
    for i in range(len(b)):
        o[i] += b[i] * (-1 if subtract else 1) # Reuse the function structure for subtraction
    return o

def subtract_polys(a, b):
    return add_polys(a, b, subtract=True)

# Divide a/b, return quotient and remainder
def div_polys(a, b):
    o = [FR(0)] * (len(a) - len(b) + 1)
    remainder = a
    while len(remainder) >= len(b):
        if b[-1] == 0:
            raise ZeroDivisionError("Division by zero polynomial")
        leading_fac = remainder[-1] / b[-1]
        pos = len(remainder) - len(b)
        o[pos] = leading_fac
        remainder = subtract_polys(remainder, multiply_polys(b, [FR(0)] * pos + [leading_fac]))[:-1]
    return o, remainder

# Evaluate a polynomial at a point
def eval_poly(poly, x):
    return sum([poly[i] * x**i for i in range(len(poly))])

def mk_singleton(point_loc, height, total_pts):
    fac = FR(1)
    for i in range(1, total_pts + 1):
        if i != point_loc:
            fac *= FR(point_loc - i)
    o = [FR(height) / fac]
    for i in range(1, total_pts + 1):
        if i != point_loc:
            o = multiply_polys(o, [FR(-i), FR(1)])
    return o

def lagrange_interp(vec):
    o = []
    for i in range(len(vec)):
        o = add_polys(o, mk_singleton(i + 1, FR(vec[i]), len(vec)))
    for i in range(len(vec)):
        assert eval_poly(o, i + 1) == vec[i], \
            (o, eval_poly(o, i + 1), i+1)
    return o

# Multiply Vector * Matrix
def multiply_vec_matrix(vec, matrix):
    # len(vec) == len(matrix.row)
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    assert len(vec) == num_rows
    target = [FR(0)]*num_cols
    for i in range(num_rows): #loop num of rows == size of vec, 0-5
        for j in range(num_cols): #loop num of columns, 0-3
            target[j] += matrix[i][j] * vec[i]
    return target

def multiply_vec_vec(vec1, vec2):
    assert len(vec1) == len(vec2)
    target = 0
    size = len(vec1)
    for i in range(size):
        target += vec1[i]*vec2[i]
    return target

def getNumWires(Ax):
    return len(Ax)

def getNumGates(Ax):
    return len(Ax[0])

# def getFRPoly1D(poly):
#     return [ FR(num) for num in poly ]

# def getFRPoly2D(poly):
#     return [ [FR(num) for num in vec] for vec in poly ]

# def eval_2d_poly(poly, x_val):
#     o = []
#     for i in range(len(poly)):
#         ax_single = _eval_poly(poly[i], x_val)
#         o.append(ax_single)
#     return o

def ax_val(Ax, x_val):
    Ax_val = []
    for i in range(len(Ax)):
        ax_single = eval_poly(Ax[i], x_val)
        Ax_val.append(ax_single)
    return Ax_val

def bx_val(Bx, x_val):
    Bx_val = []
    for i in range(len(Bx)):
        bx_single = eval_poly(Bx[i], x_val)
        Bx_val.append(bx_single)
    return Bx_val

def cx_val(Cx, x_val):
    Cx_val = []
    for i in range(len(Cx)):
        cx_single = eval_poly(Cx[i], x_val)
        Cx_val.append(cx_single)
    return Cx_val

def zx_val(Zx, x_val):
    return eval_poly(Zx, x_val)

def hx_val(Hx, x_val):
    return eval_poly(Hx, x_val)

# (Ax.R * Bx.R - Cx.R) / Zx = Hx .... r
def hxr(Ax, Bx, Cx, Zx, R):
    Rax = multiply_vec_matrix(R, Ax)
    Rbx = multiply_vec_matrix(R, Bx)
    Rcx = multiply_vec_matrix(R, Cx)
    Px = subtract_polys(multiply_polys(Rax, Rbx), Rcx)

    q, r = div_polys(Px, Zx)
    Hx = q

    return Hx, r

