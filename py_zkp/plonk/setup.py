import numpy as np
import galois
import math

import ast
from scipy.interpolate import lagrange
import random

if 'arg' not in dir(ast):
    ast.arg = type(None)

def parse(code):
    return ast.parse(code).body

# Takes code of the form
# def foo(arg1, arg2 ...):
#     x = arg1 + arg2
#     y = ...
#     return x + y
# And extracts the inputs and the body, where
# it expects the body to be a sequence of
# variable assignments (variables are immutable;
# can only be set once) and a return statement at the end
def extract_inputs_and_body(code):
    o = []
    if len(code) != 1 or not isinstance(code[0], ast.FunctionDef):
        raise Exception("Expecting function declaration")
    # Gather the list of input variables
    inputs = []
    for arg in code[0].args.args:
        if isinstance(arg, ast.arg):
            assert isinstance(arg.arg, str)
            inputs.append(arg.arg)
        elif isinstance(arg, ast.Name):
            inputs.append(arg.id)
        else:
            raise Exception("Invalid arg: %r" % ast.dump(arg))
    # Gather the body
    body = []
    returned = False
    for c in code[0].body:
        if not isinstance(c, (ast.Assign, ast.Assert, ast.Return)):
            raise Exception("Expected variable assignment or return")
        if returned:
            raise Exception("Cannot do stuff after a return statement")
        if isinstance(c, ast.Return):
            returned = True
        body.append(c)
    return inputs, body

# Convert a body with potentially complex expressions into
# simple expressions of the form x = y or x = y * z
def flatten_body(body):
    o = []
    for c in body:
        o.extend(flatten_stmt(c))
    return o

# Generate a dummy variable
next_symbol = [0]
def mksymbol():
    next_symbol[0] += 1
    return 'sym_'+str(next_symbol[0])

def initialize_symbol():
    next_symbol[0] = 0

# Get the value of a node
def get_value(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return node.value
    else:
        raise Exception("Assert comparison must be between variables or constants")

# "Flatten" a single statement into a list of simple statements.
# First extract the target variable, then flatten the expression
def flatten_stmt(stmt):
    # Get target variable
    if isinstance(stmt, ast.Assign):
        # print("flatten_stmt stmt.targets ", stmt.targets[0].id)
        # print("flatten_stmt stmt.type ", stmt.type[0])
        # print("flatten_stmt stmt.value ", stmt.value)

        assert len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
        target = stmt.targets[0].id
    elif isinstance(stmt, ast.Assert):
        assert isinstance(stmt.test, ast.Compare) and len(stmt.test.ops) == 1 and isinstance(stmt.test.ops[0], ast.Eq), "Only '==' comparison is allowed in assert statements"
        left = get_value(stmt.test.left)
        right = get_value(stmt.test.comparators[0])
        # 두 값이 모두 상수인 경우 미리 검사
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if left != right:
                raise AssertionError(f"Assert condition not satisfied: {left} != {right}")
        return [['assert', left, right]] #@Todo: assert -> '==' ?
    elif isinstance(stmt, ast.Return): #@Todo: 고정된 ~out 수정
        target = '~out'
    # Get inner content
    return flatten_expr(target, stmt.value)

# Main method for flattening an expression
def flatten_expr(target, expr):
    # x = y
    if isinstance(expr, ast.Constant):
        return [['set', target, expr.id]]
    # x = 5
    elif isinstance(expr, ast.Constant):
        return [['set', target, expr.n]]
    # x = y (op) z
    # Or, for that matter, x = y (op) 5
    # 각 연산은 flatcode 1개를 생성
    # 제곱연산은 제곱한 수만큼 flatcode를 생성
    elif isinstance(expr, ast.BinOp):
        if isinstance(expr.op, ast.Add):
            op = '+'
        elif isinstance(expr.op, ast.Mult):
            op = '*'
        elif isinstance(expr.op, ast.Sub):
            op = '-'
        elif isinstance(expr.op, ast.Div):
            op = '/'
        # Exponentiation gets compiled to repeat multiplication,
        # requires constant exponent
        elif isinstance(expr.op, ast.Pow):
            assert isinstance(expr.right, ast.Constant)
            if expr.right.value == 0:
                return [['set', target, 1]]
            elif expr.right.value == 1:
                return flatten_expr(target, expr.left)
            else: # This could be made more efficient via square-and-multiply but oh well
                if isinstance(expr.left, (ast.Name, ast.Constant)):
                    nxt = base = expr.left.id if isinstance(expr.left, ast.Name) else expr.left.n
                    o = []
                else:
                    nxt = base = mksymbol()
                    o = flatten_expr(base, expr.left)
                for i in range(1, expr.right.value):
                    latest = nxt
                    nxt = target if i == expr.right.value - 1 else mksymbol()
                    o.append(['*', nxt, latest, base])
                return o
        elif isinstance(expr, ast.Compare):
            raise Exception("Comparisons are only allowed in assert statements")
        else:
            raise Exception("Bad operation: %r" % ast.dump(expr.op))
        # If the subexpression is a variable or a number, then include it directly
        if isinstance(expr.left, (ast.Name, ast.Constant)):
            var1 = expr.left.id if isinstance(expr.left, ast.Name) else expr.left.n
            sub1 = []

        # If one of the subexpressions is itself a compound expression, recursively
        # apply this method to it using an intermediate variable
        else:
            var1 = mksymbol()
            sub1 = flatten_expr(var1, expr.left)
        # Same for right subexpression as for left subexpression
        if isinstance(expr.right, (ast.Name, ast.Constant)):
            var2 = expr.right.id if isinstance(expr.right, ast.Name) else expr.right.value
            sub2 = []
        else:
            var2 = mksymbol()
            sub2 = flatten_expr(var2, expr.right)
        # Last expression represents the assignment; sub1 and sub2 represent the
        # processing for the subexpression if any
        return sub1 + sub2 + [[op, target, var1, var2]]
    else:
        raise Exception("Unexpected statement value: %r" % ast.dump(expr.value))

# Adds a variable or number into one of the vectors; if it's a variable
# then the slot associated with that variable is set to 1, and if it's
# a number then the slot associated with 1 gets set to that number
def insert_var(arr, varz, var, used, reverse=False):
    if isinstance(var, str):
        if var not in used:
            raise Exception("Using a variable before it is set!")
        arr[varz.index(var)] += (-1 if reverse else 1)
    elif isinstance(var, int):
        arr[0] += var * (-1 if reverse else 1)

# Maps input, output and intermediate variables to indices
def get_var_placement(inputs, flatcode):    #@Todo: 고정된 ~out 수정
    used_vars = ['~one'] + inputs + ['~out']
    for c in flatcode:
        if c[0] != 'assert' and c[1] not in used_vars:
            used_vars.append(c[1])
    return used_vars

# Convert the flattened code generated above into a rank-1 constraint system
def flatcode_to_r1cs(inputs, flatcode):
    varz = get_var_placement(inputs, flatcode)
    A, B, C = [], [], []
    used = {i: True for i in inputs}
    # 일단 파라미터는 무조건 used로 바꾸고.
    for x in flatcode:
    # 모든 flatcode를 대상으로 수행. flatcode의 갯수만큼 행의 row갯수(컬럼 높이)가 결정된다.
        a, b, c = [0] * len(varz), [0] * len(varz), [0] * len(varz)
        # assert는 이미 used된 변수를 비교하기위해 사용
        if x[0] != 'assert':
            if x[1] in used:
                raise Exception("Variable already used: %r" % x[1])
            used[x[1]] = True
        # 타겟 자리도 used로 바꾸고 시작
        if x[0] == 'set':
            # a의 타겟 자리에 1을 더하고, 우항(value) 자리를 -1로 바꾼다
            a[varz.index(x[1])] += 1
            insert_var(a, varz, x[2], used, reverse=True)
            b[0] = 1
        elif x[0] == '+' or x[0] == '-':
            # c행렬의 타겟 자리를 1로 바꾸고, left는 a자리에서 1, right는 a자리에서 1(-1 if "-"라면)로 만듬
            # 더하기이기 때문에 이렇게 하면 항상 양쪽 식이 만족함
            c[varz.index(x[1])] = 1
            insert_var(a, varz, x[2], used)
            insert_var(a, varz, x[3], used, reverse=(x[0] == '-'))
            b[0] = 1
        elif x[0] == '*':
            # c행렬의 타겟 자리는 1로 만들고
            # a행렬의 left 자리를 1로, b행렬의 right 자리를 1로 만든다
            c[varz.index(x[1])] = 1
            insert_var(a, varz, x[2], used)
            insert_var(b, varz, x[3], used)
        elif x[0] == '/':
            # a행렬의 타겟 자리를 1로 만들고
            # c행렬의 left자리를 1로, b행렬의 right자리를 1로 바꾼
            # why? a = b / c --> b = a * c가 되기 때문
            insert_var(c, varz, x[2], used)
            a[varz.index(x[1])] = 1
            insert_var(b, varz, x[3], used)
        elif x[0] == 'assert':
            # assert 문 처리
            insert_var(a, varz, x[1], used)
            b[0] = 1
            insert_var(c, varz, x[2], used)
        A.append(a)
        B.append(b)
        C.append(c)
    return A, B, C

# Get a variable or number given an existing input vector
def grab_var(varz, assignment, var):
    if isinstance(var, str):
        return assignment[varz.index(var)]
    elif isinstance(var, int):
        return var
    else:
        raise Exception("What kind of expression is this? %r" % var)

# Goes through flattened code and completes the input vector
# 특정한 인풋값에 의해서 인풋백터를 완성함
def assign_variables(inputs, input_vars, flatcode):
    varz = get_var_placement(inputs, flatcode)
    assignment = [0] * len(varz)
    assignment[0] = 1
    for i, inp in enumerate(input_vars):
        assignment[i + 1] = inp
    for x in flatcode:
        if x[0] == 'set':
            assignment[varz.index(x[1])] = grab_var(varz, assignment, x[2])
        elif x[0] == '+':
            assignment[varz.index(x[1])] = grab_var(varz, assignment, x[2]) + grab_var(varz, assignment, x[3])
        elif x[0] == '-':
            assignment[varz.index(x[1])] = grab_var(varz, assignment, x[2]) - grab_var(varz, assignment, x[3])
        elif x[0] == '*':
            assignment[varz.index(x[1])] = grab_var(varz, assignment, x[2]) * grab_var(varz, assignment, x[3])
        elif x[0] == '/':
            assignment[varz.index(x[1])] = grab_var(varz, assignment, x[2]) / grab_var(varz, assignment, x[3])
    return assignment


def code_to_r1cs_with_inputs(code, input_vars):
    inputs, body = extract_inputs_and_body(parse(code))
    print('Inputs')
    print(inputs)
    print('Body')
    print(body)
    flatcode = flatten_body(body)
    print('Flatcode')
    print(flatcode)
    print('Input var assignment')
    print(get_var_placement(inputs, flatcode))
    # A, B, C = flatcode_to_r1cs(inputs, flatcode)
    # r = assign_variables(inputs, input_vars, flatcode)
    # return r, A, B, C
    return
#=======================
def gcd(a, b):
    while b > 0:
        a, b = b, a % b
    return a

def primitive_root_of_unity_by_modulo_general(p):
    r = set(range(1, p))
    res = []
    for i in r:
        if gcd(i, p) == 1:
            for j in r:
                if pow(i,j,p) == 1:
                    if j < p-1 : break
                    elif j == p-1:
                        res.append(i)
                        # print(f"a={i}, ep(a)={j}, p={p}")
    return res

def primitive_root_of_unity_by_modulo_first(p):
    r = set(range(1, p))
    one_primitive = 0
    for i in r:
        if gcd(i, p) == 1:
            for j in r:
                if pow(i,j,p) == 1:
                    if j < p-1 : break
                    elif j == p-1:
                        one_primitive = i
                        break
        if one_primitive > 0: break
    return one_primitive

def primitive_root_of_unity_by_modulo(p):
    res = []
    f_primitive = primitive_root_of_unity_by_modulo_first(p)
    if f_primitive == 0: return res
    r = set(range(1, p))
    for i in r:
        if gcd(i, p-1) == 1:
            res.append(pow(f_primitive,i,p))
    return res


def primitive_nth_root_of_unity_by_modulo(n,p):
    r = set(range(2, p))
    res = 0
    for i in r:
        if gcd(i, p) == 1:
            if pow(i,n,p) == 1:
                return i
    return res


def get_root(w,n,p):
    res = []
    for i in range(n):
        res.append(pow(w,i,p))
    return res
# https://github.com/keon/algorithms/blob/master/algorithms/maths/find_primitive_root_simple.py

# For positive integer n and given integer a that satisfies gcd(a, n) = 1,
# the order of a modulo n is the smallest positive integer k that satisfies
# pow (a, k) % n = 1. In other words, (a^k) ≡ 1 (mod n).
# Order of certain number may or may not be exist. If so, return -1.
def find_order(a, n):
    # Find order for positive integer n and given integer a that satisfies gcd(a, n) = 1.
    # Time complexity O(nlog(n))
    if (a == 1) & (n == 1):
        # Exception Handeling : 1 is the order of of 1
        return 1
    if gcd(a, n) != 1:
        print ("a and n should be relative prime!")
        return -1
    for i in range(1, n):
        if pow(a, i) % n == 1:
            return i
    return -1

# Euler's totient function, also known as phi-function ϕ(n),
# counts the number of integers between 1 and n inclusive,
# which are coprime to n.
# (Two numbers are coprime if their greatest common divisor (GCD) equals 1).
# Code from /algorithms/maths/euler_totient.py, written by 'goswami-rahul'
def euler_totient(n):
    # Euler's totient function or Phi function.
    # Time Complexity: O(sqrt(n)).
    result = n
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            while n % i == 0:
                n //= i
            result -= result // i
    if n > 1:
        result -= result // n
    return result

# For positive integer n and given integer a that satisfies gcd(a, n) = 1,
# a is the primitive root of n, if a's order k for n satisfies k = ϕ(n).
# Primitive roots of certain number may or may not exist.
# If so, return empty list.
def find_primitive_root(n):
    if n == 1:
        # Exception Handeling : 0 is the only primitive root of 1
        return [0]
    phi = euler_totient(n)
    p_root_list = []
    # It will return every primitive roots of n.
    for i in range (1, n):
        #To have order, a and n must be relative prime with each other.
        if math.gcd(i, n) == 1:
            order = find_order(i, n)
            if order == phi:
                p_root_list.append(i)
    return p_root_list

# https://asecuritysite.com/principles_pub/g_pick?val1=41
def getG(p, maxIndex):
    r = set(range(1, p))
    print(f"r = {r}")
    res = []
    for i in r:
        gen = set()
        for x in r:
            gen.add(pow(i,x,p))

        # print(f"i= {i}, gen = {gen}")
        if gen == r:
            res.append(i)
            if (len(res)>maxIndex): break
    return res

def exec_gate_code(code, a, b):
    res = 0
    left = a
    right = b
    if left == '*': left = code[2]
    if right == '*': right = code[3]
    if left != '*' and right != '*':
        left = int(left)
        right = int(right)
        if code[0] == '+': res = left + right
        elif code[0] == '-': res = left - right
        elif code[0] == '*': res = left * right
        elif code[0] == '/': res = left / right
    return res

def input_v_metrics(inputs, input, ql, qr, qm, qo, qc, vl, vr, vo, input_indexes):
    pi = [0 for i in range(len(ql)+len(inputs))]
    input_vl = ['-' for i in range(len(inputs))]
    input_vr = ['-' for i in range(len(inputs))]
    input_vo = ['-' for i in range(len(inputs))]
    for i, value in enumerate(inputs):
        add_input_constraint(ql, qr, qm, qo, qc)
        input_vl[i] = find_value_index(value, input_indexes)
        pi[i] = (input[i])
    input_vl.extend(vl)
    input_vr.extend(vr)
    input_vo.extend(vo)
    return (input_vl, input_vr, input_vo, pi)

def convert_star(value):
    if value == '*': value=0
    return value

def exec_gate_unknown_c(q, a, b):
    a = convert_star(a)
    b = convert_star(b)
    c = (q[0]*a + q[1]*b + q[2]*a*b + q[4]) / q[3] * -1
    return c

def exec_gate_unknown_a(q, b, c):
    c = convert_star(c)
    b = convert_star(b)
    a = (q[1]*b + q[3]*c + q[4]) / (q[0] + q[2]*b ) * -1
    return a

def exec_gate_unknown_b(q, a, c):
    c = convert_star(c)
    a = convert_star(a)
    b = (q[0]*a + q[3]*c + q[4]) / (q[1] + q[2]*a ) * -1
    return b

def find_value(indexes, values, input_values):
    for i, value in enumerate(values):
        if indexes[i] != '-' and value == '-' and indexes[i] < len(input_values):
            values[i] = input_values[indexes[i]]
        elif indexes[i] == '-':
            values[i] = '*'
    return values

def find_value_index(key, input_indexes, input_values):
    find = '-'
    for i, value in enumerate(input_indexes):
        if key != '-'  and key == value:
            find = i
    return find

def trace_matrix(flatcode, q_zipped, v_zipped, v_values, input_indexes, input_values):
    for i, q in enumerate(q_zipped):
        abc = find_value(v_zipped[i], v_values[i], input_values)
        a = abc[0]
        b = abc[1]
        c = abc[2]
        if (a == '*' or b == '*') and c == '-':
            c = exec_gate_code(flatcode[i], a, b)
            v_values[i][2] = c
            input_values[v_zipped[i][2]] = c

        if a != '-' and b != '-' and c =='-':
            abc[2] = exec_gate_unknown_c(q ,a, b)
            input_values[v_zipped[i][2]] = abc[2]
        elif a == '-' and b != '-' and c !='-':
            abc[2] = exec_gate_unknown_a(q ,a, b)
            input_values[v_zipped[i][2]] = abc[2]
        elif a != '-' and b == '-' and c !='-':
            abc[2] = exec_gate_unknown_b(q ,a, b)
            input_values[v_zipped[i][2]] = abc[2]

        v_values[i] = abc
    return v_values

def set_values(inputs, input, input_indexes, input_values):

    for i, key in enumerate(input_indexes):
        for j, target in  enumerate(inputs):
            if key == target:
                input_values[i] = input[j]
    return input_values

def get_v_metrics(gates, input_indexes):
    vl = []
    vr = []
    vo = []
    for gate in enumerate(gates):
        (vl, vr, vo) = get_v_vector(gate, input_indexes, vl, vr, vo)
    return  (vl, vr, vo)

def get_v_vector(gate, input_indexes, vl, vr, vo):
    try:
        vl.append(input_indexes.index(gate[1][2]))
    except ValueError:
        vl.append('-')

    try:
        vr.append(input_indexes.index(gate[1][3]))
    except ValueError:
        vr.append('-')

    try:
        vo.append(input_indexes.index(gate[1][1]))
    except ValueError:
        vo.append('-')

    return (vl, vr, vo)

def get_q_metrics(gates):
    ql = []
    qr = []
    qm = []
    qo = []
    qc = []
    for gate in enumerate(gates):
        (ql, qr, qm, qo, qc) = get_q_vector(gate, ql, qr, qm, qo, qc)
    return  (ql, qr, qm, qo, qc)

# / operator and - operator are not implemented
def get_q_vector(gate, ql, qr, qm, qo, qc):
    if gate[1][0] == '+':
        if isinstance(gate[1][2], int):
            (ql, qr, qm, qo, qc) = add_constant_constraint(ql, qr, qm, qo, qc, gate[1][2])
        elif isinstance(gate[1][3], int):
            (ql, qr, qm, qo, qc) = add_constant_constraint(ql, qr, qm, qo, qc, gate[1][3])
        else:
             (ql, qr, qm, qo, qc) = add_add_constarint(ql, qr, qm, qo, qc)
    elif gate[1][0] == '-':
        if isinstance(gate[1][2], int):
            (ql, qr, qm, qo, qc) = add_constant_constraint(ql, qr, qm, qo, qc, (-1)*gate[1][2])
        elif isinstance(gate[1][3], int):
            (ql, qr, qm, qo, qc) = add_constant_constraint(ql, qr, qm, qo, qc, (-1)*gate[1][3])
        else:
            (ql, qr, qm, qo, qc) = add_minus_constarint(ql, qr, qm, qo, qc)
    elif gate[1][0] == '*':
        if isinstance(gate[1][2], int):
            (ql, qr, qm, qo, qc) = add_mul_rconstant_constarint(ql, qr, qm, qo, qc, gate[1][2])
        elif isinstance(gate[1][3], int):
            (ql, qr, qm, qo, qc) = add_mul_lconstant_constarint(ql, qr, qm, qo, qc, gate[1][3])
        else:
            (ql, qr, qm, qo, qc) = add_mul_constarint(ql, qr, qm, qo, qc)
    # elif gate[1][0] == '/':
    #     (ql, qr, qm, qo, qc) = add_mul_constarint(ql, qr, qm, qo, qc)
    # elif gate[1][0] == 'set':
    #     (ql, qr, qm, qo, qc) = add_constant_constraint(ql, qr, qm, qo, qc)
    return (ql, qr, qm, qo, qc)

def add_mul_constarint(Ql, Qr, Qm, Qo, Qc):

    Ql.append(0)
    Qr.append(0)
    Qm.append(1)
    Qo.append(-1)
    Qc.append(0)

    return (Ql, Qr, Qm, Qo, Qc)

def add_mul_lconstant_constarint(Ql, Qr, Qm, Qo, Qc, const):

    Ql.append(const)
    Qr.append(0)
    Qm.append(0)
    Qo.append(-1)
    Qc.append(0)

    return (Ql, Qr, Qm, Qo, Qc)


def add_mul_rconstant_constarint(Ql, Qr, Qm, Qo, Qc, const):

    Ql.append(0)
    Qr.append(const)
    Qm.append(0)
    Qo.append(-1)
    Qc.append(0)

    return (Ql, Qr, Qm, Qo, Qc)

def add_add_constarint(Ql, Qr, Qm, Qo, Qc):
    Ql.append(1)
    Qr.append(1)
    Qm.append(0)
    Qo.append(-1)
    Qc.append(0)
    return (Ql, Qr, Qm, Qo, Qc)

def add_minus_constarint(Ql, Qr, Qm, Qo, Qc):
    Ql.append(1)
    Qr.append(-1)
    Qm.append(0)
    Qo.append(-1)
    Qc.append(0)
    return (Ql, Qr, Qm, Qo, Qc)

def add_constant_constraint(Ql, Qr, Qm, Qo, Qc, const):
    Ql.append(1)
    Qr.append(0)
    Qm.append(0)
    Qo.append(-1)
    Qc.append(const)
    return (Ql, Qr, Qm, Qo, Qc)

def set_constant_constraint(Ql, Qr, Qm, Qo, Qc, const):
    Ql.append(0)
    Qr.append(0)
    Qm.append(0)
    Qo.append(-1)
    Qc.append(const)
    return (Ql, Qr, Qm, Qo, Qc)

def add_input_constraint(Ql, Qr, Qm, Qo, Qc):
    Ql.insert(0, -1)
    Qr.insert(0,0)
    Qm.insert(0,0)
    Qo.insert(0,0)
    Qc.insert(0,0)
    return (Ql, Qr, Qm, Qo, Qc)

def constraint_polynomial(Qli, Qri, Qmi, Qoi, Qci, ai, bi, ci):
    return(Qli*ai + Qri*bi + Qoi*ci + Qmi*ai*bi + Qci == 0)

def validate_native(Ql, Qr, Qm, Qo, Qc, a, b, c):
    for Qli,Qri,Qmi,Qoi,Qci,ai,bi,ci in zip (Ql,Qr,Qm,Qo,Qc,a,b,c):
        if (constraint_polynomial(Qli,Qri,Qmi,Qoi,Qci,ai,bi,ci) == False):
            return(False)
    return(True)

def validNumber(a):
    if a == '-' or a == '*': return 0
    return a

def gen_abc(v):
    a = []
    b = []
    c = []
    for i, e in enumerate(v):
        a.append(validNumber(e[0]))
        b.append(validNumber(e[1]))
        c.append(validNumber(e[2]))
    return a, b, c

def copy_constraint_simple(eval_domain, Xcoef, Ycoef, v1, v2):
    Px = [1]
    Y = []
    rlc = []
    x = []

    for i in range(0, len(eval_domain)):
        x.append(polynomial_eval(Xcoef, eval_domain[i]))
        Y.append(polynomial_eval(Ycoef, x[i]))

        rlc.append(v1 + x[i] + v2 * Y[i])
        Px.append(Px[i] * (v1 + x[i] + v2 * Y[i]))

    return (x, Y, Px, rlc)


def find_permutation(copies, eval_domain):

    # print('find_permutation', copies, eval_domain)
    # for i, e in enumerate(copies):
    #     print('copies', i, e)

    # for i, e in enumerate(eval_domain):
    #     print('eval_domain', i, e)

    perm = lagrange(eval_domain, copies)

    # print('find_permutation perm ', perm)

    perm = [float(x) for x in reversed(perm.coefficients)]
    return perm

def polynomial_eval(coef, x):
    res = []
    power = 1
    for i in coef:
        res.append(i * power)
        power = power * x
    return round(sum(res))


def polynomial_division(poly, q):
    result = []
    for i in reversed(range(0, len(poly))):
        factor = poly[i] / q[-1]
        result.append(factor)
        poly[i] = poly[i] - (factor * q[-1])
        poly[i - 1] = poly[i - 1] - (q[0] * factor)
        if sum(poly) == 0:
            return (True, result)
    return (False, result)


def gen_poly(y):
    x = range(0, len(y))
    poly = lagrange(x, y)
    poly = [float(x) for x in reversed(poly.coefficients)]
    return poly

def pad_array( a, n ):
    return a + [ 0 ]*(n - len (a))

def isPowerOf2(n):
    return (n&(n-1))==0

def isSquare(n):
    return int(n ** 0.5) ** 2 == n

def minPowerOf2(n):
    if isPowerOf2(n) == False:
        for i in range(31):
            if 2**i > n: return 2**i
    return n

def set_sigma_index(n, vl, vr, vo, input_index):
    sigma_ai = [ i for i in range(0, n)]
    sigma_bi = [ i for i in range(n, 2*n)]
    sigma_ci = [ i for i in range(2*n, 3*n)]

    for i, value in enumerate(vl):
        if not value in input_index:
            findI = find_sigma_index(value, vr, n)
            if findI == 0:
                findI = find_sigma_index(value, vo, 2*n)
                if findI != 0:
                    sigma_ai[i] = findI
            else:
                sigma_bi[i] = findI

    for i, value in enumerate(vr):
        if not value in input_index:
            findI = find_sigma_index(value, vl, 0)
            if findI == 0:
                findI = find_sigma_index(value, vo, 2*n)
                if findI != 0:
                    sigma_bi[i] = findI
            else:
                sigma_bi[i] = findI

    for i, value in enumerate(vo):
        findI = find_sigma_index(value, vl, 0)
        if findI == 0:
            findI = find_sigma_index(value, vr, n)
            if findI != 0:
                sigma_ci[i] = findI
        else:
            sigma_ci[i] = findI

    return (sigma_ai, sigma_bi, sigma_ci)

def find_sigma_index(value, target_vector, padding):
    for i, e in enumerate(target_vector):
        if e == value: return i+padding
    return 0

def find_input_index(inputs, input_indexes):
    i_index = []
    for i, ei in enumerate(inputs):
        for j, ej in enumerate(input_indexes):
            if ei == ej: i_index.append(j)
    return i_index

def to_galois_array(vector, field):
    # normalize to positive values
    a = [x % field.order for x in vector]
    return field(a)

def to_poly(x, v, field):
    assert len(x) == len(v)
    y = to_galois_array(v, field) if type(v) == list else v
    return galois.lagrange_poly(x, y)

def to_vanishing_poly(roots, field):
    # Z^n - 1 = (Z - 1)(Z - w)(Z - w^2)...(Z - w^(n-1))
    return galois.Poly.Degrees([len(roots), 0], coeffs=[1, -1], field=field)

def adjust_witness_size(ql, qr, qm, qo, qc, a, b , c, n):
    if len(ql) != n :
        ql = pad_array(ql, n)
        qr = pad_array(qr, n)
        qm = pad_array(qm, n)
        qc = pad_array(qc, n)
        qo = pad_array(qo, n)
        a = pad_array(a, n)
        b = pad_array(b, n)
        c = pad_array(c, n)
    return (ql, qr, qm, qo, qc, a, b , c)

def get_setup_values(code, input):
    inputs, body = extract_inputs_and_body(parse(code))
    flatcode = flatten_body(body)
    input_indexes = get_var_placement(inputs, flatcode)
    (ql, qr, qm, qo, qc) = get_q_metrics(flatcode)
    (vl, vr, vo) = get_v_metrics(flatcode, input_indexes)
    input_values = ['-' for i in range(len(input_indexes))]
    input_values = set_values(inputs, input, input_indexes, input_values)

    q_zipped = list(zip(ql, qr, qm, qo, qc))
    v_zipped = list(zip(vl, vr, vo))

    v_values = [ ['-', '-','-'] for i in range(len(v_zipped))]
    v_values = trace_matrix(flatcode, q_zipped, v_zipped, v_values, input_indexes, input_values)
    a, b, c = gen_abc(v_values)

    assert(validate_native(ql, qr, qm, qo, qc, a, b, c)== True)

    # -------------------------------------
    # find omega, roots
    n = len(ql)
    n = 2**int(np.ceil(np.log2(n)))
    assert n & n - 1 == 0, "n must be a power of 2"
    p = 241

    omega = primitive_nth_root_of_unity_by_modulo(n,p)
    root = get_root(omega,n,p)
    # print(f"root = {root}")
    # print(f"omega = {omega}")

    Fp = galois.GF(p)
    omega1 = Fp.primitive_root_of_unity(n)
    # print(f"omega1 = {omega1}")
    roots = Fp([omega1**i for i in range(n)])
    # print(f"roots = {roots}")
    # print(f"p = {p}")
    # print(f"n = {n}")
    # print(f"omega = {omega}")
    assert pow(omega, n, p) == 1, f"omega (ω) {omega} is not a root of unity"

    # -------------------------------------
    # find omega, roots
    (ql, qr, qm, qo, qc, a, b, c) = adjust_witness_size(ql, qr, qm, qo, qc, a, b, c, n)

    # create permutation vectors (sigma) for a, b, and c:
    input_index = find_input_index(inputs, input_indexes)
    (sigma_ai, sigma_bi, sigma_ci) = set_sigma_index(n, vl, vr, vo, input_index)
    k1 = 2
    k2 = 4
    c1_roots = roots
    c2_roots = c1_roots * k1
    c3_roots = c1_roots * k2
    c_roots = np.concatenate((c1_roots, c2_roots, c3_roots))

    check = set()
    for r in c_roots:
        assert not int(r) in check, f"Duplicate root {r} in {c_roots}"
        check.add(int(r))

    sigma1 = Fp([c_roots[sigma_ai[i]] for i in range(0, n)])
    sigma2 = Fp([c_roots[sigma_bi[i]] for i in range(0, n)])
    sigma3 = Fp([c_roots[sigma_ci[i]] for i in range(0, n)])

    return (ql, qr, qm, qo, qc, a, b, c, sigma1, sigma2, sigma3, omega1, roots)

