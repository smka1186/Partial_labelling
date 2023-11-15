from numpy import array, dot, dtype
from qpsolvers import solve_qp
import numpy as np


M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = dot(M.T, M) # this is a positive definite matrix
q = dot(array([3., 2., 3.]), M).reshape((3,))
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.]).reshape((3,))
A = array([1., 1., 1.])
b = array([1.])
lb = np.array([0,0,0],dtype=float)
ub = np.array([100,100,100],dtype=float)
print("DD ", P.shape)
print("lb ",lb.shape)
print("ub ",ub.shape)
print("Q ", q.shape)
print("Aeq ", A.shape)
print("Beq ", b.shape)

x = solve_qp(P, q, None, None, A, b,lb,ub)
print("QP solution: x = {}".format(x))