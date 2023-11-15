import numpy as np
from sklearn.preprocessing import *
from sklearn.neighbors import *
import scipy.io
from scipy.linalg import block_diag
from qpsolvers import solve_qp
from scipy.sparse import csr_matrix
import ipdb
from cvxopt import solvers
from cvxopt import matrix

def update_f(W, train_p_target, train_outputs, mu):
    #
    p, q = train_p_target.shape
    WT = np.transpose(W)
    sm = np.sum(WT)
    print('Obtain Hessian matrix...\n')

    T = 2*(np.transpose(np.eye(p) - W) @ (np.eye(p)-W)) + 2/mu*np.eye(p)

    T1 = np.kron(np.eye(q,dtype=float),T)

    N = T1 + np.transpose(T1)
    M = matrix(N,tc='d')
    temp = p*q
    lb = np.zeros(temp) #correct
    ub = np.reshape(train_p_target, temp)
    II = np.eye(p, dtype=np.float)
    A = matrix(np.tile(II, (1, q)),tc='d')
    B = matrix(np.ones(p,dtype=float),tc='d')
    print('quadprpg...\n')
    f = np.reshape(train_outputs, (p * q, 1))
    q_n = matrix((-2/mu)*f,tc='d')
    a_next = np.eye(temp)
    b_next = -1*a_next
    G_next = matrix(np.vstack((a_next,b_next)),tc='d')
    h_next = matrix(np.hstack((ub,lb)),tc='d')
    # print("B =", B.shape);
    outputs = solvers.qp(M, q_n, G_next, h_next, A, B, solver="proxqp")
    answer = np.array(outputs['x'],dtype=np.float32)
    outputs = np.reshape(answer,(p, q))
    # ipdb.set_trace()
    return outputs
