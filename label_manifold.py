import cvxopt
from scipy.linalg import block_diag
import numpy as np
from sklearn.preprocessing import *
from sklearn.neighbors import *
import scipy.io
from qpsolvers import solve_qp
#import ipdb
from scipy.sparse import csr_matrix
import numpy.matlib
from cvxopt import solvers
from cvxopt import matrix
from numpy import array, dtype


def build_label_manifold(train_data, train_p_target, k):
    p, q = train_p_target.shape
    train_data = normalize(train_data)
    kdt = KDTree(train_data, leaf_size=30, metric='euclidean')
    neighbor = kdt.query(train_data, k+1, return_distance=False)
    neighbor = neighbor[:, 1:k+1]
    W = np.zeros((p, p), dtype=float)
    print('Obtain graph matrix S...\n')

    for i in range(p):
        train_data1 = train_data[neighbor[i, :], :]
        D = np.tile(train_data[i, :], (k, 1)) - train_data1
        DD = D @ np.transpose(D)
        lb = np.zeros(k)
        ub = np.ones(k, dtype=float)
        Aeq = np.transpose(ub)
        Beq = np.array([1]);
        Q = np.zeros(k)
        P = 2*DD
        a = np.eye(k)
        b = -1*a
        G = np.vstack((a,b))
        h = np.hstack((ub,lb))
        # print("Beq =", Beq.shape);
        w = solve_qp(P, Q, G, h, Aeq, Beq, solver="proxqp");
        W[i, neighbor[i, :]] = np.transpose(w)

    print('Generate labeling confidence F... \n')
    print('Obtain Hessian...\n')
    WT = np.transpose(W)
    T = WT @ W + W @ np.ones((p, p),dtype=float) @ WT @ np.eye(p) - 2*WT
    M = np.kron(np.eye(q,dtype=float),T)
    temp = p*q
    lb = np.zeros(temp) #correct
    ub = np.reshape(train_p_target, temp) #correct
    II = np.eye(p, dtype = float) #correct
    A = matrix(np.tile(II, (1, q)),tc='d') #correct
    B = matrix(np.ones(p, dtype=float),tc='d') #correct
    M = M + np.transpose(M)
    M = matrix(M,tc='d')
    print('quadprog...\n')
    q_next = matrix(np.zeros(temp),tc='d')
    a_next = np.eye(temp)
    b_next = -1*a_next
    G_next = matrix(np.vstack((a_next,b_next)),tc='d')
    h_next = matrix(np.hstack((ub,lb)),tc='d')
    outputs = solvers.qp(M,q_next , G_next, h_next, A, B)
    answer = np.array(outputs['x'],dtype=np.float32)
    outputs = np.reshape(answer,(p, q))
    # ipdb.set_trace()
    return W, outputs
