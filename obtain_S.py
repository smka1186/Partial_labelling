import numpy as np
from sklearn.preprocessing import *
from sklearn.neighbors import *
import scipy.io
from qpsolvers import solve_qp
from scipy.sparse import csr_matrix
from cvxopt import solvers
from cvxopt import matrix
import ipdb

def obtain_s(train_data, Y, k, lamda, mu):

    p, q = Y.shape
    train_data = normalize(train_data)
    kdt = KDTree(train_data,leaf_size=30,metric='euclidean')
    neighbor = kdt.query(train_data, k+1, return_distance=False)
    neighbor = neighbor[:, 1:k+1]
    W = np.zeros((p, p),dtype=np.float64)
    print('Obtain graph matrix S...\n')
    step = p/100
    count = 0
    steps = 100/p
    print('workingg')

    for i in range(p):
        if (i % step) < 1:
            #print(np.tile('\b', 1, count-1)
            print('\b'*(count-1))
            ##count = print(1, '>%d%%', round((i+1)*steps))
        train_data1 = train_data[neighbor[i, :], :]
        D = np.tile(train_data[i, :], (k, 1)) - train_data1
        DD = D @ np.transpose(D)
        Y1 = Y[neighbor[i, :], :]
        Dy = np.tile(Y[i, :], (k, 1)) - Y1
        DyDy = Dy @ np.transpose(Dy)
        DDDD = lamda*DD + mu*DyDy
        lb = np.zeros(k)
        ub = np.ones(k, dtype=float)
        Aeq = np.transpose(ub)
        Beq = np.array([1]);
        Q = np.zeros(k)
        P = 2*DDDD
        a = np.eye(k)
        b = -1*a
        G = np.vstack((a,b))
        h = np.hstack((ub,lb))
        w = solve_qp(P, Q, G, h, Aeq, Beq, solver="proxqp")
        W[i, neighbor[i, :]] = np.transpose(w)
    return W




