from scipy.linalg import block_diag
import numpy as np
from sklearn.preprocessing import *
from sklearn.neighbors import *
import scipy.io
from qpsolvers import solve_qp
import ipdb 
from scipy.sparse import csr_matrix
import numpy.matlib



def build_label_manifold(train_data, train_p_target, k):
    p, q = train_p_target.shape
    train_data = normalize(train_data)
    kdt = KDTree(train_data, leaf_size=30, metric='euclidean')
    neighbor = kdt.query(train_data, k+1, return_distance=False)
    neighbor = neighbor[:, 1:k+1]
    W = np.zeros((p, p), dtype=float)
    # print('Obtain graph matrix S...\n')

    for i in range(p):
        train_data1 = train_data[neighbor[i, :], :]
        D = np.tile(train_data[i, :], (k, 1)) - train_data1
        
        DD = D @ np.transpose(D)
        lb = np.zeros(k)
        ub = np.ones(k, dtype=float)
        Aeq = np.transpose(ub)
        # Aeq = ub
        Beq = 1
        Q = np.zeros(k)
        P = 2*DD
        w = solve_qp(P, Q, None, None, Aeq, Beq,lb,ub)
        W[i, neighbor[i, :]] = np.transpose(w)

    print("HEELLOOOO" , W)
    tempmat = np.zeros((p,p))
    comparison = W == tempmat
    equal_arrays = comparison.all()
    
    print("HERE" , equal_arrays)
    print('\n')
    print('Generate labeling confidence F... \n')

    # M = csr_matrix((p, p), dtype=np.float).toarray()
    print('Obtain Hessian...\n')
    WT = np.transpose(W)
    T = WT @ W + W @ np.ones((p, p),dtype=float) @ WT @ np.eye(p) - 2*WT
    # T1 = np.tile(T, (1, q))
    # T1 = numpy.matlib.repmat(T,q,q)
    # M = csr_matrix(block_diag(T1[:]), dtype=np.float64)
    # M = np.zeros(block_diag(T[:]))
    M = np.kron(np.eye(q,dtype=float),T)
    print("M" , M.shape)
    print(M)
    print("W" , W.shape)
    print("T" , T.shape)
    # print("T1" , T1.shape)
    # lb = csr_matrix((p*q, 1), dtype=np.float).toarray()
    temp = p*q
    lb = np.zeros(temp) #correct
    ub = np.reshape(train_p_target, temp) #correct
    II = np.eye(p, dtype = float) #correct
    A = np.tile(II, (1, q)) #correct
    print("UB: " , ub)
    B = np.ones(p, dtype=float) #correct
    print("M",M.shape)
    M = M + np.transpose(M)
    print('quadprog...\n')
    q = np.zeros(temp)
    print("lb ",lb.shape)
    print("ub ",ub.shape)
    print("Aeq ", A.shape)
    print("Beq ", B.shape)

    a_next = np.eye(temp)
    b_next = -1*a_next
    G_next = np.vstack((a_next,b_next))
    h_next = np.hstack((ub,lb))


    outputs = solve_qp(M,q , G_next, h_next, A, B)
    print("OUUUUU", outputs)
    print("Output" , outputs.shape)
    outputs = np.reshape(outputs,(p, q))
    return W, outputs



