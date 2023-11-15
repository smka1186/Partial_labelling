import numpy as np
from kernelmarix import *
import ipdb




def mulRegression(train_data, train_p_target, test_data, lamda, par):
    m, k = train_data.shape
    t, u = test_data.shape
    # K = kernelmatrix(np.transpose(train_data), np.transpose(train_data), par)
    K = kernelmatrix(train_data,train_data,par)
    Kt = kernelmatrix(test_data, train_data, par)
    I = np.eye(m, m)
    print(I.shape)
    print(K.shape)
    print(lamda)
    # ipdb.set_trace()
    H = (1/(2*lamda))*K + 1/2*I
    m1 = np.ones((m, 1))
    s = np.transpose(np.linalg.inv(H) @ m1)
    P = train_p_target
    numerator = s @ P
    denominator = s @ m1
    b = numerator / denominator
    alpha = np.linalg.inv(H) @ (P - np.tile(b, (m, 1)))

    train_outputs = 1/(2*lamda)*K@alpha + np.tile(b, (m, 1))
    test_outputs = 1/(2*lamda)*Kt@alpha + np.tile(b, (t, 1))

    return train_outputs, test_outputs

