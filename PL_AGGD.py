from label_manifold import *
# from GNN import *
from upgrade_F import *
from obtain_S import *
import numpy as np
from scipy.spatial.distance import pdist
from MulRegression import *

#S = W, F = Y

def PL_AGGD(train_data, train_p_target, test_data, k, par, MaxIter, lamda, mu, gamma):
    

    S, Y = build_label_manifold(train_data, train_p_target, k)
    print('Update parameters...\n')
    [train_outputs, test_outputs] = mulRegression(train_data, Y, test_data, gamma, par)    #GNN(W,S,X,F)
    for i in range(MaxIter):
        print('The ', i, 'th iteration\n')
        W = obtain_s(train_data, Y, k, lamda, mu)
        print('Generating labeling confidence..\n')
        Y = update_f(W, train_p_target, train_outputs, mu)
        print('Update parameters...\n')
        [train_outputs, test_outputs] = mulRegression(train_data, Y, test_data, gamma, par)    #GNN(W,S,X,F)

    return test_outputs, W, Y                                 #W==S==AdjMat, Y==F==LabelConfidence
    
    
