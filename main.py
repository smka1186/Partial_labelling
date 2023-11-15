import numpy as np
from sklearn.preprocessing import *
from sklearn.neighbors import *
import scipy.io
from scipy.spatial.distance import pdist
from PL_AGGD import PL_AGGD
from CalAccuracy import calAccuracy


#loading .mat file for dataset

mat = scipy.io.loadmat('./Lost.mat')

# print(mat);

test_data = [[element for element in upperElement] for upperElement in mat['test_data']]
test_data = np.array(test_data)
# print(test_data.shape);
test_target = [[element for element in upperElement] for upperElement in mat['test_target']]
test_target = np.array(test_target)
train_data = [[element for element in upperElement] for upperElement in mat['train_data']]
train_data = np.array(train_data)
train_p_target = [[element for element in upperElement] for upperElement in mat['train_p_target']]
train_p_target = np.array(train_p_target)
# print("TARGETS", train_p_target)
# print(train_data.shape, ' ', test_data.shape)



k = 10
lamda = 1
mu = 1
gama = 0.05
Maxiter = 1
par = np.mean(pdist(train_data))

test_outputs, S, F = PL_AGGD(train_data, train_p_target, test_data, k, par, Maxiter, lamda, mu, gama)
# test_outputs = gnn(S, train_data, train_p_target, test_data, F)
accuracy = calAccuracy(test_outputs, test_target)
print('The accuracy of PL-AGGD is: %f \n', accuracy)
