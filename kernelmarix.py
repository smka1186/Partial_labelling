from sklearn.metrics.pairwise import rbf_kernel


def kernelmatrix(x, y, gamma):
    K = rbf_kernel(x, y, gamma)
    return K
