'''
1) Input/output
2) Initalization Methods
3) Underflow of zero
4) Metrics
5) Examples
'''

import nimfa
import numpy as np

# Read the medulloblastoma gene expression data. The matrix's shape is 5893 (genes) x 34 (samples).
# Normalize: V = (V - V.min()) / (V.max() - V.min())
# V = nimfa.examples.medulloblastoma.read(normalize = True)

# actually is nimfa.methods.factorization.lsnmf
# print "Number of zero elements: ", V.size - np.count_nonzero(V)

'''
Example-1: Simple Usage -- Projected Gradient Update
Covers: 1) Input/Output 2) Metrics
'''

V = np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]])
print('Target:\n%s' % V)

lsnmf = nimfa.Lsnmf(V, distance = "euclidean", seed='random_vcol', max_iter=10, rank=3 )
lsnmf_fit = lsnmf()

W = lsnmf_fit.basis()
print('Basis matrix:\n%s' % W)

H = lsnmf_fit.coef()
print('Mixture matrix:\n%s' % H)

print('Target estimate:\n%s' % np.dot(W, H))

print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))
print('Euclidean distance: %5.3f' % lsnmf_fit.distance(metric='euclidean'))
print('Iterations: %d' % lsnmf_fit.n_iter)

print('Rss: %5.3f' % lsnmf_fit.fit.rss())
print('Evar: %5.3f' % lsnmf_fit.fit.evar())

'''
Example-2: Simple Usage -- Multiplicative update

'''
V = np.random.rand(30, 20)

init_W = np.random.rand(30, 4)
init_H = np.random.rand(4, 20)

# Fixed initialization of latent matrices
nmf = nimfa.Nmf(V, seed="fixed", W=init_W, H=init_H, rank=4)
nmf_fit = nmf()

print("Euclidean distance: %5.3f" % nmf_fit.distance(metric="euclidean"))
print('Initialization type: %s' % nmf_fit.seeding)
print('Iterations: %d' % nmf_fit.n_iter)


'''
Example-3: Initialization methods
'''
#Read the medulloblastoma gene expression data. The matrix's shape is 5893 (genes) x 34 (samples).
#Normalize: V = (V - V.min()) / (V.max() - V.min())
print("="*15)
print("Testing the initialization methods")
print("="*15)
V = nimfa.examples.medulloblastoma.read(normalize = True)

# Random Initialization
lsnmf = nimfa.Lsnmf(V, rank=30, max_iter=100, seed = "random", distance = "kl")
lsnmf_fit = lsnmf()
print('Initialization type: %s' % lsnmf_fit.seeding)
print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))
print("="*15)

# NNDSVD Initialization
lsnmf = nimfa.Lsnmf(V, rank=30, max_iter=100, seed = "nndsvd", distance = "kl")
lsnmf_fit = lsnmf()
W0 = lsnmf_fit.basis()
H0 = lsnmf_fit.coef()
print('Initialization type: %s' % lsnmf_fit.seeding)
print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))
print("="*15)

# Fix Initialization
lsnmf = nimfa.Nmf(V, seed="fixed", W=W0, H=H0, rank=30, max_iter = 100, distance = "kl")
lsnmf_fit = lsnmf()
print('Starting from NNDSVD -> Initialization type: %s' % lsnmf_fit.seeding)
print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))
print("="*15)

# Random_C Initialization
lsnmf = nimfa.Lsnmf(V, rank=30, max_iter=100, seed = "random_c", distance = "kl")
lsnmf_fit = lsnmf()
print('Initialization type: %s' % lsnmf_fit.seeding)
print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))
print("="*15)

# Random_Vcol Initialization
lsnmf = nimfa.Lsnmf(V, rank=30, max_iter=100, seed = "random_vcol", distance = "kl")
lsnmf_fit = lsnmf()
W0 = lsnmf_fit.basis()
H0 = lsnmf_fit.coef()
print('Initialization type: %s' % lsnmf_fit.seeding)
print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))
print("="*15)

# Fix Initialization
lsnmf = nimfa.Nmf(V, seed="fixed", W=W0, H=H0, rank=30, max_iter = 100, distance = "kl")
lsnmf_fit = lsnmf()
print('Starting from Random-Vcol -> Initialization type: %s' % lsnmf_fit.seeding)
print('K-L divergence: %5.3f' % lsnmf_fit.distance(metric='kl'))
print("="*15)





