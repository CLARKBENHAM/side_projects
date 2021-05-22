#goal: Code PCA approx for X
import numpy as np
def check(x,y):
    if np.allclose(x,y):
       return True
    elif np.allclose(np.sort(x), np.sort(y)):
    	print("mis-ordered")
    	return True
    else:
        return x - y

def zero(x, column= True, n=1):
    assert(n > 0)
    if column:
        return np.hstack([x[:,:-n], np.zeros((x.shape[0], n))])
    else:
        return np.vstack([x[:-n,:], np.zeros((n, x.shape[1]))])
   
params = 4
points = 10
X = np.random.rand(points,params)*5
X_bar =  np.tile(np.mean(X, axis=0), (points, 1))#sum across the data points at each colums parameters
X_cen = X - X_bar #0 centered
C = X_cen.T @ X_cen / (points - 1) #covariance matrix
C_val, C_vec = np.linalg.eig(C)
s_val = np.sqrt(C_val * (points-1))

U, s, Vt = np.linalg.svd(X_cen)
S = np.vstack([np.diag(s), np.zeros((points-params, params))])
check(X, U @ S @ Vt + X_bar)
check(C, Vt.T @ S.T @ S @ Vt/(points -1))
check(C_vec, Vt.T)
check(s_val, s)
check(C @ C_vec[:,0], C_vec[:,0]* C_val[0])

for num_comp in range(1,params):
    SVD_app = zero(U, n=num_comp) @ zero(S, n=num_comp) @ zero(Vt, column = False, n=num_comp)
    print(f"With {params - num_comp} Components Approx has Norm of Error: {np.linalg.norm(X_cen - SVD_app)}")
    PC_val, PC_vec = C_val[ix[:num_comp]], C_vec[:,ix[:num_comp]]
    PC_vec = np.hstack([PC_vec, np.zeros((params, params - num_comp))])
    print(np.linalg.norm(X_cen - X_cen @ PC_vec))

num_comp = 2
ix = np.argsort(C_val)[::-1]
PC_val, PC_vec = C_val[ix[:num_comp]], C_vec[:,ix[:num_comp]]
PC_vec = np.hstack([PC_vec, np.zeros((params, params - num_comp))])
np.linalg.norm(X_cen - X_cen @ PC_vec)

from sklearn.decomposition  import PCA
pca = PCA(num_comp)
pca.fit_transform(X_cen)
print(pca.transform(C), '\n', C @ PC_vec)
print("++++++++++++++++++")
print(pca.transform(X_cen), X_cen @ PC_vec, sep='\n')
#isssue with changing from right- to left- cordinates

#How to get from eigenvectors of covariance matrix to the scores for the given data points in new basis?

#SD of points on a loading
