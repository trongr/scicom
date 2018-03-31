import numpy as np

A = np.array([[1, 2, 3], [3, 2, 1], [1, 0, -1]])
w, v = np.linalg.eig(A)
print(w)  # eigenvalues
print(v)  # eigenvectors

l, u = w[0], v[:, 0]
print(np.dot(A, u))
print(l * u)
assert np.allclose(np.dot(A, u), l * u), "u is an eigenvector and l is the corresponding eigenvalue"

A = np.random.rand(3, 3)
w, v = np.linalg.eig(A)
print(w)
print(v)
