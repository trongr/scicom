import numpy as np
from numpy import linalg as LA


def is_positive_definite(A):
    return np.all(LA.eigvals(A) > 0)


def main():
    A = np.array([[1, 2, 3], [3, 2, 1], [1, 0, -1]])
    w, v = LA.eig(A)
    print(w)  # eigenvalues
    print(v)  # eigenvectors

    l, u = w[0], v[:, 0]
    print(np.dot(A, u))
    print(l * u)
    assert np.allclose(np.dot(A, u), l * u), "u is an eigenvector and l is the corresponding eigenvalue"

    A = np.random.rand(3, 3)
    w, v = LA.eig(A)
    print("Eigenvalues of A", w)
    print("Is A positive definite?", is_positive_definite(A))
    print(v)


if __name__ == "__main__":
    main()
