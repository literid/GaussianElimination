import numpy as np


class GaussianEliminationAlgo:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        assert A.shape[0] == A.shape[1], "A has to be square matrix"
        assert A.shape[0] == b.shape[0], "b and A shapes doesnt match"
        self.A = A
        self.b = b

    def run(self):
        A, b = self._forward_elimination(self.A, self.b)
        x = self._backward_elimination(A, b)
        return x

    def _forward_elimination(self, A: np.ndarray, b: np.ndarray):
        n = A.shape[0]
        for col_ind in range(n):
            non_zero_row_ind = self._find_non_zero_row(A, col_ind)
            if non_zero_row_ind is None:
                raise Exception("Singular matrix")

            # swap rows if needed
            if non_zero_row_ind != col_ind:
                A[[col_ind, non_zero_row_ind]] = A[[non_zero_row_ind, col_ind]]
                b[[col_ind, non_zero_row_ind]] = b[[non_zero_row_ind, col_ind]]

            # now non_zero_row index is same as col_ind
            non_zero_row_ind = col_ind
            for row_ind in range(col_ind, n):
                if row_ind == non_zero_row_ind:
                    continue
                multiplier = -A[row_ind, col_ind] / A[non_zero_row_ind, col_ind]
                A[row_ind] += multiplier * A[non_zero_row_ind]
                b[row_ind] += multiplier * b[non_zero_row_ind]

            print(np.c_[A, b])
        return A, b

    def _find_non_zero_row(self, A: np.ndarray, col_ind: int):
        mask = A[:, col_ind] != 0
        for i in range(col_ind, len(mask)):
            if mask[i]:
                return i
        return None

    def _backward_elimination(self, A: np.ndarray, b: np.ndarray):
        n = A.shape[0]
        x = np.zeros(n)
        x[-1] = b[-1] / A[-1, -1]
        for i in range(n - 2, -1, -1):
            s = (A[i, i + 1:] @ x[i + 1:])
            x[i] = 1 / A[i, i] * (b[i] - s)

        return x


A = np.array([[2, 3, 5], [2, 15, -2], [2, -1, 0]], dtype=np.float64)
b = np.array([6, 15, 10], dtype=np.float64)
gauss_elem = GaussianEliminationAlgo(A, b)
# gauss_elem._forward_elimination(A, b)
# gauss_elem._backward_elimination(gauss_elem._forward_elimination(A, b))
print(gauss_elem.run())
