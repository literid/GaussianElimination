import numpy as np


class GaussianEliminationAlgo:
    def _forward_elimination(self, A: np.ndarray, b: np.ndarray):
        assert A.shape[0] == A.shape[1], "A has to be square matrix"
        assert A.shape[0] == b.shape[0], "b and A shapes doesnt match"

        self.starting_rows = set()
        n = A.shape[0]
        for col_ind in range(n):
            non_zero_row_ind = self._find_row(A, col_ind)
            if non_zero_row_ind is None:
                raise Exception("Singular matrix")

            self.starting_rows.add(non_zero_row_ind)
            for row_ind in range(n):
                if row_ind in self.starting_rows:
                    continue
                multiplier = -A[row_ind, col_ind] / A[non_zero_row_ind, col_ind]
                A[row_ind] += multiplier * A[non_zero_row_ind]
                b[row_ind] += multiplier * b[non_zero_row_ind]
            print(np.c_[A, b])

    def _find_row(self, A: np.ndarray, col_ind: int):
        mask = A[:, col_ind] != 0
        for i in range(len(mask)):
            if i not in self.starting_rows and mask[i]:
                return i
        return None


A = np.array([[1, 2, 3, 4], [1, 2, -2, 4], [1, 4, 3, 2], [1, 0, 5, 0]], dtype=np.float64)
b = np.array([1, 1, 1, 1], dtype=np.float64)
gauss_elem = GaussianEliminationAlgo()
gauss_elem._forward_elimination(A, b)
