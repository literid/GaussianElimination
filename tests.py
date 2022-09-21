import numpy as np

from GaussianElimination import GaussianEliminationAlgo, GaussianEliminationMainElementAlgo


def compute_error(algo_type: GaussianEliminationAlgo, A: np.ndarray, b: np.ndarray, real_solution: np.ndarray = None):
    """
    Computes residual and error, if real_solution unknown computes only residual
    :param algo_type: type of algorythm
    :param A: system matrix
    :param b: system rhs column
    :param real_solution: system true solution, default: None
    :return: (residual, error)
    """
    algo = algo_type(A, b, copy_input=True)
    numerical_solution = algo.run()
    residual = b - A @ numerical_solution
    if real_solution:
        error = real_solution - numerical_solution
    else:
        error = None
    return residual, error


def test_case1(algo_type: GaussianEliminationAlgo = GaussianEliminationAlgo):
    A = np.array([[2]], dtype=np.float64)
    b = np.array([5], dtype=np.float64)
    real_solution = 5 / 2
    residual, error = compute_error(algo_type, A, b, real_solution)
    print(f"Невязка = {residual}, ошибка = {error}")


def test_case2(algo_type: GaussianEliminationAlgo = GaussianEliminationAlgo):
    A = np.array([[2, 5], [-6, 10]], dtype=np.float64)
    b = np.array([5, 8], dtype=np.float64)
    real_solution = [1 / 5, 23 / 25]
    residual, error = compute_error(algo_type, A, b, real_solution)
    print(f"Невязка = {residual}, ошибка = {error}")


def test_case3(algo_type: GaussianEliminationAlgo = GaussianEliminationAlgo):
    A = np.array([[4, 5, 43], [-6, 10, -243], [26, 56, 91]], dtype=np.float64)
    b = np.array([5, 8, 9], dtype=np.float64)
    real_solution = [10487 / 512, -14233 / 1792, -1551 / 1792]
    residual, error = compute_error(algo_type, A, b, real_solution)
    print(f"Невязка = {residual}, ошибка = {error}")


def test_case(algo_type: GaussianEliminationAlgo = GaussianEliminationAlgo, n=5):
    np.random.seed(986)
    A = np.random.random(size=(n, n))
    b = np.random.random(size=n)
    real_solution = None
    residual, error = compute_error(algo_type, A, b, real_solution)
    print(f"Невязка = {np.abs(residual).max()}, ошибка = {error}")

if __name__ == '__main__':
    print("Тест для методы Гаусса с главным элементом")
    test_case(GaussianEliminationMainElementAlgo, n = 1000)
    print("Тест для метода обычного Гаусса")
    test_case(GaussianEliminationAlgo, n = 1000)
