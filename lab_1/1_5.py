import numpy as np
import warnings


warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)


def QR_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix_size = matrix.shape[0]

    matrix_cpy = np.copy(matrix)
    Q = np.eye(matrix_size, dtype=float)

    for i in range(matrix_size):
        v = np.zeros((matrix_size, 1), dtype=float)
        v[i] = matrix_cpy[i][i] + np.sign(matrix_cpy[i][i]) * np.sqrt(
            sum(
                [matrix_cpy[j][i] ** 2 for j in range(i, matrix_size)]
            )
        )

        for j in range(i + 1, matrix_size):
            v[j] = matrix_cpy[j][i]

        v_transpose = np.transpose(v)

        E = np.eye(matrix_size, dtype=float)
        H = E - 2 * (v @ v_transpose) / (v_transpose @ v)

        Q @= H
        matrix_cpy = H @ matrix_cpy

    return Q, matrix_cpy


def QR_algorithm(
        matrix: np.ndarray,
        epsilon: float =1e-10,
        max_iterations: int =1000
) -> np.ndarray:
    A_k = np.array(matrix, dtype=complex)
    matrix_size = matrix.shape[0]

    for _ in range(max_iterations):
        Q, R = QR_decomposition(A_k)
        A_k = R @ Q

        if np.sum(np.abs(np.tril(A_k, -1))) < epsilon:
            break

    eigenvalues = []
    i = 0
    while i < matrix_size:
        if i < matrix_size - 1 and abs(A_k[i + 1, i]) > epsilon:
            a, b = A_k[i, i], A_k[i, i + 1]
            c, d = A_k[i + 1, i], A_k[i + 1, i + 1]
            trace = a + d
            det = a * d - b * c
            discr = trace ** 2 - 4 * det

            if discr < 0:
                val1 = trace / 2 + 1j * np.sqrt(-discr) / 2
                val2 = trace / 2 - 1j * np.sqrt(-discr) / 2
                eigenvalues.extend([val1, val2])
            else:
                val1 = (trace + np.sqrt(discr)) / 2
                val2 = (trace - np.sqrt(discr)) / 2
                eigenvalues.extend([val1, val2])
            i += 2
        else:
            eigenvalues.append(A_k[i, i])
            i += 1

    return np.array(eigenvalues)


def check_QR_decomposition(
        matrix: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        eps: float = 0.001
) -> bool:
    founded_matrix = Q @ R

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if abs(matrix[i, j] - founded_matrix[i, j]) > eps:
                return False

    return True


if __name__ == '__main__':
    n = int(input('Input matrix size (one positive number): '))

    print('Input matrix:')
    input_matrix = np.empty((n, n), dtype=float)
    for row in range(n):
        input_matrix[row] = np.array(list(map(
            float,
            input().split())))

    eps = float(input('Input error rate: '))

    Q, R = QR_decomposition(input_matrix)

    print("Founded QR decomposition")
    print("\nQ matrix:\n", Q)
    print("\nR matrix:\n", R)
    print("\nQR matrix:\n", Q @ R)

    if check_QR_decomposition(input_matrix, Q, R, eps):
        print("\nQR decomposition founded correctly")
    else:
        print("\nERROR: QR decomposition is not correct")
        exit(1)

    eigenvalues = QR_algorithm(input_matrix, eps)

    print('\nEigenvalues:')
    for i, val in enumerate(eigenvalues):
        if val.imag == 0:
            print(f"λ_{i + 1} = {val.real:.8f} {'+ 0.0i'}")
        else:
            print(f"λ_{i + 1} = {val.real:.8f} {'+' if val.imag >= 0 else '-'} {abs(val.imag):.8f}i")

    print()

    print("Numpy eigenvalues (builtin function):")
    for i, val in enumerate(np.linalg.eig(input_matrix).eigenvalues):
        if val.imag == 0:
            print(f"λ_{i + 1} = {val.real:.8f} {'+ 0.0i'}")
        else:
            print(f"λ_{i + 1} = {val.real:.8f} {'+' if val.imag >= 0 else '-'} {abs(val.imag):.8f}i")