import pathlib
import math
import re

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    matrix = []
    vector = []
    content = path.read_text().splitlines()
    for line in content :
        line = line.replace(" ", "").strip()
        print(line)

        x_match = re.search(r'([+-]?\d*\.?\d*)x', line)
        y_match = re.search(r'([+-]?\d*\.?\d*)y', line)
        z_match = re.search(r'([+-]?\d*\.?\d*)z', line)
        constant_match = re.search(r'=([+-]?\d*)', line)

        # Handle missing coefficients (empty means 1 or -1)
        x_coeff = float(x_match.group(1)) if x_match and x_match.group(1) not in ["", "+", "-"] else float(
            f"{x_match.group(1)}1") if x_match else 0
        y_coeff = float(y_match.group(1)) if y_match and y_match.group(1) not in ["", "+", "-"] else float(
            f"{y_match.group(1)}1") if y_match else 0
        z_coeff = float(z_match.group(1)) if z_match and z_match.group(1) not in ["", "+", "-"] else float(
            f"{z_match.group(1)}1") if z_match else 0
        constant = float(constant_match.group(1)) if constant_match else 0

        # Append the coefficients and constant to the matrix and vector
        matrix.append([x_coeff, y_coeff, z_coeff])
        vector.append(constant)
    return matrix, vector

def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif len(matrix) == 3:
        return (
                matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
    else:
        raise ValueError("Only 2x2 or 3x3 matrices are supported by the program.")

def trace(matrix: list[list[float]]) -> float:
    sum = matrix[0][0] + matrix[1][1] + matrix[2][2]
    return sum

def norm(vector: list[float]) -> float:
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    return norm

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = []
    for row in matrix:
        row_product = sum(row[i] * vector[i] for i in range(len(vector)))
        result.append(row_product)
    return result

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(matrix)
    if det_A == 0:
        print("The system has no unique solution (det(A) = 0).")
        return [0, 0, 0]
    else:
        # Form Ax, Ay, and Az matrices
        A_x = replace_column(matrix, vector, 0)  # Replace 1st column for x
        A_y = replace_column(matrix, vector, 1)  # Replace 2nd column for y
        A_z = replace_column(matrix, vector, 2)  # Replace 3rd column for z

        # Compute determinants for Ax, Ay, Az
        det_Ax = determinant(A_x)
        det_Ay = determinant(A_y)
        det_Az = determinant(A_z)

        # Cramer's rule
        x = det_Ax / det_A
        y = det_Ay / det_A
        z = det_Az / det_A

        return [x, y, z]

def replace_column(matrix: list[list[float]], vector: list[float], col_index: int):
    new_matrix = [row[:] for row in matrix]  # copy of the matrix
    for i in range(len(matrix)):
        new_matrix[i][col_index] = vector[i]  # Replace the column with the vector
    return new_matrix

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    """Returns the 2x2 minor of a 3x3 matrix after removing the given row and column."""
    return [
        [matrix[row][col] for col in range(3) if col != j]
        for row in range(3) if row != i
    ]

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    """Computes the cofactor matrix of a 3x3 matrix."""
    cofactor_matrix = []
    for i in range(3):
        cofactor_row = []
        for j in range(3):
            minor_matrix = minor(matrix, i, j)
            cofactor_value = ((-1) ** (i + j)) * determinant(minor_matrix)
            cofactor_row.append(cofactor_value)
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det = determinant(matrix)
    if det == 0:
        raise ValueError("The matrix is singular (det(A) = 0), so it cannot be inverted.")
    adj_matrix = adjoint(matrix)
    inv_matrix = [[adj_matrix[i][j] / det for j in range(3)] for i in range(3)]
    return multiply(inv_matrix, vector)

A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=} {B=}")
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")
print(f"{solve_cramer(A, B)=}")
print(f"{solve(A, B)=}")