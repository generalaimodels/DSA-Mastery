"""
Matrix Manipulation Module

This module provides comprehensive matrix manipulation functionalities
ranging from basic operations to advanced algorithms. It leverages PyTorch
for optimized tensor computations and adheres to PEP-8 standards with
type annotations for clarity and maintainability.
"""

from typing import List, Tuple
import torch


class Matrix:
    """
    A class to represent a mathematical matrix and perform various
    matrix operations using PyTorch tensors.
    """

    def __init__(self, data: List[List[float]]) -> None:
        """
        Initialize the Matrix with a 2D list.

        Args:
            data (List[List[float]]): 2D list representing the matrix.

        Raises:
            ValueError: If the input data is not a valid 2D list or rows have inconsistent lengths.
        """
        if not data or not all(isinstance(row, list) for row in data):
            raise ValueError("Data must be a non-empty 2D list.")

        row_length = len(data[0])
        if not all(len(row) == row_length for row in data):
            raise ValueError("All rows must have the same number of columns.")

        self.data: torch.Tensor = torch.tensor(data, dtype=torch.float32)
        self.rows: int = self.data.size(0)
        self.cols: int = self.data.size(1)

    def __repr__(self) -> str:
        return f"Matrix({self.data.tolist()})"

    def add(self, other: 'Matrix') -> 'Matrix':
        """
        Add two matrices.

        Args:
            other (Matrix): The matrix to add.

        Returns:
            Matrix: The result of the addition.

        Raises:
            ValueError: If matrix dimensions do not match.
        """
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions to add.")
        result = self.data + other.data
        return Matrix(result.tolist())

    def subtract(self, other: 'Matrix') -> 'Matrix':
        """
        Subtract one matrix from another.

        Args:
            other (Matrix): The matrix to subtract.

        Returns:
            Matrix: The result of the subtraction.

        Raises:
            ValueError: If matrix dimensions do not match.
        """
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions to subtract.")
        result = self.data - other.data
        return Matrix(result.tolist())

    def multiply(self, other: 'Matrix') -> 'Matrix':
        """
        Multiply two matrices using matrix multiplication.

        Args:
            other (Matrix): The matrix to multiply with.

        Returns:
            Matrix: The result of the multiplication.

        Raises:
            ValueError: If the number of columns in the first matrix does not equal the number of rows in the second matrix.
        """
        if self.cols != other.rows:
            raise ValueError(
                "Number of columns in the first matrix must equal the number of rows in the second matrix."
            )
        result = torch.matmul(self.data, other.data)
        return Matrix(result.tolist())

    def scalar_multiply(self, scalar: float) -> 'Matrix':
        """
        Multiply matrix by a scalar.

        Args:
            scalar (float): The scalar value to multiply with.

        Returns:
            Matrix: The result of the scalar multiplication.
        """
        result = self.data * scalar
        return Matrix(result.tolist())

    def transpose(self) -> 'Matrix':
        """
        Transpose the matrix.

        Returns:
            Matrix: The transposed matrix.
        """
        result = self.data.t()
        return Matrix(result.tolist())

    def determinant(self) -> float:
        """
            Compute the determinant of the matrix.

            Returns:
                float: The determinant value.

            Raises:
                ValueError: If the matrix is not square.
            """
        if self.rows != self.cols:
            raise ValueError("Determinant is defined for square matrices only.")
        det = torch.det(self.data)
        return det.item()

    def inverse(self) -> 'Matrix':
        """
            Compute the inverse of the matrix.

            Returns:
                Matrix: The inverse matrix.

            Raises:
                ValueError: If the matrix is not square or is singular.
            """
        if self.rows != self.cols:
            raise ValueError("Inverse is defined for square matrices only.")
        try:
            inv = torch.inverse(self.data)
            return Matrix(inv.tolist())
        except RuntimeError as e:
            raise ValueError("Matrix is singular and cannot be inverted.") from e

    def trace(self) -> float:
        """
            Compute the trace of the matrix.

            Returns:
                float: The trace value.

            Raises:
                ValueError: If the matrix is not square.
            """
        if self.rows != self.cols:
            raise ValueError("Trace is defined for square matrices only.")
        trace_val = torch.trace(self.data)
        return trace_val.item()

    def shape(self) -> Tuple[int, int]:
        """
            Get the shape of the matrix.

            Returns:
                Tuple[int, int]: A tuple representing (rows, columns).
            """
        return self.rows, self.cols

    def is_square(self) -> bool:
        """
            Check if the matrix is square.

            Returns:
                bool: True if square, False otherwise.
            """
        return self.rows == self.cols

    def copy(self) -> 'Matrix':
        """
            Create a copy of the matrix.

            Returns:
                Matrix: A new matrix with the same data.
            """
        return Matrix(self.data.tolist())

    def element_wise_multiply(self, other: 'Matrix') -> 'Matrix':
        """
            Perform element-wise multiplication of two matrices.

            Args:
                other (Matrix): The matrix to multiply element-wise.

            Returns:
                Matrix: The result of element-wise multiplication.

            Raises:
                ValueError: If matrix dimensions do not match.
            """
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions for element-wise multiplication.")
        result = self.data * other.data
        return Matrix(result.tolist())

    def element_wise_divide(self, other: 'Matrix') -> 'Matrix':
        """
            Perform element-wise division of two matrices.

            Args:
                other (Matrix): The matrix to divide by element-wise.

            Returns:
                Matrix: The result of element-wise division.

            Raises:
                ValueError: If matrix dimensions do not match.
                ZeroDivisionError: If any element in the divisor matrix is zero.
            """
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions for element-wise division.")
        if torch.any(other.data == 0):
            raise ZeroDivisionError("Division by zero encountered in the divisor matrix.")
        result = self.data / other.data
        return Matrix(result.tolist())


class AdvancedMatrix(Matrix):
    """
    A subclass of Matrix that includes advanced matrix operations.
    """

    def eigenvalues(self) -> Tuple[List[float], List[List[float]]]:
        """
            Compute the eigenvalues and eigenvectors of the matrix.

            Returns:
                Tuple[List[float], List[List[float]]]: A tuple containing a list of eigenvalues and a list of eigenvectors.

            Raises:
                ValueError: If the matrix is not square.
            """
        if not self.is_square():
            raise ValueError("Eigenvalues are defined for square matrices only.")
        eigen_decomposition = torch.linalg.eig(self.data)
        eigenvalues = eigen_decomposition.eigenvalues
        eigenvectors = eigen_decomposition.eigenvectors
        eigenvalues_list = eigenvalues.real.tolist()
        eigenvectors_list = eigenvectors.real.tolist()
        return eigenvalues_list, eigenvectors_list

    def singular_value_decomposition(self) -> Tuple['Matrix', 'Matrix', 'Matrix']:
        """
            Perform Singular Value Decomposition (SVD) on the matrix.

            Returns:
                Tuple[Matrix, Matrix, Matrix]: The U, S, and V^T matrices from the SVD.
            """
        U, S, Vh = torch.linalg.svd(self.data, full_matrices=False)
        U_matrix = Matrix(U.tolist())
        S_matrix = Matrix(torch.diag(S).tolist())
        Vh_matrix = Matrix(Vh.tolist())
        return U_matrix, S_matrix, Vh_matrix

    def rank(self) -> int:
        """
            Compute the rank of the matrix.

            Returns:
                int: The rank of the matrix.
            """
        return torch.linalg.matrix_rank(self.data).item()

    def determinant_recursive(self) -> float:
        """
            Compute the determinant of the matrix recursively.

            Note:
                This method is inefficient for large matrices and is for educational purposes only.

            Returns:
                float: The determinant value.

            Raises:
                ValueError: If the matrix is not square.
            """
        if self.rows != self.cols:
            raise ValueError("Determinant is defined for square matrices only.")

        if self.rows == 1:
            return self.data[0][0].item()
        if self.rows == 2:
            return (self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]).item()

        det = 0.0
        for col in range(self.cols):
            minor = self.minor(0, col)
            cofactor = ((-1) ** col) * self.data[0][col] * minor.determinant_recursive()
            det += cofactor
        return det

    def minor(self, row: int, col: int) -> 'Matrix':
        """
            Compute the minor of the matrix excluding the specified row and column.

            Args:
                row (int): The row to exclude.
                col (int): The column to exclude.

            Returns:
                Matrix: The minor matrix.
            """
        minor_data = [
            [
                self.data[r][c].item()
                for c in range(self.cols) if c != col
            ]
            for r in range(self.rows) if r != row
        ]
        return Matrix(minor_data)

    def condition_number(self) -> float:
        """
            Compute the condition number of the matrix.

            Returns:
                float: The condition number.

            Raises:
                ValueError: If the matrix is not square.
            """
        if not self.is_square():
            raise ValueError("Condition number is defined for square matrices only.")
        cond = torch.linalg.cond(self.data)
        return cond.item()

    def norm(self, ord: int = 2) -> float:
        """
            Compute the norm of the matrix.

            Args:
                ord (int, optional): The order of the norm. Defaults to 2.

            Returns:
                float: The norm of the matrix.
            """
        norm_val = torch.linalg.norm(self.data, ord=ord)
        return norm_val.item()

    def solve_linear_system(self, b: 'Matrix') -> 'Matrix':
        """
            Solve the linear system Ax = b.

            Args:
                b (Matrix): The right-hand side matrix.

            Returns:
                Matrix: The solution matrix x.

            Raises:
                ValueError: If the matrix is not square or dimensions do not align.
            """
        if not self.is_square():
            raise ValueError("Matrix A must be square to solve Ax = b.")
        if b.rows != self.rows:
            raise ValueError("Matrix b must have the same number of rows as matrix A.")
        try:
            solution = torch.linalg.solve(self.data, b.data)
            return Matrix(solution.tolist())
        except RuntimeError as e:
            raise ValueError("Linear system cannot be solved.") from e

    def power(self, exponent: int) -> 'Matrix':
        """
            Compute the matrix raised to a non-negative integer power.

            Args:
                exponent (int): The exponent to raise the matrix to.

            Returns:
                Matrix: The resulting matrix after exponentiation.

            Raises:
                ValueError: If the matrix is not square or exponent is negative.
            """
        if not self.is_square():
            raise ValueError("Exponentiation is defined for square matrices only.")
        if exponent < 0:
            raise ValueError("Exponent must be a non-negative integer.")
        result = torch.matrix_power(self.data, exponent)
        return Matrix(result.tolist())

    def trace_norm(self) -> float:
        """
            Compute the trace norm (nuclear norm) of the matrix.

            Returns:
                float: The trace norm of the matrix.
            """
        trace_norm = torch.linalg.norm(self.data, ord='nuc')
        return trace_norm.item()

    def frobenius_norm(self) -> float:
        """
            Compute the Frobenius norm of the matrix.

            Returns:
                float: The Frobenius norm of the matrix.
            """
        fro_norm = torch.linalg.norm(self.data, ord='fro')
        return fro_norm.item()

    def is_symmetric(self) -> bool:
        """
            Check if the matrix is symmetric.

            Returns:
                bool: True if symmetric, False otherwise.
            """
        if not self.is_square():
            return False
        return torch.allclose(self.data, self.data.t()).item()

    def is_positive_definite(self) -> bool:
        """
            Check if the matrix is positive definite.

            Returns:
                bool: True if positive definite, False otherwise.
            """
        if not self.is_square():
            return False
        try:
            torch.linalg.cholesky(self.data)
            return True
        except RuntimeError:
            return False

    def hadamard_product(self, other: 'Matrix') -> 'Matrix':
        """
            Compute the Hadamard product (element-wise multiplication) of two matrices.

            Args:
                other (Matrix): The matrix to multiply element-wise.

            Returns:
                Matrix: The Hadamard product of the two matrices.

            Raises:
                ValueError: If matrix dimensions do not match.
            """
        return self.element_wise_multiply(other)

    def kronecker_product(self, other: 'Matrix') -> 'Matrix':
        """
            Compute the Kronecker product of two matrices.

            Args:
                other (Matrix): The matrix to compute the Kronecker product with.

            Returns:
                Matrix: The Kronecker product of the two matrices.
            """
        result = torch.kron(self.data, other.data)
        return Matrix(result.tolist())

    def matrix_exponential(self) -> 'Matrix':
        """
            Compute the matrix exponential of the matrix.

            Returns:
                Matrix: The matrix exponential.

            Raises:
                ValueError: If the matrix is not square.
            """
        if not self.is_square():
            raise ValueError("Matrix exponential is defined for square matrices only.")
        expm = torch.matrix_exp(self.data)
        return Matrix(expm.tolist())


def main():
    """
    Demonstrate the usage of the Matrix and AdvancedMatrix classes with example operations.
    """
    # Basic Matrix Operations
    A_data = [
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0]
    ]
    B_data = [
        [7, 8, 9],
        [2, 3, 4],
        [5, 6, 1]
    ]

    A = AdvancedMatrix(A_data)
    B = AdvancedMatrix(B_data)

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # Addition
    C = A.add(B)
    print("\nA + B:")
    print(C)

    # Subtraction
    D = A.subtract(B)
    print("\nA - B:")
    print(D)

    # Multiplication
    E = A.multiply(B)
    print("\nA * B:")
    print(E)

    # Scalar Multiplication
    F = A.scalar_multiply(2)
    print("\nA * 2:")
    print(F)

    # Transpose
    At = A.transpose()
    print("\nTranspose of A:")
    print(At)

    # Determinant
    det_A = A.determinant()
    print(f"\nDeterminant of A: {det_A}")

    # Inverse
    try:
        A_inv = A.inverse()
        print("\nInverse of A:")
        print(A_inv)
    except ValueError as e:
        print(f"\nError computing inverse of A: {e}")

    # Trace
    trace_A = A.trace()
    print(f"\nTrace of A: {trace_A}")

    # Advanced Operations
    # Eigenvalues and Eigenvectors
    try:
        eigenvalues, eigenvectors = A.eigenvalues()
        print(f"\nEigenvalues of A: {eigenvalues}")
        print(f"Eigenvectors of A: {eigenvectors}")
    except ValueError as e:
        print(f"\nError computing eigenvalues of A: {e}")

    # Singular Value Decomposition
    U, S, Vh = A.singular_value_decomposition()
    print("\nSingular Value Decomposition of A:")
    print("U matrix:")
    print(U)
    print("S matrix:")
    print(S)
    print("V^T matrix:")
    print(Vh)

    # Rank
    rank_A = A.rank()
    print(f"\nRank of A: {rank_A}")

    # Solving Linear System Ax = b
    b_data = [
        [1],
        [2],
        [3]
    ]
    b = Matrix(b_data)
    try:
        x = A.solve_linear_system(b)
        print("\nSolution to Ax = b:")
        print(x)
    except ValueError as e:
        print(f"\nError solving Ax = b: {e}")

    # Matrix Power
    A_squared = A.power(2)
    print("\nA squared:")
    print(A_squared)

    # Condition Number
    cond_A = A.condition_number()
    print(f"\nCondition number of A: {cond_A}")

    # Norms
    fro_norm_A = A.frobenius_norm()
    trace_norm_A = A.trace_norm()
    print(f"\nFrobenius norm of A: {fro_norm_A}")
    print(f"Trace norm of A: {trace_norm_A}")

    # Symmetry Check
    symmetric = A.is_symmetric()
    print(f"\nIs A symmetric? {symmetric}")

    # Positive Definiteness
    pos_def = A.is_positive_definite()
    print(f"Is A positive definite? {pos_def}")

    # Recursive Determinant (Educational Purpose)
    det_A_recursive = A.determinant_recursive()
    print(f"\nRecursive Determinant of A: {det_A_recursive}")

    # Additional Operations
    # Element-wise Multiplication
    try:
        elem_mul = A.element_wise_multiply(B)
        print("\nElement-wise Multiplication of A and B:")
        print(elem_mul)
    except ValueError as e:
        print(f"\nError in element-wise multiplication: {e}")

    # Element-wise Division
    try:
        elem_div = A.element_wise_divide(B)
        print("\nElement-wise Division of A by B:")
        print(elem_div)
    except (ValueError, ZeroDivisionError) as e:
        print(f"\nError in element-wise division: {e}")

    # Hadamard Product
    try:
        hadamard = A.hadamard_product(B)
        print("\nHadamard Product of A and B:")
        print(hadamard)
    except ValueError as e:
        print(f"\nError in Hadamard product: {e}")

    # Kronecker Product
    kronecker = A.kronecker_product(B)
    print("\nKronecker Product of A and B:")
    print(kronecker)

    # Matrix Exponential
    try:
        A_expm = A.matrix_exponential()
        print("\nMatrix Exponential of A:")
        print(A_expm)
    except ValueError as e:
        print(f"\nError computing matrix exponential: {e}")


if __name__ == "__main__":
    main()