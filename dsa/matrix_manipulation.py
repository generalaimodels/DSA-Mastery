"""
Matrix Manipulation Module

This module provides a comprehensive set of matrix manipulation functionalities
ranging from basic operations to advanced algorithms using PyTorch tensors.
All functions adhere to PEP-8 standards, include type hints for clarity, and
implement robust error handling to ensure reliability and maintainability.
"""

import torch
from typing import Union, Tuple


class MatrixManipulation:
    """
    A class for performing various matrix operations using PyTorch tensors.
    """

    def __init__(self, matrix: torch.Tensor):
        """
        Initializes the MatrixManipulation instance with a given matrix.

        Args:
            matrix (torch.Tensor): A 2D tensor representing the matrix.

        Raises:
            ValueError: If the input is not a 2D tensor.
        """
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if matrix.dim() != 2:
            raise ValueError("Input tensor must be 2-dimensional.")
        self.matrix = matrix

    def add(self, other: torch.Tensor) -> torch.Tensor:
        """
        Adds another matrix to the current matrix.

        Args:
            other (torch.Tensor): A 2D tensor to add.

        Returns:
            torch.Tensor: The result of the addition.

        Raises:
            ValueError: If the matrices are not of the same shape.
        """
        if not isinstance(other, torch.Tensor):
            raise TypeError("The addend must be a PyTorch tensor.")
        if other.shape != self.matrix.shape:
            raise ValueError("Matrices must have the same dimensions for addition.")
        return self.matrix + other

    def subtract(self, other: torch.Tensor) -> torch.Tensor:
        """
        Subtracts another matrix from the current matrix.

        Args:
            other (torch.Tensor): A 2D tensor to subtract.

        Returns:
            torch.Tensor: The result of the subtraction.

        Raises:
            ValueError: If the matrices are not of the same shape.
        """
        if not isinstance(other, torch.Tensor):
            raise TypeError("The subtrahend must be a PyTorch tensor.")
        if other.shape != self.matrix.shape:
            raise ValueError("Matrices must have the same dimensions for subtraction.")
        return self.matrix - other

    def multiply(self, other: torch.Tensor) -> torch.Tensor:
        """
        Multiplies the current matrix with another matrix.

        Args:
            other (torch.Tensor): A 2D tensor to multiply with.

        Returns:
            torch.Tensor: The result of the multiplication.

        Raises:
            ValueError: If the matrices have incompatible dimensions for multiplication.
        """
        if not isinstance(other, torch.Tensor):
            raise TypeError("The multiplier must be a PyTorch tensor.")
        if self.matrix.shape[1] != other.shape[0]:
            raise ValueError(
                "Number of columns in the first matrix must equal the number of rows in the second matrix."
            )
        return torch.matmul(self.matrix, other)

    def transpose(self) -> torch.Tensor:
        """
        Transposes the current matrix.

        Returns:
            torch.Tensor: The transposed matrix.
        """
        return self.matrix.t()

    def determinant(self) -> Union[float, torch.Tensor]:
        """
        Computes the determinant of the current matrix.

        Returns:
            Union[float, torch.Tensor]: The determinant of the matrix.

        Raises:
            ValueError: If the matrix is not square.
        """
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Determinant is defined only for square matrices.")
        return torch.det(self.matrix)

    def inverse(self) -> torch.Tensor:
        """
        Computes the inverse of the current matrix.

        Returns:
            torch.Tensor: The inverse matrix.

        Raises:
            ValueError: If the matrix is not square or is singular.
        """
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Inverse is defined only for square matrices.")
        det = self.determinant()
        if det == 0:
            raise ValueError("Singular matrix does not have an inverse.")
        return torch.inverse(self.matrix)

    def eigenvalues_eigenvectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the eigenvalues and eigenvectors of the current matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing eigenvalues and eigenvectors.

        Raises:
            ValueError: If the matrix is not square.
        """
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Eigenvalues and eigenvectors are defined only for square matrices.")
        eigen = torch.linalg.eig(self.matrix)
        return eigen.eigenvalues, eigen.eigenvectors

    def singular_value_decomposition(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs Singular Value Decomposition (SVD) on the current matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: U, S, V^T matrices from SVD.

        Raises:
            ValueError: If the matrix is empty.
        """
        if self.matrix.numel() == 0:
            raise ValueError("Cannot perform SVD on an empty matrix.")
        U, S, Vh = torch.linalg.svd(self.matrix)
        return U, S, Vh

    def rank(self) -> int:
        """
        Computes the rank of the current matrix.

        Returns:
            int: The rank of the matrix.
        """
        return torch.linalg.matrix_rank(self.matrix).item()

    def condition_number(self) -> float:
        """
        Computes the condition number of the current matrix.

        Returns:
            float: The condition number.

        Raises:
            ValueError: If the matrix is not square.
        """
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Condition number is defined only for square matrices.")
        return torch.linalg.cond(self.matrix).item()

    def trace(self) -> float:
        """
        Computes the trace of the current matrix.

        Returns:
            float: The trace of the matrix.

        Raises:
            ValueError: If the matrix is not square.
        """
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Trace is defined only for square matrices.")
        return torch.trace(self.matrix).item()

    def power(self, exponent: int) -> torch.Tensor:
        """
        Raises the current square matrix to a specified integer power.

        Args:
            exponent (int): The exponent to raise the matrix to.

        Returns:
            torch.Tensor: The resulting matrix after exponentiation.

        Raises:
            ValueError: If the matrix is not square.
            TypeError: If exponent is not an integer.
        """
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix power is defined only for square matrices.")
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer.")
        return torch.matrix_power(self.matrix, exponent)

    def trace_product(self, other: torch.Tensor) -> float:
        """
        Computes the trace of the product of the current matrix with another matrix.

        Args:
            other (torch.Tensor): A 2D tensor to multiply with.

        Returns:
            float: The trace of the product matrix.

        Raises:
            ValueError: If the matrices have incompatible dimensions.
        """
        product = self.multiply(other)
        return torch.trace(product).item()

    def kronecker_product(self, other: torch.Tensor) -> torch.Tensor:
        """
        Computes the Kronecker product of the current matrix with another matrix.

        Args:
            other (torch.Tensor): A 2D tensor to compute the Kronecker product with.

        Returns:
            torch.Tensor: The Kronecker product matrix.
        """
        return torch.kron(self.matrix, other)

    def hadamard_product(self, other: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hadamard (element-wise) product of the current matrix with another matrix.

        Args:
            other (torch.Tensor): A 2D tensor to compute the Hadamard product with.

        Returns:
            torch.Tensor: The Hadamard product matrix.

        Raises:
            ValueError: If the matrices are not of the same shape.
        """
        if other.shape != self.matrix.shape:
            raise ValueError("Matrices must have the same dimensions for Hadamard product.")
        return self.matrix * other

    def row_echelon_form(self) -> torch.Tensor:
        """
        Transforms the current matrix into its Row Echelon Form (REF).

        Returns:
            torch.Tensor: The REF of the matrix.
        """
        m = self.matrix.clone().float()
        rows, cols = m.shape
        lead = 0
        for r in range(rows):
            if lead >= cols:
                break
            i = r
            while m[i, lead] == 0:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        return m
            m[[r, i]] = m[[i, r]]
            lv = m[r, lead]
            m[r] = m[r] / lv
            for i in range(rows):
                if i != r:
                    lv = m[i, lead]
                    m[i] = m[i] - lv * m[r]
            lead += 1
        return m

    def reduced_row_echelon_form(self) -> torch.Tensor:
        """
        Transforms the current matrix into its Reduced Row Echelon Form (RREF).

        Returns:
            torch.Tensor: The RREF of the matrix.
        """
        ref = self.row_echelon_form()
        rows, cols = ref.shape
        for r in range(rows):
            # Find the leading 1
            row = ref[r]
            non_zero = torch.nonzero(row, as_tuple=False)
            if non_zero.numel() == 0:
                continue
            lead = non_zero[0].item()
            ref[r] = ref[r] / ref[r, lead]
            for i in range(rows):
                if i != r and ref[i, lead] != 0:
                    ref[i] = ref[i] - ref[i, lead] * ref[r]
        return ref

    def solve_linear_system(self, b: torch.Tensor) -> torch.Tensor:
        """
        Solves the linear system Ax = b using Gaussian elimination.

        Args:
            b (torch.Tensor): A 1D or 2D tensor representing the constants.

        Returns:
            torch.Tensor: The solution vector x.

        Raises:
            ValueError: If the system has no solution or infinitely many solutions.
        """
        if self.matrix.shape[0] != b.shape[0]:
            raise ValueError("The number of rows in A must match the size of b.")
        augmented = torch.cat((self.matrix, b.reshape(-1, 1)), dim=1)
        ref = MatrixManipulation(augmented).reduced_row_echelon_form()
        rows, cols = ref.shape
        solution = torch.zeros(self.matrix.shape[1])
        for r in range(rows):
            row = ref[r]
            if torch.all(row[:-1] == 0) and row[-1] != 0:
                raise ValueError("The system has no solution.")
            if torch.all(row[:-1] == 0) and row[-1] == 0:
                continue
            leading = torch.argmax(row[:-1] != 0).item()
            solution[leading] = row[-1]
        return solution

    def __str__(self) -> str:
        """
        Returns a string representation of the matrix.

        Returns:
            str: The matrix as a string.
        """
        return str(self.matrix)

    def __repr__(self) -> str:
        """
        Returns an unambiguous string representation of the matrix.

        Returns:
            str: The matrix representation.
        """
        return f"MatrixManipulation(matrix={self.matrix})"