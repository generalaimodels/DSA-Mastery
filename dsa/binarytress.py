#!/usr/bin/env python3
"""
Binary Trees Implementation with Comprehensive Operations

This module provides an advanced implementation of Binary Trees,
including various operations such as insertion, deletion, traversal,
searching, and balancing. It adheres to PEP-8 standards and uses
type hints for enhanced clarity and maintainability.

Author: generalmodelai-agent
Date: 2024-10-15
"""

from __future__ import annotations
from typing import Any, Callable, Generator, Optional, TypeVar, Union

T = TypeVar('T')


class BinaryTreeNode:
    """
    Represents a node in a binary tree.

    Attributes:
        value (Any): The value stored in the node.
        left (Optional[BinaryTreeNode]): Reference to the left child node.
        right (Optional[BinaryTreeNode]): Reference to the right child node.
    """

    def __init__(self, value: Any) -> None:
        self.value: Any = value
        self.left: Optional[BinaryTreeNode] = None
        self.right: Optional[BinaryTreeNode] = None

    def __repr__(self) -> str:
        return f"BinaryTreeNode({self.value})"


class BinaryTree:
    """
    Represents a Binary Tree with various operations.

    Supported Operations:
        - Insertion
        - Deletion
        - Searching
        - Traversals (In-order, Pre-order, Post-order, Level-order)
        - Checking if the tree is balanced
        - Balancing the tree
    """

    def __init__(self, root: Optional[BinaryTreeNode] = None) -> None:
        self.root: Optional[BinaryTreeNode] = root

    def insert(self, value: Any) -> None:
        """
        Inserts a value into the binary search tree.

        Args:
            value (Any): The value to insert.

        Raises:
            ValueError: If the value already exists in the tree.
        """
        if self.root is None:
            self.root = BinaryTreeNode(value)
            print(f"Inserted root node with value: {value}")
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node: BinaryTreeNode, value: Any) -> None:
        if value == node.value:
            raise ValueError(f"Duplicate value '{value}' not allowed in Binary Search Tree.")
        elif value < node.value:
            if node.left is None:
                node.left = BinaryTreeNode(value)
                print(f"Inserted node with value: {value} to the left of {node.value}")
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = BinaryTreeNode(value)
                print(f"Inserted node with value: {value} to the right of {node.value}")
            else:
                self._insert_recursive(node.right, value)

    def search(self, value: Any) -> Optional[BinaryTreeNode]:
        """
        Searches for a value in the binary search tree.

        Args:
            value (Any): The value to search for.

        Returns:
            Optional[BinaryTreeNode]: The node containing the value, if found.
        """
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node: Optional[BinaryTreeNode], value: Any) -> Optional[BinaryTreeNode]:
        if node is None:
            print(f"Value {value} not found in the tree.")
            return None
        if value == node.value:
            print(f"Value {value} found in the tree.")
            return node
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def delete(self, value: Any) -> None:
        """
        Deletes a value from the binary search tree.

        Args:
            value (Any): The value to delete.

        Raises:
            ValueError: If the value is not found in the tree.
        """
        self.root, deleted = self._delete_recursive(self.root, value)
        if deleted:
            print(f"Deleted node with value: {value}")
        else:
            raise ValueError(f"Value '{value}' not found in the tree.")

    def _delete_recursive(
        self, node: Optional[BinaryTreeNode], value: Any
    ) -> (Optional[BinaryTreeNode], bool):
        if node is None:
            return node, False

        if value < node.value:
            node.left, deleted = self._delete_recursive(node.left, value)
            return node, deleted
        elif value > node.value:
            node.right, deleted = self._delete_recursive(node.right, value)
            return node, deleted
        else:
            # Node with only one child or no child
            if node.left is None:
                temp = node.right
                node = None
                return temp, True
            elif node.right is None:
                temp = node.left
                node = None
                return temp, True

            # Node with two children: Get the inorder successor
            successor = self._min_value_node(node.right)
            node.value = successor.value
            node.right, _ = self._delete_recursive(node.right, successor.value)
            return node, True

    def _min_value_node(self, node: BinaryTreeNode) -> BinaryTreeNode:
        current = node
        while current.left is not None:
            current = current.left
        return current

    def inorder_traversal(self) -> Generator[Any, None, None]:
        """
        Generator for in-order traversal of the tree.

        Yields:
            Any: The next value in in-order traversal.
        """
        yield from self._inorder_recursive(self.root)

    def _inorder_recursive(self, node: Optional[BinaryTreeNode]) -> Generator[Any, None, None]:
        if node:
            yield from self._inorder_recursive(node.left)
            yield node.value
            yield from self._inorder_recursive(node.right)

    def preorder_traversal(self) -> Generator[Any, None, None]:
        """
        Generator for pre-order traversal of the tree.

        Yields:
            Any: The next value in pre-order traversal.
        """
        yield from self._preorder_recursive(self.root)

    def _preorder_recursive(self, node: Optional[BinaryTreeNode]) -> Generator[Any, None, None]:
        if node:
            yield node.value
            yield from self._preorder_recursive(node.left)
            yield from self._preorder_recursive(node.right)

    def postorder_traversal(self) -> Generator[Any, None, None]:
        """
        Generator for post-order traversal of the tree.

        Yields:
            Any: The next value in post-order traversal.
        """
        yield from self._postorder_recursive(self.root)

    def _postorder_recursive(self, node: Optional[BinaryTreeNode]) -> Generator[Any, None, None]:
        if node:
            yield from self._postorder_recursive(node.left)
            yield from self._postorder_recursive(node.right)
            yield node.value

    def level_order_traversal(self) -> Generator[Any, None, None]:
        """
        Generator for level-order (breadth-first) traversal of the tree.

        Yields:
            Any: The next value in level-order traversal.
        """
        if self.root is None:
            return

        queue = [self.root]
        while queue:
            current = queue.pop(0)
            yield current.value
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)

    def height(self) -> int:
        """
        Computes the height of the tree.

        Returns:
            int: The height of the tree.
        """
        return self._height_recursive(self.root)

    def _height_recursive(self, node: Optional[BinaryTreeNode]) -> int:
        if node is None:
            return -1  # Height of empty tree is -1
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        return max(left_height, right_height) + 1

    def is_balanced(self) -> bool:
        """
        Checks if the tree is balanced. A tree is balanced if the heights
        of the two child subtrees of any node never differ by more than one.

        Returns:
            bool: True if the tree is balanced, False otherwise.
        """

        def check_balance(node: Optional[BinaryTreeNode]) -> (bool, int):
            if node is None:
                return True, -1

            left_balanced, left_height = check_balance(node.left)
            if not left_balanced:
                return False, 0

            right_balanced, right_height = check_balance(node.right)
            if not right_balanced:
                return False, 0

            balanced = abs(left_height - right_height) <= 1
            height = max(left_height, right_height) + 1
            return balanced, height

        balanced, _ = check_balance(self.root)
        print(f"Tree is {'balanced' if balanced else 'not balanced'}.")
        return balanced

    def balance(self) -> None:
        """
        Balances the binary search tree to ensure it is height-balanced.
        This is achieved by performing an in-order traversal to get a sorted list
        of elements and then rebuilding the tree from the sorted list.
        """
        elements = list(self.inorder_traversal())
        print(f"Balancing the tree with elements: {elements}")
        self.root = self._build_balanced_tree(elements, 0, len(elements) - 1)

    def _build_balanced_tree(
        self, elements: list, start: int, end: int) -> Optional[BinaryTreeNode]:
        if start > end:
            return None

        mid = (start + end) // 2
        node = BinaryTreeNode(elements[mid])
        print(f"Creating node with value: {elements[mid]} from elements[{start}:{end}]")
        node.left = self._build_balanced_tree(elements, start, mid - 1)
        node.right = self._build_balanced_tree(elements, mid + 1, end)
        return node

    def __repr__(self) -> str:
        """
        Returns a string representation of the tree using level-order traversal.
        """
        nodes = list(self.level_order_traversal())
        return f"BinaryTree({nodes})"

    def __str__(self) -> str:
        """
        Returns a detailed string representation of the tree.
        """
        lines = []
        self._build_string(self.root, lines, "", True)
        return "\n".join(lines)

    def _build_string(
        self,
        node: Optional[BinaryTreeNode],
        lines: list,
        prefix: str,
        is_tail: bool
    ) -> None:
        if node:
            lines.append(prefix + ("└── " if is_tail else "├── ") + str(node.value))
            children = [child for child in [node.left, node.right] if child]
            for i, child in enumerate(children):
                is_last = i == (len(children) - 1)
                extension = "    " if is_tail else "│   "
                self._build_string(child, lines, prefix + extension, is_last)


if __name__ == "__main__":
    # Example Usage
    try:
        bt = BinaryTree()
        elements = [50, 30, 70, 20, 40, 60, 80]
        for elem in elements:
            bt.insert(elem)

        print("In-order Traversal:")
        print(list(bt.inorder_traversal()))

        print("\nPre-order Traversal:")
        print(list(bt.preorder_traversal()))

        print("\nPost-order Traversal:")
        print(list(bt.postorder_traversal()))

        print("\nLevel-order Traversal:")
        print(list(bt.level_order_traversal()))

        print(f"\nHeight of tree: {bt.height()}")

        print("\nIs the tree balanced?")
        bt.is_balanced()

        print("\nDeleting value 20...")
        bt.delete(20)
        print("In-order Traversal after deletion:")
        print(list(bt.inorder_traversal()))

        print("\nDeleting value 30...")
        bt.delete(30)
        print("In-order Traversal after deletion:")
        print(list(bt.inorder_traversal()))

        print("\nDeleting value 50...")
        bt.delete(50)
        print("In-order Traversal after deletion:")
        print(list(bt.inorder_traversal()))

        print("\nIs the tree balanced?")
        bt.is_balanced()

        print("\nBalancing the tree...")
        bt.balance()
        print("In-order Traversal after balancing:")
        print(list(bt.inorder_traversal()))
        print("Is the tree balanced now?")
        bt.is_balanced()

        print("\nTree Structure:")
        print(bt)

    except Exception as e:
        print(f"An error occurred: {e}")