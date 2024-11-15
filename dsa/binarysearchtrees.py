"""
Binary Search Trees (BST) Implementation in Python

This module provides a comprehensive implementation of Binary Search Trees (BST),
covering basic to advanced functionalities. It adheres to PEP-8 standards, utilizes
type hints for clarity, and incorporates robust error handling to ensure reliability
and scalability.

Features:
- Node creation and management
- Insertion, deletion, and search operations
- Traversal algorithms (in-order, pre-order, post-order, level-order)
- Advanced functionalities like finding the lowest common ancestor and tree balancing
"""

from __future__ import annotations
from typing import Any, Optional, List, Generator, Tuple
import torch


class BSTNode:
    """
    Represents a node in a Binary Search Tree.
    """
    def __init__(self, key: Any, value: Any = None) -> None:
        """
        Initializes a BST node with a key, optional value, and child pointers.

        Args:
            key (Any): The key of the node.
            value (Any, optional): The value associated with the key. Defaults to None.
        """
        self.key: Any = key
        self.value: Any = value
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None
        self.parent: Optional[BSTNode] = None

    def __repr__(self) -> str:
        return f"BSTNode(key={self.key}, value={self.value})"


class BinarySearchTree:
    """
    A Binary Search Tree (BST) implementation with comprehensive functionalities.
    """
    def __init__(self) -> None:
        """
        Initializes an empty Binary Search Tree.
        """
        self.root: Optional[BSTNode] = None

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Inserts a key-value pair into the BST.

        Args:
            key (Any): The key to insert.
            value (Any, optional): The value associated with the key. Defaults to None.

        Raises:
            ValueError: If the key already exists in the BST.
        """
        if self.root is None:
            self.root = BSTNode(key, value)
            print(f"Inserted root: {self.root}")
            return

        current = self.root
        while True:
            if key < current.key:
                if current.left is None:
                    current.left = BSTNode(key, value)
                    current.left.parent = current
                    print(f"Inserted {current.left} to the left of {current}")
                    return
                current = current.left
            elif key > current.key:
                if current.right is None:
                    current.right = BSTNode(key, value)
                    current.right.parent = current
                    print(f"Inserted {current.right} to the right of {current}")
                    return
                current = current.right
            else:
                raise ValueError(f"Key {key} already exists in the BST.")

    def search(self, key: Any) -> Optional[BSTNode]:
        """
        Searches for a node with the given key.

        Args:
            key (Any): The key to search for.

        Returns:
            Optional[BSTNode]: The node with the specified key, or None if not found.
        """
        current = self.root
        while current:
            if key == current.key:
                print(f"Found node: {current}")
                return current
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        print(f"Key {key} not found in the BST.")
        return None

    def delete(self, key: Any) -> None:
        """
        Deletes a node with the specified key from the BST.

        Args:
            key (Any): The key of the node to delete.

        Raises:
            KeyError: If the key is not found in the BST.
        """
        node_to_delete = self.search(key)
        if node_to_delete is None:
            raise KeyError(f"Key {key} not found in the BST.")

        print(f"Deleting node: {node_to_delete}")

        # Case 1: Node has no children
        if node_to_delete.left is None and node_to_delete.right is None:
            self._transplant(node_to_delete, None)

        # Case 2: Node has one child
        elif node_to_delete.left is None:
            self._transplant(node_to_delete, node_to_delete.right)
        elif node_to_delete.right is None:
            self._transplant(node_to_delete, node_to_delete.left)

        # Case 3: Node has two children
        else:
            successor = self._minimum(node_to_delete.right)
            print(f"Successor of {node_to_delete} is {successor}")
            if successor.parent != node_to_delete:
                self._transplant(successor, successor.right)
                successor.right = node_to_delete.right
                if successor.right:
                    successor.right.parent = successor
            self._transplant(node_to_delete, successor)
            successor.left = node_to_delete.left
            if successor.left:
                successor.left.parent = successor

    def _transplant(self, u: BSTNode, v: Optional[BSTNode]) -> None:
        """
        Replaces the subtree rooted at node u with the subtree rooted at node v.

        Args:
            u (BSTNode): The node to be replaced.
            v (Optional[BSTNode]): The node to replace with.
        """
        if u.parent is None:
            self.root = v
            print(f"Replaced root {u} with {v}")
        elif u == u.parent.left:
            u.parent.left = v
            print(f"Replaced left child of {u.parent} with {v}")
        else:
            u.parent.right = v
            print(f"Replaced right child of {u.parent} with {v}")
        if v:
            v.parent = u.parent

    def _minimum(self, node: BSTNode) -> BSTNode:
        """
        Finds the node with the minimum key in the subtree rooted at the given node.

        Args:
            node (BSTNode): The root of the subtree.

        Returns:
            BSTNode: The node with the minimum key.
        """
        current = node
        while current.left:
            current = current.left
        return current

    def inorder_traversal(self) -> Generator[BSTNode, None, None]:
        """
        Performs in-order traversal of the BST.

        Yields:
            BSTNode: The next node in in-order sequence.
        """
        yield from self._inorder_traversal_recursive(self.root)

    def _inorder_traversal_recursive(self, node: Optional[BSTNode]) -> Generator[BSTNode, None, None]:
        """
        Recursive helper for in-order traversal.

        Args:
            node (Optional[BSTNode]): The current node.

        Yields:
            BSTNode: The next node in in-order sequence.
        """
        if node:
            yield from self._inorder_traversal_recursive(node.left)
            print(f"In-order visiting: {node}")
            yield node
            yield from self._inorder_traversal_recursive(node.right)

    def preorder_traversal(self) -> Generator[BSTNode, None, None]:
        """
        Performs pre-order traversal of the BST.

        Yields:
            BSTNode: The next node in pre-order sequence.
        """
        yield from self._preorder_traversal_recursive(self.root)

    def _preorder_traversal_recursive(self, node: Optional[BSTNode]) -> Generator[BSTNode, None, None]:
        """
        Recursive helper for pre-order traversal.

        Args:
            node (Optional[BSTNode]): The current node.

        Yields:
            BSTNode: The next node in pre-order sequence.
        """
        if node:
            print(f"Pre-order visiting: {node}")
            yield node
            yield from self._preorder_traversal_recursive(node.left)
            yield from self._preorder_traversal_recursive(node.right)

    def postorder_traversal(self) -> Generator[BSTNode, None, None]:
        """
        Performs post-order traversal of the BST.

        Yields:
            BSTNode: The next node in post-order sequence.
        """
        yield from self._postorder_traversal_recursive(self.root)

    def _postorder_traversal_recursive(self, node: Optional[BSTNode]) -> Generator[BSTNode, None, None]:
        """
        Recursive helper for post-order traversal.

        Args:
            node (Optional[BSTNode]): The current node.

        Yields:
            BSTNode: The next node in post-order sequence.
        """
        if node:
            yield from self._postorder_traversal_recursive(node.left)
            yield from self._postorder_traversal_recursive(node.right)
            print(f"Post-order visiting: {node}")
            yield node

    def level_order_traversal(self) -> Generator[BSTNode, None, None]:
        """
        Performs level-order (breadth-first) traversal of the BST.

        Yields:
            BSTNode: The next node in level-order sequence.
        """
        if self.root is None:
            return

        queue: List[BSTNode] = [self.root]
        while queue:
            current = queue.pop(0)
            print(f"Level-order visiting: {current}")
            yield current
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)

    def find_lowest_common_ancestor(self, key1: Any, key2: Any) -> Optional[BSTNode]:
        """
        Finds the lowest common ancestor (LCA) of two nodes with the given keys.

        Args:
            key1 (Any): The key of the first node.
            key2 (Any): The key of the second node.

        Returns:
            Optional[BSTNode]: The LCA node, or None if either key is not found.
        """
        current = self.root
        while current:
            if key1 < current.key and key2 < current.key:
                current = current.left
            elif key1 > current.key and key2 > current.key:
                current = current.right
            else:
                print(f"Lowest Common Ancestor of {key1} and {key2} is {current}")
                return current
        return None

    def is_balanced(self) -> bool:
        """
        Checks if the BST is balanced.

        Returns:
            bool: True if balanced, False otherwise.
        """
        def check_balance(node: Optional[BSTNode]) -> Tuple[bool, int]:
            if node is None:
                return True, 0
            left_balanced, left_height = check_balance(node.left)
            right_balanced, right_height = check_balance(node.right)
            balanced = (
                left_balanced and
                right_balanced and
                abs(left_height - right_height) <= 1
            )
            height = max(left_height, right_height) + 1
            return balanced, height

        balanced, _ = check_balance(self.root)
        print(f"BST is {'balanced' if balanced else 'not balanced'}.")
        return balanced

    def rebalance(self) -> None:
        """
        Rebalances the BST to ensure optimal operations.

        Raises:
            ValueError: If the BST is empty.
        """
        nodes = list(self.inorder_traversal())
        if not nodes:
            raise ValueError("Cannot rebalance an empty BST.")

        def build_balanced_tree(sorted_nodes: List[BSTNode]) -> Optional[BSTNode]:
            if not sorted_nodes:
                return None
            mid_idx = len(sorted_nodes) // 2
            node = sorted_nodes[mid_idx]
            node.left = build_balanced_tree(sorted_nodes[:mid_idx])
            node.right = build_balanced_tree(sorted_nodes[mid_idx + 1:])
            if node.left:
                node.left.parent = node
            if node.right:
                node.right.parent = node
            return node

        self.root = build_balanced_tree(nodes)
        print("BST has been rebalanced.")

    def height(self) -> int:
        """
        Calculates the height of the BST.

        Returns:
            int: The height of the BST.
        """
        def node_height(node: Optional[BSTNode]) -> int:
            if node is None:
                return -1
            return 1 + max(node_height(node.left), node_height(node.right))

        h = node_height(self.root)
        print(f"Height of BST: {h}")
        return h

    def to_list_inorder(self) -> List[Any]:
        """
        Converts the BST to a list using in-order traversal.

        Returns:
            List[Any]: The list of keys in in-order sequence.
        """
        return [node.key for node in self.inorder_traversal()]

    def validate_bst(self) -> bool:
        """
        Validates whether the tree satisfies BST properties.

        Returns:
            bool: True if valid BST, False otherwise.
        """
        def validate(node: Optional[BSTNode], low: Any, high: Any) -> bool:
            if node is None:
                return True
            if not (low < node.key < high):
                print(f"Validation failed at node: {node}")
                return False
            return validate(node.left, low, node.key) and validate(node.right, node.key, high)

        is_valid = validate(self.root, float('-inf'), float('inf'))
        print(f"BST validation result: {'valid' if is_valid else 'invalid'}.")
        return is_valid


def main() -> None:
    """
    Demonstrates the usage of the BinarySearchTree class with various operations.
    """
    bst = BinarySearchTree()

    # Insert nodes
    try:
        for key in [50, 30, 70, 20, 40, 60, 80]:
            bst.insert(key)
    except ValueError as e:
        print(e)

    # Search for nodes
    bst.search(40)
    bst.search(90)

    # In-order traversal
    print("In-order Traversal:")
    for node in bst.inorder_traversal():
        print(node.key, end=' ')
    print()

    # Pre-order traversal
    print("Pre-order Traversal:")
    for node in bst.preorder_traversal():
        print(node.key, end=' ')
    print()

    # Post-order traversal
    print("Post-order Traversal:")
    for node in bst.postorder_traversal():
        print(node.key, end=' ')
    print()

    # Level-order traversal
    print("Level-order Traversal:")
    for node in bst.level_order_traversal():
        print(node.key, end=' ')
    print()

    # Find Lowest Common Ancestor
    bst.find_lowest_common_ancestor(20, 40)
    bst.find_lowest_common_ancestor(20, 90)

    # Check if BST is balanced
    bst.is_balanced()

    # Delete a node
    try:
        bst.delete(20)
        bst.delete(30)
        bst.delete(50)
    except KeyError as e:
        print(e)

    # Validate BST
    bst.validate_bst()

    # Rebalance the BST
    bst.rebalance()

    # Check height
    bst.height()

    # Convert to list (in-order)
    print("BST as list (in-order):", bst.to_list_inorder())


if __name__ == "__main__":
    main()