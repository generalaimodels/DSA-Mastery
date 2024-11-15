"""
Binary Search Trees Implementation
==================================

This module provides a comprehensive implementation of Binary Search Trees (BSTs) in Python.
It covers basic operations such as insertion, deletion, and search, as well as advanced features
like tree traversal, balancing (AVL Trees), and serialization. The implementation adheres to
PEP-8 standards, utilizes type hints for clarity, and includes robust error handling to ensure
reliability and maintainability.

Author: OpenAI
Date: 2023-10-05
"""

from __future__ import annotations
from typing import Any, Optional, Callable, List, Generator
import torch


class BSTNode:
    """
    A node in the Binary Search Tree.

    Attributes:
        key (Any): The value of the node.
        left (Optional[BSTNode]): Left child node.
        right (Optional[BSTNode]): Right child node.
        height (int): Height of the node for AVL balancing.
    """

    def __init__(self, key: Any) -> None:
        self.key: Any = key
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None
        self.height: int = 1  # For AVL Tree balancing


class BinarySearchTree:
    """
    A Binary Search Tree (BST) implementation supporting insertion, deletion, search, and traversal.

    Attributes:
        root (Optional[BSTNode]): Root node of the BST.
    """

    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None

    def insert(self, key: Any) -> None:
        """
        Insert a key into the BST.

        Args:
            key (Any): The key to insert.

        Raises:
            ValueError: If the key already exists in the BST.
        """
        try:
            self.root = self._insert_recursive(self.root, key)
        except ValueError as e:
            print(f"Insertion Error: {e}")

    def _insert_recursive(self, node: Optional[BSTNode], key: Any) -> BSTNode:
        if node is None:
            return BSTNode(key)

        if key < node.key:
            node.left = self._insert_recursive(node.left, key)
        elif key > node.key:
            node.right = self._insert_recursive(node.right, key)
        else:
            raise ValueError(f"Key '{key}' already exists.")

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        return self._balance(node)

    def delete(self, key: Any) -> None:
        """
        Delete a key from the BST.

        Args:
            key (Any): The key to delete.

        Raises:
            KeyError: If the key is not found in the BST.
        """
        try:
            self.root = self._delete_recursive(self.root, key)
        except KeyError as e:
            print(f"Deletion Error: {e}")

    def _delete_recursive(self, node: Optional[BSTNode], key: Any) -> Optional[BSTNode]:
        if node is None:
            raise KeyError(f"Key '{key}' not found.")

        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            temp = self._get_min_node(node.right)
            node.key = temp.key
            node.right = self._delete_recursive(node.right, temp.key)

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        return self._balance(node)

    def search(self, key: Any) -> bool:
        """
        Search for a key in the BST.

        Args:
            key (Any): The key to search for.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node: Optional[BSTNode], key: Any) -> bool:
        if node is None:
            return False
        if key == node.key:
            return True
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)

    def inorder_traversal(self) -> Generator[Any, None, None]:
        """
        In-order traversal of the BST.

        Yields:
            Any: The next key in in-order sequence.
        """
        yield from self._inorder_recursive(self.root)

    def _inorder_recursive(self, node: Optional[BSTNode]) -> Generator[Any, None, None]:
        if node:
            yield from self._inorder_recursive(node.left)
            yield node.key
            yield from self._inorder_recursive(node.right)

    def preorder_traversal(self) -> Generator[Any, None, None]:
        """
        Pre-order traversal of the BST.

        Yields:
            Any: The next key in pre-order sequence.
        """
        yield from self._preorder_recursive(self.root)

    def _preorder_recursive(self, node: Optional[BSTNode]) -> Generator[Any, None, None]:
        if node:
            yield node.key
            yield from self._preorder_recursive(node.left)
            yield from self._preorder_recursive(node.right)

    def postorder_traversal(self) -> Generator[Any, None, None]:
        """
        Post-order traversal of the BST.

        Yields:
            Any: The next key in post-order sequence.
        """
        yield from self._postorder_recursive(self.root)

    def _postorder_recursive(self, node: Optional[BSTNode]) -> Generator[Any, None, None]:
        if node:
            yield from self._postorder_recursive(node.left)
            yield from self._postorder_recursive(node.right)
            yield node.key

    def get_height(self) -> int:
        """
        Get the height of the BST.

        Returns:
            int: The height of the BST.
        """
        return self._get_height(self.root)

    def is_balanced(self) -> bool:
        """
        Check if the BST is balanced (AVL).

        Returns:
            bool: True if balanced, False otherwise.
        """
        return self._check_balance(self.root) != -1

    def to_list_inorder(self) -> List[Any]:
        """
        Convert the BST to a list using in-order traversal.

        Returns:
            List[Any]: The list of keys in in-order.
        """
        return list(self.inorder_traversal())

    def to_list_preorder(self) -> List[Any]:
        """
        Convert the BST to a list using pre-order traversal.

        Returns:
            List[Any]: The list of keys in pre-order.
        """
        return list(self.preorder_traversal())

    def to_list_postorder(self) -> List[Any]:
        """
        Convert the BST to a list using post-order traversal.

        Returns:
            List[Any]: The list of keys in post-order.
        """
        return list(self.postorder_traversal())

    def serialize(self) -> List[Any]:
        """
        Serialize the BST to a list using level-order traversal.

        Returns:
            List[Any]: The serialized list of keys.
        """
        if not self.root:
            return []
        serialized: List[Any] = []
        queue: List[BSTNode] = [self.root]
        while queue:
            current = queue.pop(0)
            serialized.append(current.key)
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        return serialized

    def deserialize(self, data: List[Any]) -> None:
        """
        Deserialize a list to reconstruct the BST using level-order insertion.

        Args:
            data (List[Any]): The list of keys to deserialize.
        """
        if not data:
            self.root = None
            return
        self.root = BSTNode(data[0])
        queue: List[BSTNode] = [self.root]
        i = 1
        while i < len(data):
            current = queue.pop(0)
            if i < len(data):
                current.left = BSTNode(data[i])
                queue.append(current.left)
                i += 1
            if i < len(data):
                current.right = BSTNode(data[i])
                queue.append(current.right)
                i += 1
        # Rebalance the tree after deserialization
        self.root = self._rebalance_from_sorted(self.to_list_inorder())

    def _rebalance_from_sorted(self, sorted_keys: List[Any]) -> Optional[BSTNode]:
        if not sorted_keys:
            return None
        mid = len(sorted_keys) // 2
        node = BSTNode(sorted_keys[mid])
        node.left = self._rebalance_from_sorted(sorted_keys[:mid])
        node.right = self._rebalance_from_sorted(sorted_keys[mid + 1:])
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        return node

    # AVL Tree specific methods for balancing

    def _get_height(self, node: Optional[BSTNode]) -> int:
        return node.height if node else 0

    def _get_balance(self, node: Optional[BSTNode]) -> int:
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _balance(self, node: BSTNode) -> BSTNode:
        balance_factor = self._get_balance(node)

        # Left heavy
        if balance_factor > 1:
            if self._get_balance(node.left) < 0:
                node.left = self._left_rotate(node.left)
            return self._right_rotate(node)

        # Right heavy
        if balance_factor < -1:
            if self._get_balance(node.right) > 0:
                node.right = self._right_rotate(node.right)
            return self._left_rotate(node)

        return node

    def _left_rotate(self, z: BSTNode) -> BSTNode:
        y = z.right
        if y is None:
            return z  # Cannot perform rotate

        T2 = y.left
        y.left = z
        z.right = T2

        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def _right_rotate(self, y: BSTNode) -> BSTNode:
        x = y.left
        if x is None:
            return y  # Cannot perform rotate

        T2 = x.right
        x.right = y
        y.left = T2

        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))

        return x

    def _get_min_node(self, node: BSTNode) -> BSTNode:
        current = node
        while current.left is not None:
            current = current.left
        return current

    def _check_balance(self, node: Optional[BSTNode]) -> int:
        if node is None:
            return 0
        left_height = self._check_balance(node.left)
        if left_height == -1:
            return -1
        right_height = self._check_balance(node.right)
        if right_height == -1:
            return -1
        if abs(left_height - right_height) > 1:
            return -1
        return max(left_height, right_height) + 1

    # Advanced Features

    def kth_smallest(self, k: int) -> Any:
        """
        Find the k-th smallest element in the BST.

        Args:
            k (int): The order of the smallest element to find.

        Returns:
            Any: The k-th smallest key.

        Raises:
            IndexError: If k is out of bounds.
        """
        elements = self.to_list_inorder()
        if k < 1 or k > len(elements):
            raise IndexError("k is out of bounds")
        return elements[k - 1]

    def lowest_common_ancestor(self, key1: Any, key2: Any) -> Optional[Any]:
        """
        Find the lowest common ancestor of two keys in the BST.

        Args:
            key1 (Any): The first key.
            key2 (Any): The second key.

        Returns:
            Optional[Any]: The lowest common ancestor key, or None if not found.
        """
        return self._lca_recursive(self.root, key1, key2)

    def _lca_recursive(self, node: Optional[BSTNode], key1: Any, key2: Any) -> Optional[Any]:
        if node is None:
            return None

        if key1 < node.key and key2 < node.key:
            return self._lca_recursive(node.left, key1, key2)
        if key1 > node.key and key2 > node.key:
            return self._lca_recursive(node.right, key1, key2)

        if self.search(key1) and self.search(key2):
            return node.key
        return None

    def range_query(self, low: Any, high: Any) -> List[Any]:
        """
        Find all keys within the range [low, high].

        Args:
            low (Any): The lower bound.
            high (Any): The upper bound.

        Returns:
            List[Any]: List of keys within the range.
        """
        result: List[Any] = []
        self._range_query_recursive(self.root, low, high, result)
        return result

    def _range_query_recursive(
        self, node: Optional[BSTNode], low: Any, high: Any, result: List[Any]
    ) -> None:
        if node is None:
            return
        if low < node.key:
            self._range_query_recursive(node.left, low, high, result)
        if low <= node.key <= high:
            result.append(node.key)
        if node.key < high:
            self._range_query_recursive(node.right, low, high, result)

    def find_inorder_successor(self, key: Any) -> Optional[Any]:
        """
        Find the in-order successor of a given key in the BST.

        Args:
            key (Any): The key to find the successor for.

        Returns:
            Optional[Any]: The in-order successor key, or None if not found.
        """
        successor: Optional[BSTNode] = None
        current = self.root
        while current:
            if key < current.key:
                successor = current
                current = current.left
            elif key > current.key:
                current = current.right
            else:
                if current.right:
                    successor = self._get_min_node(current.right)
                break
        return successor.key if successor else None

    def find_inorder_predecessor(self, key: Any) -> Optional[Any]:
        """
        Find the in-order predecessor of a given key in the BST.

        Args:
            key (Any): The key to find the predecessor for.

        Returns:
            Optional[Any]: The in-order predecessor key, or None if not found.
        """
        predecessor: Optional[BSTNode] = None
        current = self.root
        while current:
            if key > current.key:
                predecessor = current
                current = current.right
            elif key < current.key:
                current = current.left
            else:
                if current.left:
                    predecessor = self._get_max_node(current.left)
                break
        return predecessor.key if predecessor else None

    def _get_max_node(self, node: BSTNode) -> BSTNode:
        current = node
        while current.right is not None:
            current = current.right
        return current

    # Additional Utility Methods

    def is_bst_util(self, node: Optional[BSTNode], left: Any, right: Any) -> bool:
        if node is None:
            return True
        if not (left < node.key < right):
            return False
        return (
            self.is_bst_util(node.left, left, node.key)
            and self.is_bst_util(node.right, node.key, right)
        )

    def is_bst(self) -> bool:
        """
        Verify if the tree is a valid Binary Search Tree.

        Returns:
            bool: True if valid BST, False otherwise.
        """
        return self.is_bst_util(self.root, float('-inf'), float('inf'))

    def visualize(self) -> None:
        """
        Visualize the BST using ASCII representation.
        """
        lines, *_ = self._display_aux(self.root)
        for line in lines:
            print(line)

    def _display_aux(
        self, node: Optional[BSTNode]
    ) -> tuple[List[str], int, int, int]:
        if node is None:
            return ["<empty tree>"], 0, 0, 0

        if node.right is None and node.left is None:
            line = f"{node.key}"
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        if node.right is None:
            lines, n, p, x = self._display_aux(node.left)
            s = f"{node.key}"
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return (
                [first_line, second_line] + shifted_lines,
                n + u,
                p + 2,
                n + u // 2,
            )

        if node.left is None:
            lines, n, p, x = self._display_aux(node.right)
            s = f"{node.key}"
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return (
                [first_line, second_line] + shifted_lines,
                n + u,
                p + 2,
                u // 2,
            )

        left, n, p, x = self._display_aux(node.left)
        right, m, q, y = self._display_aux(node.right)
        s = f"{node.key}"
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [a + u * ' ' + b for a, b in zipped_lines]
        return [first_line, second_line] + lines, n + m + u, max(p, q) + 2, n + u // 2


# Example Usage
if __name__ == "__main__":
    bst = BinarySearchTree()
    keys = [50, 30, 70, 20, 40, 60, 80]

    print("Inserting keys:")
    for key in keys:
        print(f"Insert {key}")
        bst.insert(key)

    print("\nBST In-order Traversal:")
    print(bst.to_list_inorder())

    print("\nBST Pre-order Traversal:")
    print(bst.to_list_preorder())

    print("\nBST Post-order Traversal:")
    print(bst.to_list_postorder())

    print("\nVisual representation of BST:")
    bst.visualize()

    print("\nSearching for key 40:")
    print("Found" if bst.search(40) else "Not Found")

    print("\nDeleting key 20:")
    bst.delete(20)
    print("In-order after deletion:", bst.to_list_inorder())

    print("\nBST is balanced:", bst.is_balanced())

    print("\nLowest Common Ancestor of 40 and 60:")
    print(bst.lowest_common_ancestor(40, 60))

    print("\nRange Query [30, 70]:")
    print(bst.range_query(30, 70))

    print("\nIn-order Successor of 40:")
    print(bst.find_inorder_successor(40))

    print("\nIn-order Predecessor of 40:")
    print(bst.find_inorder_predecessor(40))

    print("\nSerializing BST:")
    serialized = bst.serialize()
    print(serialized)

    print("\nDeserializing BST from serialized data:")
    bst_new = BinarySearchTree()
    bst_new.deserialize(serialized)
    print("In-order of deserialized BST:", bst_new.to_list_inorder())

    print("\nBST is valid:", bst_new.is_bst())