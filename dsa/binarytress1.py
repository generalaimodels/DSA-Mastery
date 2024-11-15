"""
binary_trees.py

A comprehensive implementation of Binary Trees in Python, covering basic to advanced concepts.
Adheres to PEP-8 standards, utilizes type hints, and includes robust error handling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, List, Callable
import torch


class BinaryTreeError(Exception):
    """Custom exception class for BinaryTree operations."""
    pass


@dataclass
class TreeNode:
    """
    Represents a node in a binary tree.

    Attributes:
        value (Any): The value stored in the node.
        left (Optional[TreeNode]): Reference to the left child node.
        right (Optional[TreeNode]): Reference to the right child node.
    """
    value: Any
    left: Optional[TreeNode] = None
    right: Optional[TreeNode] = None


class BinaryTree:
    """
    A class representing a Binary Tree with various operations.

    Methods:
        insert(value): Inserts a value into the binary tree.
        inorder_traversal(): Returns a list of values from inorder traversal.
        preorder_traversal(): Returns a list of values from preorder traversal.
        postorder_traversal(): Returns a list of values from postorder traversal.
        search(value): Searches for a value in the binary tree.
        height(): Returns the height of the binary tree.
        is_balanced(): Checks if the binary tree is height-balanced.
        serialize(): Serializes the binary tree into a list.
        deserialize(data): Deserializes a list into a binary tree.
    """

    def __init__(self, root: Optional[TreeNode] = None) -> None:
        """
        Initializes the BinaryTree with an optional root node.

        Args:
            root (Optional[TreeNode]): The root node of the binary tree.
        """
        self.root = root

    def insert(self, value: Any) -> None:
        """
        Inserts a value into the binary tree following binary search tree rules.

        Args:
            value (Any): The value to be inserted.

        Raises:
            BinaryTreeError: If the tree is not a binary search tree.
        """
        if self.root is None:
            self.root = TreeNode(value)
            print(f"Inserted root node with value: {value}")
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node: TreeNode, value: Any) -> None:
        """
        Helper method to recursively insert a value into the binary tree.

        Args:
            node (TreeNode): The current node in traversal.
            value (Any): The value to be inserted.

        Raises:
            BinaryTreeError: If the tree is not a binary search tree.
        """
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
                print(f"Inserted {value} to the left of {node.value}")
            else:
                self._insert_recursive(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = TreeNode(value)
                print(f"Inserted {value} to the right of {node.value}")
            else:
                self._insert_recursive(node.right, value)
        else:
            raise BinaryTreeError(f"Duplicate value '{value}' not allowed in Binary Search Tree.")

    def inorder_traversal(self) -> List[Any]:
        """
        Performs inorder traversal of the binary tree.

        Returns:
            List[Any]: A list of values in inorder sequence.
        """
        result: List[Any] = []
        self._inorder_recursive(self.root, result)
        print(f"Inorder Traversal: {result}")
        return result

    def _inorder_recursive(self, node: Optional[TreeNode], result: List[Any]) -> None:
        """
        Helper method for inorder traversal.

        Args:
            node (Optional[TreeNode]): The current node.
            result (List[Any]): The list accumulating traversal results.
        """
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

    def preorder_traversal(self) -> List[Any]:
        """
        Performs preorder traversal of the binary tree.

        Returns:
            List[Any]: A list of values in preorder sequence.
        """
        result: List[Any] = []
        self._preorder_recursive(self.root, result)
        print(f"Preorder Traversal: {result}")
        return result

    def _preorder_recursive(self, node: Optional[TreeNode], result: List[Any]) -> None:
        """
        Helper method for preorder traversal.

        Args:
            node (Optional[TreeNode]): The current node.
            result (List[Any]): The list accumulating traversal results.
        """
        if node:
            result.append(node.value)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)

    def postorder_traversal(self) -> List[Any]:
        """
        Performs postorder traversal of the binary tree.

        Returns:
            List[Any]: A list of values in postorder sequence.
        """
        result: List[Any] = []
        self._postorder_recursive(self.root, result)
        print(f"Postorder Traversal: {result}")
        return result

    def _postorder_recursive(self, node: Optional[TreeNode], result: List[Any]) -> None:
        """
        Helper method for postorder traversal.

        Args:
            node (Optional[TreeNode]): The current node.
            result (List[Any]): The list accumulating traversal results.
        """
        if node:
            self._postorder_recursive(node.left, result)
            self._postorder_recursive(node.right, result)
            result.append(node.value)

    def search(self, value: Any) -> bool:
        """
        Searches for a value in the binary tree.

        Args:
            value (Any): The value to search for.

        Returns:
            bool: True if value is found, False otherwise.
        """
        found = self._search_recursive(self.root, value)
        print(f"Search for {value}: {'Found' if found else 'Not Found'}")
        return found

    def _search_recursive(self, node: Optional[TreeNode], value: Any) -> bool:
        """
        Helper method to recursively search for a value.

        Args:
            node (Optional[TreeNode]): The current node.
            value (Any): The value to search for.

        Returns:
            bool: True if found, False otherwise.
        """
        if node is None:
            return False
        if value == node.value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def height(self) -> int:
        """
        Calculates the height of the binary tree.

        Returns:
            int: The height of the tree.
        """
        h = self._height_recursive(self.root)
        print(f"Height of the tree: {h}")
        return h

    def _height_recursive(self, node: Optional[TreeNode]) -> int:
        """
        Helper method to calculate the height recursively.

        Args:
            node (Optional[TreeNode]): The current node.

        Returns:
            int: The height from the current node.
        """
        if node is None:
            return 0
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        return max(left_height, right_height) + 1

    def is_balanced(self) -> bool:
        """
        Checks if the binary tree is height-balanced.

        Returns:
            bool: True if balanced, False otherwise.
        """
        balanced, _ = self._check_balance(self.root)
        print(f"Is the tree balanced? {'Yes' if balanced else 'No'}")
        return balanced

    def _check_balance(self, node: Optional[TreeNode]) -> (bool, int):
        """
        Helper method to check balance and calculate height.

        Args:
            node (Optional[TreeNode]): The current node.

        Returns:
            tuple: (is_balanced, height)
        """
        if node is None:
            return True, 0
        left_balanced, left_height = self._check_balance(node.left)
        right_balanced, right_height = self._check_balance(node.right)
        current_balanced = (
            left_balanced and
            right_balanced and
            abs(left_height - right_height) <= 1
        )
        current_height = max(left_height, right_height) + 1
        return current_balanced, current_height

    def serialize(self) -> List[Any]:
        """
        Serializes the binary tree into a list using level-order traversal.

        Returns:
            List[Any]: The serialized list representation of the tree.
        """
        serialized: List[Any] = []
        self._serialize_level_order(self.root, serialized)
        print(f"Serialized tree: {serialized}")
        return serialized

    def _serialize_level_order(self, node: Optional[TreeNode], serialized: List[Any]) -> None:
        """
        Helper method to serialize using level-order traversal.

        Args:
            node (Optional[TreeNode]): The root node.
            serialized (List[Any]): The list to store serialized values.
        """
        if not node:
            return
        queue: List[Optional[TreeNode]] = [node]
        while queue:
            current = queue.pop(0)
            if current:
                serialized.append(current.value)
                queue.append(current.left)
                queue.append(current.right)
            else:
                serialized.append(None)

    def deserialize(self, data: List[Any]) -> BinaryTree:
        """
        Deserializes a list into a binary tree.

        Args:
            data (List[Any]): The list representation of the tree.

        Returns:
            BinaryTree: The deserialized binary tree.

        Raises:
            BinaryTreeError: If the data list is empty.
        """
        if not data:
            raise BinaryTreeError("Cannot deserialize an empty list.")
        iter_data = iter(data)
        root_value = next(iter_data)
        if root_value is None:
            return BinaryTree()
        root = TreeNode(root_value)
        tree = BinaryTree(root)
        queue: List[TreeNode] = [root]
        try:
            while queue:
                current = queue.pop(0)
                left_val = next(iter_data)
                if left_val is not None:
                    current.left = TreeNode(left_val)
                    queue.append(current.left)
                right_val = next(iter_data)
                if right_val is not None:
                    current.right = TreeNode(right_val)
                    queue.append(current.right)
        except StopIteration:
            pass
        print("Deserialization complete.")
        return tree

    def find_lowest_common_ancestor(self, value1: Any, value2: Any) -> Optional[TreeNode]:
        """
        Finds the lowest common ancestor of two values in the binary tree.

        Args:
            value1 (Any): The first value.
            value2 (Any): The second value.

        Returns:
            Optional[TreeNode]: The lowest common ancestor node.

        Raises:
            BinaryTreeError: If one or both values are not present in the tree.
        """
        if not self.search(value1) or not self.search(value2):
            raise BinaryTreeError("Both values must be present in the tree.")
        lca = self._find_lca_recursive(self.root, value1, value2)
        print(f"Lowest Common Ancestor of {value1} and {value2}: {lca.value if lca else 'None'}")
        return lca

    def _find_lca_recursive(self, node: Optional[TreeNode], value1: Any, value2: Any) -> Optional[TreeNode]:
        """
        Helper method to find the lowest common ancestor recursively.

        Args:
            node (Optional[TreeNode]): The current node.
            value1 (Any): The first value.
            value2 (Any): The second value.

        Returns:
            Optional[TreeNode]: The lowest common ancestor node.
        """
        if node is None:
            return None
        if node.value > value1 and node.value > value2:
            return self._find_lca_recursive(node.left, value1, value2)
        if node.value < value1 and node.value < value2:
            return self._find_lca_recursive(node.right, value1, value2)
        return node

    def diameter(self) -> int:
        """
        Calculates the diameter of the binary tree.

        Returns:
            int: The diameter of the tree.
        """
        diameter = self._diameter_recursive(self.root)
        print(f"Diameter of the tree: {diameter}")
        return diameter

    def _diameter_recursive(self, node: Optional[TreeNode]) -> int:
        """
        Helper method to calculate diameter and height simultaneously.

        Args:
            node (Optional[TreeNode]): The current node.

        Returns:
            int: The diameter at the current node.
        """
        def dfs(n: Optional[TreeNode]) -> (int, int):
            if n is None:
                return 0, 0
            left_diameter, left_height = dfs(n.left)
            right_diameter, right_height = dfs(n.right)
            current_height = max(left_height, right_height) + 1
            current_diameter = max(left_diameter, right_diameter, left_height + right_height)
            return current_diameter, current_height

        diameter, _ = dfs(node)
        return diameter

    def mirror(self) -> None:
        """
        Converts the binary tree into its mirror.
        """
        self._mirror_recursive(self.root)
        print("The tree has been mirrored.")

    def _mirror_recursive(self, node: Optional[TreeNode]) -> None:
        """
        Helper method to create a mirror of the binary tree.

        Args:
            node (Optional[TreeNode]): The current node.
        """
        if node:
            node.left, node.right = node.right, node.left
            self._mirror_recursive(node.left)
            self._mirror_recursive(node.right)


class AVLTree(BinaryTree):
    """
    A self-balancing Binary Search Tree (AVL Tree).

    Methods:
        insert(value): Inserts a value and rebalances the tree.
        delete(value): Deletes a value and rebalances the tree.
    """

    def insert(self, value: Any) -> None:
        """
        Inserts a value into the AVL tree and rebalances it.

        Args:
            value (Any): The value to be inserted.
        """
        self.root = self._insert_avl(self.root, value)
        print(f"Inserted {value} into AVL Tree.")

    def _insert_avl(self, node: Optional[TreeNode], value: Any) -> TreeNode:
        """
        Helper method to insert a value and maintain AVL balance.

        Args:
            node (Optional[TreeNode]): The current node.
            value (Any): The value to insert.

        Returns:
            TreeNode: The updated node after insertion and rotation.
        """
        if node is None:
            return TreeNode(value)
        if value < node.value:
            node.left = self._insert_avl(node.left, value)
        elif value > node.value:
            node.right = self._insert_avl(node.right, value)
        else:
            raise BinaryTreeError(f"Duplicate value '{value}' not allowed in AVL Tree.")

        node = self._rebalance(node)
        return node

    def _rebalance(self, node: TreeNode) -> TreeNode:
        """
        Rebalances the tree at the given node.

        Args:
            node (TreeNode): The current node.

        Returns:
            TreeNode: The balanced node.
        """
        balance_factor = self._get_balance(node)
        print(f"Rebalancing node {node.value}, Balance factor: {balance_factor}")

        # Left Heavy
        if balance_factor > 1:
            if self._get_balance(node.left) < 0:
                print(f"Left-Right case at node {node.value}")
                node.left = self._rotate_left(node.left)
            print(f"Right rotation at node {node.value}")
            return self._rotate_right(node)

        # Right Heavy
        if balance_factor < -1:
            if self._get_balance(node.right) > 0:
                print(f"Right-Left case at node {node.value}")
                node.right = self._rotate_right(node.right)
            print(f"Left rotation at node {node.value}")
            return self._rotate_left(node)

        return node

    def _get_height(self, node: Optional[TreeNode]) -> int:
        """
        Calculates the height of the node.

        Args:
            node (Optional[TreeNode]): The node.

        Returns:
            int: The height of the node.
        """
        if node is None:
            return 0
        return max(self._get_height(node.left), self._get_height(node.right)) + 1

    def _get_balance(self, node: Optional[TreeNode]) -> int:
        """
        Calculates the balance factor of the node.

        Args:
            node (Optional[TreeNode]): The node.

        Returns:
            int: The balance factor.
        """
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_left(self, z: TreeNode) -> TreeNode:
        """
        Performs a left rotation.

        Args:
            z (TreeNode): The root of the subtree to rotate.

        Returns:
            TreeNode: The new root after rotation.
        """
        y = z.right
        T2 = y.left if y else None

        if y is None:
            raise BinaryTreeError("Cannot perform left rotation without a right child.")

        y.left = z
        z.right = T2

        print(f"Left rotation: {z.value} -> {y.value}")
        return y

    def _rotate_right(self, y: TreeNode) -> TreeNode:
        """
        Performs a right rotation.

        Args:
            y (TreeNode): The root of the subtree to rotate.

        Returns:
            TreeNode: The new root after rotation.
        """
        x = y.left
        T2 = x.right if x else None

        if x is None:
            raise BinaryTreeError("Cannot perform right rotation without a left child.")

        x.right = y
        y.left = T2

        print(f"Right rotation: {y.value} -> {x.value}")
        return x

    def delete(self, value: Any) -> None:
        """
        Deletes a value from the AVL tree and rebalances it.

        Args:
            value (Any): The value to be deleted.
        """
        self.root = self._delete_avl(self.root, value)
        print(f"Deleted {value} from AVL Tree.")

    def _delete_avl(self, node: Optional[TreeNode], value: Any) -> Optional[TreeNode]:
        """
        Helper method to delete a value and maintain AVL balance.

        Args:
            node (Optional[TreeNode]): The current node.
            value (Any): The value to delete.

        Returns:
            Optional[TreeNode]: The updated node after deletion and rotation.
        """
        if node is None:
            raise BinaryTreeError(f"Value '{value}' not found in AVL Tree.")

        if value < node.value:
            node.left = self._delete_avl(node.left, value)
        elif value > node.value:
            node.right = self._delete_avl(node.right, value)
        else:
            # Node with only one child or no child
            if node.left is None:
                temp = node.right
                node = None
                return temp
            elif node.right is None:
                temp = node.left
                node = None
                return temp

            # Node with two children:
            temp = self._get_min_node(node.right)
            node.value = temp.value
            node.right = self._delete_avl(node.right, temp.value)

        if node is None:
            return node

        node = self._rebalance(node)
        return node

    def _get_min_node(self, node: TreeNode) -> TreeNode:
        """
        Finds the node with the minimum value in the subtree.

        Args:
            node (TreeNode): The root of the subtree.

        Returns:
            TreeNode: The node with the minimum value.
        """
        current = node
        while current.left is not None:
            current = current.left
        return current


class RedBlackTree(BinaryTree):
    """
    A Red-Black Tree implementation for self-balancing binary search trees.

    Note:
        This is a placeholder for Red-Black Tree implementation.
        Complete implementation requires additional properties and methods.
    """
    # Red-Black Tree implementation would go here
    pass


def main() -> None:
    """
    Demonstrates the usage of BinaryTree and AVLTree with various operations.
    """
    print("=== Binary Tree Operations ===")
    bt = BinaryTree()
    try:
        bt.insert(10)
        bt.insert(5)
        bt.insert(15)
        bt.insert(3)
        bt.insert(7)
        bt.insert(12)
        bt.insert(18)
    except BinaryTreeError as e:
        print(e)

    bt.inorder_traversal()
    bt.preorder_traversal()
    bt.postorder_traversal()
    bt.search(7)
    bt.search(20)
    bt.height()
    bt.is_balanced()
    serialized = bt.serialize()
    bt_mirror = BinaryTree()
    bt_mirror.deserialize(serialized)
    bt_mirror.mirror()
    bt.diameter()
    try:
        bt.find_lowest_common_ancestor(3, 7)
    except BinaryTreeError as e:
        print(e)

    print("\n=== AVL Tree Operations ===")
    avl = AVLTree()
    try:
        avl.insert(20)
        avl.insert(4)
        avl.insert(15)
        avl.insert(70)
        avl.insert(50)
        avl.insert(100)
    except BinaryTreeError as e:
        print(e)

    avl.inorder_traversal()
    avl.height()
    avl.is_balanced()
    avl.delete(70)
    avl.inorder_traversal()
    avl.height()
    avl.is_balanced()

    print("\n=== End of Demonstration ===")


if __name__ == "__main__":
    main()