"""
Trie Data Structure Implementation in Python

This module provides an advanced implementation of the Trie (Prefix Tree) data structure,
adhering to PEP-8 standards and utilizing Python's typing module for type hints.
The Trie supports insertion, search, deletion, and prefix-based retrieval operations
with comprehensive error handling and optimizations for performance and scalability.

"""

from typing import Dict, Optional, List


class TrieNode:
    """
    Represents a node in the Trie.

    Attributes:
        children (Dict[str, TrieNode]): Mapping from character to child TrieNode.
        is_end_of_word (bool): Flag indicating if the node represents the end of a word.
    """

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False


class Trie:
    """
    Trie (Prefix Tree) data structure supporting insertion, search, deletion, and prefix retrieval.

    Methods:
        insert(word: str) -> None
        search(word: str) -> bool
        delete(word: str) -> bool
        starts_with(prefix: str) -> List[str]
        _delete_recursive(node: TrieNode, word: str, depth: int) -> bool
    """

    def __init__(self) -> None:
        """Initialize the Trie with a root TrieNode."""
        self.root: TrieNode = TrieNode()

    def insert(self, word: str) -> None:
        """
        Insert a word into the Trie.

        Args:
            word (str): The word to insert.

        Raises:
            ValueError: If the word contains non-alphabetic characters.
        """
        if not word.isalpha():
            raise ValueError("Only alphabetic characters are allowed in words.")

        current_node = self.root
        for char in word.lower():
            if char not in current_node.children:
                current_node.children[char] = TrieNode()
            current_node = current_node.children[char]
        current_node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """
        Search for a word in the Trie.

        Args:
            word (str): The word to search for.

        Returns:
            bool: True if the word exists in the Trie, False otherwise.

        Raises:
            ValueError: If the word contains non-alphabetic characters.
        """
        if not word.isalpha():
            raise ValueError("Only alphabetic characters are allowed in words.")

        current_node = self.root
        for char in word.lower():
            if char not in current_node.children:
                return False
            current_node = current_node.children[char]
        return current_node.is_end_of_word

    def delete(self, word: str) -> bool:
        """
        Delete a word from the Trie.

        Args:
            word (str): The word to delete.

        Returns:
            bool: True if the word was successfully deleted, False if the word was not found.

        Raises:
            ValueError: If the word contains non-alphabetic characters.
        """
        if not word.isalpha():
            raise ValueError("Only alphabetic characters are allowed in words.")

        return self._delete_recursive(self.root, word.lower(), 0)

    def _delete_recursive(self, node: TrieNode, word: str, depth: int) -> bool:
        """
        Recursively delete a word from the Trie.

        Args:
            node (TrieNode): The current Trie node.
            word (str): The word to delete.
            depth (int): The current depth in the Trie.

        Returns:
            bool: True if the parent should delete the reference to this node, False otherwise.
        """
        if depth == len(word):
            if not node.is_end_of_word:
                return False  # Word not found.
            node.is_end_of_word = False
            return len(node.children) == 0  # If no children, node can be deleted.

        char = word[depth]
        child_node = node.children.get(char)
        if not child_node:
            return False  # Word not found.

        should_delete_child = self._delete_recursive(child_node, word, depth + 1)

        if should_delete_child:
            del node.children[char]
            return len(node.children) == 0 and not node.is_end_of_word

        return False

    def starts_with(self, prefix: str) -> List[str]:
        """
        Retrieve all words in the Trie that start with the given prefix.

        Args:
            prefix (str): The prefix to search for.

        Returns:
            List[str]: A list of words with the given prefix.

        Raises:
            ValueError: If the prefix contains non-alphabetic characters.
        """
        if not prefix.isalpha():
            raise ValueError("Only alphabetic characters are allowed in prefixes.")

        results: List[str] = []
        current_node = self.root
        prefix_lower = prefix.lower()

        for char in prefix_lower:
            if char not in current_node.children:
                return results  # Empty list; prefix not found.
            current_node = current_node.children[char]

        self._dfs(current_node, prefix_lower, results)
        return results

    def _dfs(self, node: TrieNode, path: str, results: List[str]) -> None:
        """
        Perform a depth-first search from the given node to find all words.

        Args:
            node (TrieNode): The current Trie node.
            path (str): The path representing the current word.
            results (List[str]): The list to append found words to.
        """
        if node.is_end_of_word:
            results.append(path)

        for char, child in node.children.items():
            self._dfs(child, path + char, results)