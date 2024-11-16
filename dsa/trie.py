"""
trie.py

A comprehensive implementation of the Trie (Prefix Tree) data structure in Python,
adhering to PEP-8 standards, utilizing type hints, and incorporating robust error handling.
This implementation covers basic to advanced functionalities, ensuring efficiency and scalability.
"""

from typing import Dict, Optional, List


class TrieNode:
    """
    Represents a single node within the Trie.
    Each node contains a dictionary of child nodes and a flag indicating
    whether it marks the end of a complete word.
    """

    def __init__(self) -> None:
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False


class Trie:
    """
    Trie (Prefix Tree) implementation supporting insertion, search, deletion,
    and autocomplete functionalities with optimized time and space complexities.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the Trie.

        Args:
            word (str): The word to be inserted.

        Raises:
            ValueError: If the input word is empty.
        """
        if not word:
            raise ValueError("Cannot insert an empty word into the Trie.")

        current_node = self.root
        for char in word:
            if char not in current_node.children:
                current_node.children[char] = TrieNode()
            current_node = current_node.children[char]
        current_node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """
        Searches for a word in the Trie.

        Args:
            word (str): The word to search for.

        Returns:
            bool: True if the word exists in the Trie, False otherwise.

        Raises:
            ValueError: If the input word is empty.
        """
        if not word:
            raise ValueError("Cannot search for an empty word in the Trie.")

        current_node = self.root
        for char in word:
            if char not in current_node.children:
                return False
            current_node = current_node.children[char]
        return current_node.is_end_of_word

    def delete(self, word: str) -> bool:
        """
        Deletes a word from the Trie.

        Args:
            word (str): The word to be deleted.

        Returns:
            bool: True if the word was successfully deleted, False if the word was not found.

        Raises:
            ValueError: If the input word is empty.
        """

        if not word:
            raise ValueError("Cannot delete an empty word from the Trie.")

        def _delete(current: TrieNode, word: str, index: int) -> bool:
            if index == len(word):
                if not current.is_end_of_word:
                    return False  # Word not found
                current.is_end_of_word = False
                return len(current.children) == 0  # If leaf node, it can be deleted

            char = word[index]
            if char not in current.children:
                return False  # Word not found

            should_delete_child = _delete(current.children[char], word, index + 1)

            if should_delete_child:
                del current.children[char]
                return len(current.children) == 0 and not current.is_end_of_word

            return False

        return _delete(self.root, word, 0)

    def starts_with(self, prefix: str) -> bool:
        """
        Checks if there is any word in the Trie that starts with the given prefix.

        Args:
            prefix (str): The prefix to check.

        Returns:
            bool: True if there is a word with the given prefix, False otherwise.

        Raises:
            ValueError: If the input prefix is empty.
        """
        if not prefix:
            raise ValueError("Prefix cannot be empty.")

        current_node = self.root
        for char in prefix:
            if char not in current_node.children:
                return False
            current_node = current_node.children[char]
        return True

    def autocomplete(self, prefix: str) -> List[str]:
        """
        Returns all words in the Trie that start with the given prefix.

        Args:
            prefix (str): The prefix to autocomplete.

        Returns:
            List[str]: A list of words starting with the prefix.

        Raises:
            ValueError: If the input prefix is empty.
        """
        if not prefix:
            raise ValueError("Prefix cannot be empty.")

        results: List[str] = []
        current_node = self.root

        for char in prefix:
            if char not in current_node.children:
                return results  # Empty list if prefix not in Trie
            current_node = current_node.children[char]

        def _dfs(node: TrieNode, path: List[str]) -> None:
            if node.is_end_of_word:
                results.append(prefix + ''.join(path))
            for char, child_node in node.children.items():
                path.append(char)
                _dfs(child_node, path)
                path.pop()

        _dfs(current_node, [])
        return results

    def serialize(self) -> Dict:
        """
        Serializes the Trie into a nested dictionary.

        Returns:
            Dict: The serialized Trie.
        """

        def _serialize_node(node: TrieNode) -> Dict:
            serialized = {"is_end_of_word": node.is_end_of_word, "children": {}}
            for char, child in node.children.items():
                serialized["children"][char] = _serialize_node(child)
            return serialized

        return _serialize_node(self.root)

    @staticmethod
    def deserialize(data: Dict) -> 'Trie':
        """
        Deserializes a nested dictionary into a Trie.

        Args:
            data (Dict): The serialized Trie data.

        Returns:
            Trie: The deserialized Trie object.

        Raises:
            ValueError: If the input data is invalid.
        """

        if not isinstance(data, dict):
            raise ValueError("Invalid data format for deserialization.")

        trie = Trie()

        def _deserialize_node(node: TrieNode, data_node: Dict) -> None:
            node.is_end_of_word = data_node.get("is_end_of_word", False)
            children = data_node.get("children", {})
            for char, child_data in children.items():
                node.children[char] = TrieNode()
                _deserialize_node(node.children[char], child_data)

        _deserialize_node(trie.root, data)
        return trie

    def __len__(self) -> int:
        """
        Returns the number of words stored in the Trie.

        Returns:
            int: The total number of words.
        """

        def _count(node: TrieNode) -> int:
            count = 1 if node.is_end_of_word else 0
            for child in node.children.values():
                count += _count(child)
            return count

        return _count(self.root)


def main() -> None:
    """
    Demonstrates the usage of the Trie data structure with various operations.
    """
    trie = Trie()
    words = ["hello", "helium", "hero", "heroplane", "help", "held", "heap"]

    # Inserting words
    for word in words:
        trie.insert(word)
        print(f"Inserted: {word}")

    # Searching for words
    search_terms = ["hero", "heroplane", "help", "helipad"]
    for term in search_terms:
        found = trie.search(term)
        print(f"Search '{term}': {'Found' if found else 'Not Found'}")

    # Autocomplete
    prefix = "he"
    completions = trie.autocomplete(prefix)
    print(f"Autocomplete for prefix '{prefix}': {completions}")

    # Deleting a word
    delete_word = "hero"
    deleted = trie.delete(delete_word)
    print(f"Deleted '{delete_word}': {'Successful' if deleted else 'Failed'}")

    # Search after deletion
    found = trie.search(delete_word)
    print(f"Search '{delete_word}' after deletion: {'Found' if found else 'Not Found'}")

    # Serialization
    serialized_trie = trie.serialize()
    print(f"Serialized Trie: {serialized_trie}")

    # Deserialization
    new_trie = Trie.deserialize(serialized_trie)
    print(f"Deserialized Trie contains 'hello': {new_trie.search('hello')}")

    # Trie size
    print(f"Total words in Trie: {len(trie)}")


if __name__ == "__main__":
    main()