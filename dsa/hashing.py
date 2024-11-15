"""
hashing.py

An advanced implementation of Hashing data structures and algorithms in Python.
This module provides a comprehensive HashTable class with support for various
collision resolution strategies, efficient hash functions, and robustness features.

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from __future__ import annotations
from typing import Any, Optional, List, Tuple, TypeVar,Iterable
import torch

T = TypeVar('T')
V = TypeVar('V')


class HashTable:
    """
    A hash table implementation using separate chaining for collision resolution.

    Attributes:
        initial_capacity (int): The initial number of buckets in the hash table.
        load_factor_threshold (float): The load factor threshold to trigger resizing.
        buckets (List[List[Tuple[T, V]]]): The list of buckets containing key-value pairs.
        size (int): The current number of key-value pairs in the hash table.
    """

    def __init__(self, initial_capacity: int = 16, load_factor_threshold: float = 0.75) -> None:
        """
        Initializes the HashTable with a specified initial capacity and load factor threshold.

        Args:
            initial_capacity (int): The initial number of buckets.
            load_factor_threshold (float): The threshold to determine when to resize.
        """
        if initial_capacity <= 0:
            raise ValueError("Initial capacity must be a positive integer.")
        if not (0 < load_factor_threshold < 1):
            raise ValueError("Load factor threshold must be between 0 and 1.")

        self.initial_capacity: int = initial_capacity
        self.load_factor_threshold: float = load_factor_threshold
        self.buckets: List[List[Tuple[T, V]]] = [[] for _ in range(self.initial_capacity)]
        self.size: int = 0

    def _hash(self, key: T) -> int:
        """
        Computes the hash index for a given key using torch's tensor operations.

        Args:
            key (T): The key to hash.

        Returns:
            int: The computed hash index.
        """
        key_bytes = self._key_to_bytes(key)
        tensor = torch.tensor(list(key_bytes), dtype=torch.uint8)
        hash_tensor = torch.sum(tensor).item()
        index = hash_tensor % len(self.buckets)
        return index

    @staticmethod
    def _key_to_bytes(key: Any) -> bytes:
        """
        Converts a key to its byte representation.

        Args:
            key (Any): The key to convert.

        Returns:
            bytes: The byte representation of the key.
        """
        if isinstance(key, str):
            return key.encode('utf-8')
        elif isinstance(key, int):
            return key.to_bytes((key.bit_length() + 7) // 8, byteorder='big', signed=True)
        elif isinstance(key, bytes):
            return key
        else:
            raise TypeError("Unsupported key type.")

    def _resize(self) -> None:
        """
        Resizes the hash table by doubling its capacity and rehashing all existing keys.
        """
        old_buckets = self.buckets
        new_capacity = 2 * len(old_buckets)
        self.buckets = [[] for _ in range(new_capacity)]
        self.size = 0

        for bucket in old_buckets:
            for key, value in bucket:
                self.insert(key, value)

    def insert(self, key: T, value: V) -> None:
        """
        Inserts a key-value pair into the hash table. If the key already exists,
        its value is updated.

        Args:
            key (T): The key to insert.
            value (V): The value associated with the key.

        Raises:
            TypeError: If the key type is unsupported.
        """
        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)
                return

        bucket.append((key, value))
        self.size += 1

        if self.size / len(self.buckets) > self.load_factor_threshold:
            self._resize()

    def remove(self, key: T) -> bool:
        """
        Removes a key-value pair from the hash table.

        Args:
            key (T): The key to remove.

        Returns:
            bool: True if the key was found and removed, False otherwise.

        Raises:
            TypeError: If the key type is unsupported.
        """
        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self.size -= 1
                return True

        return False

    def get(self, key: T) -> Optional[V]:
        """
        Retrieves the value associated with a given key.

        Args:
            key (T): The key to search for.

        Returns:
            Optional[V]: The value if found, otherwise None.

        Raises:
            TypeError: If the key type is unsupported.
        """
        index = self._hash(key)
        bucket = self.buckets[index]

        for existing_key, value in bucket:
            if existing_key == key:
                return value

        return None

    def contains(self, key: T) -> bool:
        """
        Checks if the hash table contains a given key.

        Args:
            key (T): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.

        Raises:
            TypeError: If the key type is unsupported.
        """
        return self.get(key) is not None

    def __len__(self) -> int:
        """
        Returns the number of key-value pairs in the hash table.

        Returns:
            int: The size of the hash table.
        """
        return self.size

    def __iter__(self) -> Iterable[Tuple[T, V]]:
        """
        Allows iteration over key-value pairs in the hash table.

        Returns:
            Iterable[Tuple[T, V]]: An iterator over the hash table's items.
        """
        for bucket in self.buckets:
            for item in bucket:
                yield item

    def keys(self) -> List[T]:
        """
        Retrieves all keys in the hash table.

        Returns:
            List[T]: A list of keys.
        """
        return [key for key, _ in self]

    def values(self) -> List[V]:
        """
        Retrieves all values in the hash table.

        Returns:
            List[V]: A list of values.
        """
        return [value for _, value in self]

    def items(self) -> List[Tuple[T, V]]:
        """
        Retrieves all key-value pairs in the hash table.

        Returns:
            List[Tuple[T, V]]: A list of key-value pairs.
        """
        return list(self)


def main() -> None:
    """
    Demonstrates the usage of the HashTable class with various operations.
    """
    
    hash_table = HashTable()
    # Inserting key-value pairs
    hash_table.insert("apple", 1)
    hash_table.insert("banana", 2)
    hash_table.insert("cherry", 3)
    hash_table.insert(42, "answer")
    # hash_table.insert(3.14, "pi")  # This will raise TypeError


    # Retrieving values
    print("apple:", hash_table.get("apple"))
    print("banana:", hash_table.get("banana"))
    print("cherry:", hash_table.get("cherry"))
    print("42:", hash_table.get(42))
    # print("3.14:", hash_table.get(3.14))
    # Checking existence
    print("Contains 'apple':", hash_table.contains("apple"))
    print("Contains 'durian':", hash_table.contains("durian"))
    # Removing a key
    removed = hash_table.remove("banana")
    print("Removed 'banana':", removed)
    print("Contains 'banana':", hash_table.contains("banana"))
    # Iterating over items
    for key, value in hash_table.items():
        print(f"{key}: {value}")
    # Length of hash table
    print("HashTable size:", len(hash_table))


if __name__ == "__main__":
    main()