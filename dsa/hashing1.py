# hashing.py

from typing import Any, List, Optional, Tuple


class HashTable:
    """
    A Hash Table implementation using separate chaining for collision resolution.
    Supports dynamic resizing to maintain efficient operations.
    """

    def __init__(self, initial_capacity: int = 16, load_factor: float = 0.75) -> None:
        """
        Initializes the hash table with the given initial capacity and load factor.

        Args:
            initial_capacity (int): The initial number of buckets in the hash table.
            load_factor (float): The threshold ratio to trigger resizing.
        """
        if initial_capacity <= 0:
            raise ValueError("Initial capacity must be a positive integer.")
        if not (0 < load_factor < 1):
            raise ValueError("Load factor must be between 0 and 1.")

        self._capacity: int = initial_capacity
        self._load_factor: float = load_factor
        self._size: int = 0
        self._buckets: List[List[Tuple[Any, Any]]] = [[] for _ in range(self._capacity)]

    def _hash(self, key: Any) -> int:
        """
        Computes the hash index for a given key.

        Args:
            key (Any): The key to hash.

        Returns:
            int: The index corresponding to the key's hash.
        """
        return hash(key) % self._capacity

    def _resize(self) -> None:
        """
        Resizes the hash table by doubling its capacity and rehashing all existing keys.
        """
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0  # Reset size and re-insert to recount

        for bucket in old_buckets:
            for key, value in bucket:
                self.insert(key, value)

    def insert(self, key: Any, value: Any) -> None:
        """
        Inserts a key-value pair into the hash table. If the key already exists, its value is updated.

        Args:
            key (Any): The key to insert.
            value (Any): The value associated with the key.
        """
        index = self._hash(key)
        bucket = self._buckets[index]

        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)
                return

        bucket.append((key, value))
        self._size += 1

        if self._size / self._capacity > self._load_factor:
            self._resize()

    def search(self, key: Any) -> Optional[Any]:
        """
        Searches for a key in the hash table and returns its associated value if found.

        Args:
            key (Any): The key to search for.

        Returns:
            Optional[Any]: The value associated with the key, or None if not found.
        """
        index = self._hash(key)
        bucket = self._buckets[index]

        for existing_key, value in bucket:
            if existing_key == key:
                return value
        return None

    def delete(self, key: Any) -> bool:
        """
        Deletes a key-value pair from the hash table.

        Args:
            key (Any): The key to delete.

        Returns:
            bool: True if the key was found and deleted, False otherwise.
        """
        index = self._hash(key)
        bucket = self._buckets[index]

        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self._size -= 1
                return True
        return False

    def __len__(self) -> int:
        """
        Returns the number of key-value pairs in the hash table.

        Returns:
            int: The size of the hash table.
        """
        return self._size

    def __contains__(self, key: Any) -> bool:
        """
        Checks if a key is present in the hash table.

        Args:
            key (Any): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return self.search(key) is not None

    def keys(self) -> List[Any]:
        """
        Retrieves all keys stored in the hash table.

        Returns:
            List[Any]: A list of keys.
        """
        all_keys: List[Any] = []
        for bucket in self._buckets:
            all_keys.extend([key for key, _ in bucket])
        return all_keys

    def values(self) -> List[Any]:
        """
        Retrieves all values stored in the hash table.

        Returns:
            List[Any]: A list of values.
        """
        all_values: List[Any] = []
        for bucket in self._buckets:
            all_values.extend([value for _, value in bucket])
        return all_values

    def items(self) -> List[Tuple[Any, Any]]:
        """
        Retrieves all key-value pairs stored in the hash table.

        Returns:
            List[Tuple[Any, Any]]: A list of key-value pairs.
        """
        all_items: List[Tuple[Any, Any]] = []
        for bucket in self._buckets:
            all_items.extend(bucket)
        return all_items

    def __iter__(self):
        """
        Allows iteration over the hash table's keys.

        Yields:
            Any: Next key in the hash table.
        """
        for bucket in self._buckets:
            for key, _ in bucket:
                yield key

    def __str__(self) -> str:
        """
        Returns a string representation of the hash table.

        Returns:
            str: String representation.
        """
        items = ["{!r}: {!r}".format(key, value) for key, value in self.items()]
        return "{" + ", ".join(items) + "}"

    def load_factor_ratio(self) -> float:
        """
        Calculates the current load factor ratio of the hash table.

        Returns:
            float: The load factor ratio.
        """
        return self._size / self._capacity

    @staticmethod
    def _is_prime(n: int) -> bool:
        """
        Checks if a number is prime.

        Args:
            n (int): The number to check.

        Returns:
            bool: True if prime, False otherwise.
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def _next_prime(n: int) -> int:
        """
        Finds the next prime number greater than a given number.

        Args:
            n (int): The number to start from.

        Returns:
            int: The next prime number.
        """
        while True:
            n += 1
            if HashTable._is_prime(n):
                return n

    def optimize_capacity(self) -> None:
        """
        Optimizes the hash table's capacity to the next prime number
        greater than twice its current size to reduce collision probability.
        """
        new_capacity = self._next_prime(2 * self._capacity)
        old_buckets = self._buckets
        self._capacity = new_capacity
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0

        for bucket in old_buckets:
            for key, value in bucket:
                self.insert(key, value)


def main() -> None:
    """
    Demonstrates the usage of the HashTable class with various operations.
    """
    hash_table = HashTable()

    # Inserting key-value pairs
    hash_table.insert("apple", 1)
    hash_table.insert("banana", 2)
    hash_table.insert("orange", 3)
    hash_table.insert("grape", 4)

    print("Hash Table after inserts:")
    print(hash_table)

    # Searching for keys
    print("\nSearch for 'apple':", hash_table.search("apple"))
    print("Search for 'cherry':", hash_table.search("cherry"))

    # Deleting a key
    hash_table.delete("banana")
    print("\nHash Table after deleting 'banana':")
    print(hash_table)

    # Checking containment
    print("\nIs 'orange' in hash table?", "orange" in hash_table)
    print("Is 'banana' in hash table?", "banana" in hash_table)

    # Iterating over keys
    print("\nKeys in hash table:")
    for key in hash_table:
        print(key)

    # Displaying all values
    print("\nValues in hash table:", hash_table.values())

    # Displaying all items
    print("\nItems in hash table:", hash_table.items())

    # Displaying current load factor
    print("\nCurrent load factor:", hash_table.load_factor_ratio())

    # Optimizing capacity
    hash_table.optimize_capacity()
    print("\nHash Table after optimizing capacity:")
    print(hash_table)
    print("New capacity:", hash_table._capacity)


if __name__ == "__main__":
    main()