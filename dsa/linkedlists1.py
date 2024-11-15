"""
Linked Lists Module
===================

This module provides comprehensive implementations of various types of linked lists in Python,
ranging from basic singly linked lists to more advanced structures like doubly and circular
linked lists. It includes a variety of operations and algorithms to manipulate and interact
with these data structures efficiently.

Adheres to PEP-8 standards, utilizes type hints from the `typing` module, and includes
comprehensive error handling to ensure robustness and maintainability.
Author: Generalmodelai-agent
Date: 2024-10-15
"""

from __future__ import annotations
from typing import Any, Optional, Callable, Generator


class LinkedListError(Exception):
    """Custom exception class for LinkedList operations."""
    pass


class Node:
    """
    Represents a node in a linked list.

    Attributes:
        data (Any): The data stored in the node.
        next (Optional[Node]): Reference to the next node in the list.
        prev (Optional[Node]): Reference to the previous node in the list (for doubly linked lists).
    """

    def __init__(self, data: Any) -> None:
        self.data: Any = data
        self.next: Optional[Node] = None
        self.prev: Optional[Node] = None

    def __repr__(self) -> str:
        return f"Node({self.data})"


class SinglyLinkedList:
    """
    Implements a singly linked list.

    Methods:
        append(data): Appends a new node with the specified data to the end of the list.
        prepend(data): Prepends a new node with the specified data to the beginning of the list.
        insert_after(target: Any, data): Inserts a new node after the node containing target data.
        delete(data): Deletes the first node containing the specified data.
        find(data) -> Optional[Node]: Finds and returns the node containing the specified data.
        reverse(): Reverses the linked list in place.
        __iter__(): Allows iteration over the linked list.
        __len__() -> int: Returns the number of nodes in the list.
        __repr__() -> str: Returns a string representation of the list.
    """

    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self._size: int = 0

    def append(self, data: Any) -> None:
        """Appends a new node with the specified data to the end of the list."""
        new_node: Node = Node(data)
        if not self.head:
            self.head = new_node
            print(f"Appended head: {new_node}")
        else:
            current: Node = self.head
            while current.next:
                current = current.next
            current.next = new_node
            print(f"Appended: {new_node}")
        self._size += 1

    def prepend(self, data: Any) -> None:
        """Prepends a new node with the specified data to the beginning of the list."""
        new_node: Node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1
        print(f"Prepended: {new_node}")

    def insert_after(self, target: Any, data: Any) -> None:
        """
        Inserts a new node with the specified data after the node containing target data.

        Raises:
            LinkedListError: If the target data is not found in the list.
        """
        target_node: Optional[Node] = self.find(target)
        if not target_node:
            raise LinkedListError(f"Target {target} not found in the list.")
        new_node: Node = Node(data)
        new_node.next = target_node.next
        target_node.next = new_node
        self._size += 1
        print(f"Inserted {new_node} after {target_node}")

    def delete(self, data: Any) -> None:
        """
        Deletes the first node containing the specified data.

        Raises:
            LinkedListError: If the data is not found in the list.
        """
        current: Optional[Node] = self.head
        previous: Optional[Node] = None
        while current and current.data != data:
            previous = current
            current = current.next
        if not current:
            raise LinkedListError(f"Data {data} not found in the list.")
        if previous:
            previous.next = current.next
            print(f"Deleted {current} from the list.")
        else:
            self.head = current.next
            print(f"Deleted head {current} from the list.")
        self._size -= 1

    def find(self, data: Any) -> Optional[Node]:
        """Finds and returns the node containing the specified data."""
        current: Optional[Node] = self.head
        while current:
            if current.data == data:
                print(f"Found {current}")
                return current
            current = current.next
        print(f"{data} not found in the list.")
        return None

    def reverse(self) -> None:
        """Reverses the linked list in place."""
        previous: Optional[Node] = None
        current: Optional[Node] = self.head
        while current:
            next_node: Optional[Node] = current.next
            current.next = previous
            previous = current
            current = next_node
        self.head = previous
        print("Reversed the list.")

    def __iter__(self) -> Generator[Any, None, None]:
        """Allows iteration over the linked list."""
        current: Optional[Node] = self.head
        while current:
            yield current.data
            current = current.next

    def __len__(self) -> int:
        """Returns the number of nodes in the list."""
        return self._size

    def __repr__(self) -> str:
        return " -> ".join(str(data) for data in self)


class DoublyLinkedList:
    """
    Implements a doubly linked list.

    Methods:
        append(data): Appends a new node with the specified data to the end of the list.
        prepend(data): Prepends a new node with the specified data to the beginning of the list.
        insert_after(target: Any, data): Inserts a new node after the node containing target data.
        delete(data): Deletes the first node containing the specified data.
        find(data) -> Optional[Node]: Finds and returns the node containing the specified data.
        reverse(): Reverses the linked list in place.
        __iter__(): Allows iteration over the linked list.
        __len__() -> int: Returns the number of nodes in the list.
        __repr__() -> str: Returns a string representation of the list.
    """

    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self._size: int = 0

    def append(self, data: Any) -> None:
        """Appends a new node with the specified data to the end of the list."""
        new_node: Node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
            print(f"Appended head and tail: {new_node}")
        else:
            assert self.tail is not None  # For type checker
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
            print(f"Appended: {new_node}")
        self._size += 1

    def prepend(self, data: Any) -> None:
        """Prepends a new node with the specified data to the beginning of the list."""
        new_node: Node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
            print(f"Prepended head and tail: {new_node}")
        else:
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node
            print(f"Prepended: {new_node}")
        self._size += 1

    def insert_after(self, target: Any, data: Any) -> None:
        """
        Inserts a new node with the specified data after the node containing target data.

        Raises:
            LinkedListError: If the target data is not found in the list.
        """
        target_node: Optional[Node] = self.find(target)
        if not target_node:
            raise LinkedListError(f"Target {target} not found in the list.")
        new_node: Node = Node(data)
        new_node.next = target_node.next
        new_node.prev = target_node
        if target_node.next:
            target_node.next.prev = new_node
        target_node.next = new_node
        if self.tail == target_node:
            self.tail = new_node
        self._size += 1
        print(f"Inserted {new_node} after {target_node}")

    def delete(self, data: Any) -> None:
        """
        Deletes the first node containing the specified data.

        Raises:
            LinkedListError: If the data is not found in the list.
        """
        current: Optional[Node] = self.head
        while current and current.data != data:
            current = current.next
        if not current:
            raise LinkedListError(f"Data {data} not found in the list.")
        if current.prev:
            current.prev.next = current.next
        else:
            self.head = current.next
        if current.next:
            current.next.prev = current.prev
        else:
            self.tail = current.prev
        self._size -= 1
        print(f"Deleted {current} from the list.")

    def find(self, data: Any) -> Optional[Node]:
        """Finds and returns the node containing the specified data."""
        current: Optional[Node] = self.head
        while current:
            if current.data == data:
                print(f"Found {current}")
                return current
            current = current.next
        print(f"{data} not found in the list.")
        return None

    def reverse(self) -> None:
        """Reverses the linked list in place."""
        current: Optional[Node] = self.head
        self.tail = current
        previous: Optional[Node] = None
        while current:
            previous = current.prev
            current.prev = current.next
            current.next = previous
            current = current.prev
        if previous:
            self.head = previous.prev
        print("Reversed the list.")

    def __iter__(self) -> Generator[Any, None, None]:
        """Allows iteration over the linked list."""
        current: Optional[Node] = self.head
        while current:
            yield current.data
            current = current.next

    def __len__(self) -> int:
        """Returns the number of nodes in the list."""
        return self._size

    def __repr__(self) -> str:
        return " <-> ".join(str(data) for data in self)


class CircularLinkedList:
    """
    Implements a circular singly linked list.

    Methods:
        append(data): Appends a new node with the specified data to the end of the list.
        prepend(data): Prepends a new node with the specified data to the beginning of the list.
        insert_after(target: Any, data): Inserts a new node after the node containing target data.
        delete(data): Deletes the first node containing the specified data.
        find(data) -> Optional[Node]: Finds and returns the node containing the specified data.
        is_circular() -> bool: Checks if the linked list is circular.
        __iter__(): Allows iteration over the linked list.
        __len__() -> int: Returns the number of nodes in the list.
        __repr__() -> str: Returns a string representation of the list.
    """

    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self._size: int = 0

    def append(self, data: Any) -> None:
        """Appends a new node with the specified data to the end of the list."""
        new_node: Node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = new_node
            self.tail = new_node
            print(f"Appended head and tail: {new_node}")
        else:
            assert self.tail is not None  # For type checker
            self.tail.next = new_node
            new_node.next = self.head
            self.tail = new_node
            print(f"Appended: {new_node}")
        self._size += 1

    def prepend(self, data: Any) -> None:
        """Prepends a new node with the specified data to the beginning of the list."""
        new_node: Node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = new_node
            self.tail = new_node
            print(f"Prepended head and tail: {new_node}")
        else:
            assert self.tail is not None  # For type checker
            new_node.next = self.head
            self.tail.next = new_node
            self.head = new_node
            print(f"Prepended: {new_node}")
        self._size += 1

    def insert_after(self, target: Any, data: Any) -> None:
        """
        Inserts a new node with the specified data after the node containing target data.

        Raises:
            LinkedListError: If the target data is not found in the list.
        """
        target_node: Optional[Node] = self.find(target)
        if not target_node:
            raise LinkedListError(f"Target {target} not found in the list.")
        new_node: Node = Node(data)
        new_node.next = target_node.next
        target_node.next = new_node
        if self.tail == target_node:
            self.tail = new_node
        self._size += 1
        print(f"Inserted {new_node} after {target_node}")

    def delete(self, data: Any) -> None:
        """
        Deletes the first node containing the specified data.

        Raises:
            LinkedListError: If the data is not found in the list.
        """
        if not self.head:
            raise LinkedListError("Cannot delete from an empty list.")
        current: Node = self.head
        previous: Optional[Node] = self.tail
        for _ in range(self._size):
            if current.data == data:
                if current == self.head:
                    self.head = self.head.next
                    self.tail.next = self.head
                else:
                    assert previous is not None
                    previous.next = current.next
                    if current == self.tail:
                        self.tail = previous
                self._size -= 1
                print(f"Deleted {current} from the list.")
                return
            previous = current
            current = current.next
        raise LinkedListError(f"Data {data} not found in the list.")

    def find(self, data: Any) -> Optional[Node]:
        """Finds and returns the node containing the specified data."""
        if not self.head:
            print("List is empty.")
            return None
        current: Node = self.head
        for _ in range(self._size):
            if current.data == data:
                print(f"Found {current}")
                return current
            current = current.next
        print(f"{data} not found in the list.")
        return None

    def is_circular(self) -> bool:
        """Checks if the linked list is circular."""
        return self.tail.next == self.head if self.tail else False

    def __iter__(self) -> Generator[Any, None, None]:
        """Allows iteration over the linked list."""
        if not self.head:
            return
        current: Node = self.head
        for _ in range(self._size):
            yield current.data
            current = current.next

    def __len__(self) -> int:
        """Returns the number of nodes in the list."""
        return self._size

    def __repr__(self) -> str:
        if not self.head:
            return "Empty CircularLinkedList"
        nodes = []
        current: Node = self.head
        for _ in range(self._size):
            nodes.append(str(current.data))
            current = current.next
        return " -> ".join(nodes) + " -> (back to head)"


def detect_cycle(linked_list: SinglyLinkedList) -> bool:
    """
    Detects if there is a cycle in the linked list using Floyd's Tortoise and Hare algorithm.

    Args:
        linked_list (SinglyLinkedList): The linked list to check for cycles.

    Returns:
        bool: True if a cycle is detected, False otherwise.
    """
    slow: Optional[Node] = linked_list.head
    fast: Optional[Node] = linked_list.head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            print("Cycle detected in the list.")
            return True
    print("No cycle detected in the list.")
    return False


def find_middle(linked_list: SinglyLinkedList) -> Optional[Any]:
    """
    Finds the middle node of the linked list using the slow and fast pointer approach.

    Args:
        linked_list (SinglyLinkedList): The linked list to find the middle of.

    Returns:
        Optional[Any]: The data of the middle node, or None if the list is empty.
    """
    slow: Optional[Node] = linked_list.head
    fast: Optional[Node] = linked_list.head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    if slow:
        print(f"Middle node is {slow.data}")
        return slow.data
    print("The list is empty, no middle node.")
    return None


def remove_duplicates(linked_list: SinglyLinkedList) -> None:
    """
    Removes duplicate nodes from the linked list.

    Args:
        linked_list (SinglyLinkedList): The linked list to remove duplicates from.
    """
    seen: set = set()
    current: Optional[Node] = linked_list.head
    previous: Optional[Node] = None
    while current:
        if current.data in seen:
            assert previous is not None
            previous.next = current.next
            linked_list._size -= 1
            print(f"Removed duplicate node {current}")
        else:
            seen.add(current.data)
            previous = current
        current = current.next
    print("Duplicates removed.")


def reverse_linked_list(linked_list: SinglyLinkedList) -> None:
    """
    Reverses the linked list.

    Args:
        linked_list (SinglyLinkedList): The linked list to reverse.
    """
    linked_list.reverse()


def main() -> None:
    """Demonstrates the usage of the linked list implementations and algorithms."""
    print("=== Singly Linked List ===")
    sll = SinglyLinkedList()
    sll.append(1)
    sll.append(2)
    sll.append(3)
    sll.prepend(0)
    sll.insert_after(2, 2.5)
    print(f"List: {sll}")
    sll.delete(2.5)
    print(f"After deletion: {sll}")
    print(f"Length: {len(sll)}")
    middle = find_middle(sll)
    sll.reverse()
    print(f"Reversed List: {sll}")
    has_cycle = detect_cycle(sll)
    remove_duplicates(sll)
    print(f"Final List: {sll}\n")

    print("=== Doubly Linked List ===")
    dll = DoublyLinkedList()
    dll.append('a')
    dll.append('b')
    dll.append('c')
    dll.prepend('z')
    dll.insert_after('b', 'b2')
    print(f"List: {dll}")
    dll.delete('b2')
    print(f"After deletion: {dll}")
    dll.reverse()
    print(f"Reversed List: {dll}")
    print(f"Length: {len(dll)}\n")

    print("=== Circular Linked List ===")
    cll = CircularLinkedList()
    cll.append(10)
    cll.append(20)
    cll.append(30)
    cll.prepend(5)
    cll.insert_after(20, 25)
    print(f"List: {cll}")
    cll.delete(25)
    print(f"After deletion: {cll}")
    print(f"Is circular: {cll.is_circular()}")
    print(f"Length: {len(cll)}")


if __name__ == "__main__":
    main()