"""
Linked Lists Data Structures and Algorithms
===========================================

This module provides comprehensive implementations of various types of linked lists,
ranging from basic singly linked lists to more advanced variants like doubly and
circular linked lists. Each implementation includes methods for common operations,
ensuring efficiency, robustness, and scalability.

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from __future__ import annotations
from typing import Any, Optional, Callable, TypeVar, Generic

# Define a generic type variable for nodes
T = TypeVar('T')


class Node(Generic[T]):
    """
    Represents a node in a linked list.

    Attributes:
        data (T): The value stored in the node.
        next (Optional[Node[T]]): Reference to the next node in the list.
    """

    def __init__(self, data: T) -> None:
        self.data: T = data
        self.next: Optional[Node[T]] = None


class DoublyNode(Generic[T]):
    """
    Represents a node in a doubly linked list.

    Attributes:
        data (T): The value stored in the node.
        prev (Optional[DoublyNode[T]]): Reference to the previous node.
        next (Optional[DoublyNode[T]]): Reference to the next node.
    """

    def __init__(self, data: T) -> None:
        self.data: T = data
        self.prev: Optional[DoublyNode[T]] = None
        self.next: Optional[DoublyNode[T]] = None


class SinglyLinkedList(Generic[T]):
    """
    Implements a singly linked list.

    Methods:
        append(data): Adds an element to the end of the list.
        prepend(data): Adds an element to the beginning of the list.
        insert_after(target: T, data: T): Inserts an element after the target.
        delete(data): Deletes the first occurrence of the element.
        find(data) -> Optional[Node[T]]: Finds the first node with the given data.
        to_list() -> list[T]: Converts the linked list to a Python list.
        __iter__(): Iterator for the linked list.
        __len__() -> int: Returns the number of elements in the list.
    """

    def __init__(self) -> None:
        self.head: Optional[Node[T]] = None
        self.tail: Optional[Node[T]] = None
        self._size: int = 0

    def append(self, data: T) -> None:
        """Append an element to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
            print(f"Appended head: {new_node.data}")
        else:
            assert self.tail is not None  # for type checker
            self.tail.next = new_node
            self.tail = new_node
            print(f"Appended: {new_node.data}")
        self._size += 1

    def prepend(self, data: T) -> None:
        """Prepend an element to the beginning of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
            print(f"Prepended head: {new_node.data}")
        else:
            new_node.next = self.head
            self.head = new_node
            print(f"Prepended: {new_node.data}")
        self._size += 1

    def insert_after(self, target: T, data: T) -> None:
        """Insert an element after the first occurrence of the target element."""
        current = self.find(target)
        if not current:
            raise ValueError(f"Element {target} not found in the list.")
        new_node = Node(data)
        new_node.next = current.next
        current.next = new_node
        if current == self.tail:
            self.tail = new_node
        self._size += 1
        print(f"Inserted {data} after {target}")

    def delete(self, data: T) -> None:
        """Delete the first occurrence of the element in the list."""
        current = self.head
        previous: Optional[Node[T]] = None
        while current:
            if current.data == data:
                if previous:
                    previous.next = current.next
                else:
                    self.head = current.next
                if current == self.tail:
                    self.tail = previous
                self._size -= 1
                print(f"Deleted: {data}")
                return
            previous = current
            current = current.next
        raise ValueError(f"Element {data} not found in the list.")

    def find(self, data: T) -> Optional[Node[T]]:
        """Find the first node containing the specified data."""
        current = self.head
        while current:
            if current.data == data:
                print(f"Found: {data}")
                return current
            current = current.next
        print(f"Element {data} not found.")
        return None

    def to_list(self) -> list[T]:
        """Convert the linked list to a standard Python list."""
        elements: list[T] = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        print(f"Converted to list: {elements}")
        return elements

    def __iter__(self) -> Callable[[], Any]:
        """Iterator for the linked list."""
        current = self.head
        while current:
            yield current.data
            current = current.next

    def __len__(self) -> int:
        """Return the number of elements in the list."""
        return self._size

    def reverse(self) -> None:
        """Reverse the linked list in place."""
        previous = None
        current = self.head
        self.tail = self.head
        while current:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node
        self.head = previous
        print("Reversed the linked list.")

    def detect_cycle(self) -> bool:
        """
        Detect if there is a cycle in the linked list using Floyd's Tortoise and Hare algorithm.
        
        Returns:
            bool: True if a cycle is detected, False otherwise.
        """
        slow = self.head
        fast = self.head
        while fast and fast.next:
            slow = slow.next  # move by 1
            fast = fast.next.next  # move by 2
            if slow == fast:
                print("Cycle detected in the linked list.")
                return True
        print("No cycle detected in the linked list.")
        return False


class DoublyLinkedList(Generic[T]):
    """
    Implements a doubly linked list.

    Methods:
        append(data): Adds an element to the end of the list.
        prepend(data): Adds an element to the beginning of the list.
        insert_after(target: T, data: T): Inserts an element after the target.
        insert_before(target: T, data: T): Inserts an element before the target.
        delete(data): Deletes the first occurrence of the element.
        find(data) -> Optional[DoublyNode[T]]: Finds the first node with the given data.
        to_list() -> list[T]: Converts the linked list to a Python list.
        __iter__(): Iterator for the linked list.
        __len__() -> int: Returns the number of elements in the list.
    """

    def __init__(self) -> None:
        self.head: Optional[DoublyNode[T]] = None
        self.tail: Optional[DoublyNode[T]] = None
        self._size: int = 0

    def append(self, data: T) -> None:
        """Append an element to the end of the list."""
        new_node = DoublyNode(data)
        if not self.head:
            self.head = self.tail = new_node
            print(f"Doubly Append head: {new_node.data}")
        else:
            assert self.tail is not None  # for type checker
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
            print(f"Doubly Appended: {new_node.data}")
        self._size += 1

    def prepend(self, data: T) -> None:
        """Prepend an element to the beginning of the list."""
        new_node = DoublyNode(data)
        if not self.head:
            self.head = self.tail = new_node
            print(f"Doubly Prepended head: {new_node.data}")
        else:
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node
            print(f"Doubly Prepended: {new_node.data}")
        self._size += 1

    def insert_after(self, target: T, data: T) -> None:
        """Insert an element after the first occurrence of the target element."""
        current = self.find(target)
        if not current:
            raise ValueError(f"Element {target} not found in the list.")
        new_node = DoublyNode(data)
        new_node.next = current.next
        new_node.prev = current
        if current.next:
            current.next.prev = new_node
        current.next = new_node
        if current == self.tail:
            self.tail = new_node
        self._size += 1
        print(f"Doubly Inserted {data} after {target}")

    def insert_before(self, target: T, data: T) -> None:
        """Insert an element before the first occurrence of the target element."""
        current = self.find(target)
        if not current:
            raise ValueError(f"Element {target} not found in the list.")
        new_node = DoublyNode(data)
        new_node.prev = current.prev
        new_node.next = current
        if current.prev:
            current.prev.next = new_node
        else:
            self.head = new_node
        current.prev = new_node
        self._size += 1
        print(f"Doubly Inserted {data} before {target}")

    def delete(self, data: T) -> None:
        """Delete the first occurrence of the element in the list."""
        current = self.find(data)
        if not current:
            raise ValueError(f"Element {data} not found in the list.")
        if current.prev:
            current.prev.next = current.next
        else:
            self.head = current.next
        if current.next:
            current.next.prev = current.prev
        else:
            self.tail = current.prev
        self._size -= 1
        print(f"Doubly Deleted: {data}")

    def find(self, data: T) -> Optional[DoublyNode[T]]:
        """Find the first node containing the specified data."""
        current = self.head
        while current:
            if current.data == data:
                print(f"Doubly Found: {data}")
                return current
            current = current.next
        print(f"Doubly Element {data} not found.")
        return None

    def to_list(self) -> list[T]:
        """Convert the doubly linked list to a standard Python list."""
        elements: list[T] = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        print(f"Doubly Converted to list: {elements}")
        return elements

    def __iter__(self) -> Callable[[], Any]:
        """Iterator for the doubly linked list."""
        current = self.head
        while current:
            yield current.data
            current = current.next

    def __len__(self) -> int:
        """Return the number of elements in the list."""
        return self._size

    def reverse(self) -> None:
        """Reverse the doubly linked list in place."""
        current = self.head
        self.head, self.tail = self.tail, self.head
        while current:
            current.prev, current.next = current.next, current.prev
            current = current.prev  # since we swapped next and prev
        print("Doubly reversed the linked list.")

    def detect_cycle(self) -> bool:
        """
        Detect if there is a cycle in the doubly linked list using Floyd's Tortoise and Hare algorithm.

        Returns:
            bool: True if a cycle is detected, False otherwise.
        """
        slow = self.head
        fast = self.head
        while fast and fast.next:
            slow = slow.next  # move by 1
            fast = fast.next.next  # move by 2
            if slow == fast:
                print("Doubly cycle detected in the linked list.")
                return True
        print("No cycle detected in the doubly linked list.")
        return False


class CircularLinkedList(Generic[T]):
    """
    Implements a circular singly linked list.

    Methods:
        append(data): Adds an element to the end of the list.
        prepend(data): Adds an element to the beginning of the list.
        insert_after(target: T, data: T): Inserts an element after the target.
        delete(data): Deletes the first occurrence of the element.
        find(data) -> Optional[Node[T]]: Finds the first node with the given data.
        to_list() -> list[T]: Converts the linked list to a Python list.
        __iter__(): Iterator for the linked list.
        __len__() -> int: Returns the number of elements in the list.
    """

    def __init__(self) -> None:
        self.head: Optional[Node[T]] = None
        self._size: int = 0

    def append(self, data: T) -> None:
        """Append an element to the end of the circular list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = new_node
            print(f"Circular Append head: {new_node.data}")
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = new_node
            new_node.next = self.head
            print(f"Circular Appended: {new_node.data}")
        self._size += 1

    def prepend(self, data: T) -> None:
        """Prepend an element to the beginning of the circular list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            new_node.next = new_node
            print(f"Circular Prepend head: {new_node.data}")
        else:
            current = self.head
            while current.next != self.head:
                current = current.next
            new_node.next = self.head
            current.next = new_node
            self.head = new_node
            print(f"Circular Prepended: {new_node.data}")
        self._size += 1

    def insert_after(self, target: T, data: T) -> None:
        """Insert an element after the first occurrence of the target element."""
        target_node = self.find(target)
        if not target_node:
            raise ValueError(f"Element {target} not found in the circular list.")
        new_node = Node(data)
        new_node.next = target_node.next
        target_node.next = new_node
        if target_node == self.head:
            print(f"Circular Inserted {data} after {target}")
        self._size += 1

    def delete(self, data: T) -> None:
        """Delete the first occurrence of the element in the circular list."""
        if not self.head:
            raise ValueError("Cannot delete from an empty circular list.")
        current = self.head
        previous: Optional[Node[T]] = None
        while True:
            if current.data == data:
                if previous:
                    previous.next = current.next
                else:
                    # Find the last node to update its next pointer
                    tail = self.head
                    while tail.next != self.head:
                        tail = tail.next
                    self.head = current.next
                    tail.next = self.head
                self._size -= 1
                print(f"Circular Deleted: {data}")
                return
            previous = current
            current = current.next
            if current == self.head:
                break
        raise ValueError(f"Element {data} not found in the circular list.")

    def find(self, data: T) -> Optional[Node[T]]:
        """Find the first node containing the specified data."""
        if not self.head:
            print("Circular list is empty.")
            return None
        current = self.head
        while True:
            if current.data == data:
                print(f"Circular Found: {data}")
                return current
            current = current.next
            if current == self.head:
                break
        print(f"Circular Element {data} not found.")
        return None

    def to_list(self) -> list[T]:
        """Convert the circular linked list to a standard Python list."""
        elements: list[T] = []
        if not self.head:
            print("Circular list is empty.")
            return elements
        current = self.head
        while True:
            elements.append(current.data)
            current = current.next
            if current == self.head:
                break
        print(f"Circular Converted to list: {elements}")
        return elements

    def __iter__(self) -> Callable[[], Any]:
        """Iterator for the circular linked list."""
        if not self.head:
            return
        current = self.head
        while True:
            yield current.data
            current = current.next
            if current == self.head:
                break

    def __len__(self) -> int:
        """Return the number of elements in the circular list."""
        return self._size

    def detect_cycle(self) -> bool:
        """
        Since it's a circular linked list, it always contains a cycle.

        Returns:
            bool: Always True for circular linked lists.
        """
        if self.head:
            print("Circular linked list inherently has a cycle.")
            return True
        print("Circular linked list is empty, no cycle.")
        return False

    def split_into_two(self) -> tuple[CircularLinkedList[T], CircularLinkedList[T]]:
        """
        Splits the circular linked list into two equal halves.

        Returns:
            tuple[CircularLinkedList[T], CircularLinkedList[T]]: Two half circular linked lists.
        """
        if not self.head:
            raise ValueError("Cannot split an empty circular list.")
        slow = self.head
        fast = self.head

        # Use the slow and fast pointer strategy to find the midpoint
        while fast.next != self.head and fast.next.next != self.head:
            slow = slow.next  # move by 1
            fast = fast.next.next  # move by 2

        half1 = CircularLinkedList[T]()
        half2 = CircularLinkedList[T]()

        half1.head = self.head
        if self.head.next:
            # Split the list into two halves
            if fast.next.next == self.head:
                fast = fast.next
            half2.head = slow.next
            fast.next = half2.head
            slow.next = half1.head

        # Calculate sizes
        half1._size = (self._size + 1) // 2
        half2._size = self._size // 2

        print(f"Circular list split into two halves: {half1.to_list()} and {half2.to_list()}")
        return half1, half2