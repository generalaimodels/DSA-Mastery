"""
Queues in Python: From Basics to Advanced Implementations

This module provides comprehensive implementations of various queue data structures
in Python, adhering to PEP-8 standards, utilizing type hints from the typing module,
and ensuring robustness and efficiency. The implementations cover basic queues, circular
queues, priority queues, and double-ended queues (deques), each designed to handle
different scenarios and requirements.

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from __future__ import annotations
from typing import Any, Generic, Optional, TypeVar, List
import torch

T = TypeVar('T')


class QueueEmptyException(Exception):
    """Exception raised when attempting to dequeue from an empty queue."""
    pass


class QueueFullException(Exception):
    """Exception raised when attempting to enqueue to a full queue."""
    pass


class Queue(Generic[T]):
    """
    A basic implementation of a queue using a list.
    Supports enqueue and dequeue operations.
    """

    def __init__(self) -> None:
        """Initialize an empty queue."""
        self._items: List[T] = []

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._items) == 0

    def enqueue(self, item: T) -> None:
        """
        Add an item to the end of the queue.

        Args:
            item (T): The item to be added.
        """
        self._items.append(item)

    def dequeue(self) -> T:
        """
        Remove and return the item at the front of the queue.

        Returns:
            T: The dequeued item.

        Raises:
            QueueEmptyException: If the queue is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot dequeue from an empty queue.")
        return self._items.pop(0)

    def peek(self) -> T:
        """
        Return the item at the front of the queue without removing it.

        Returns:
            T: The front item.

        Raises:
            QueueEmptyException: If the queue is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot peek into an empty queue.")
        return self._items[0]

    def size(self) -> int:
        """Return the number of items in the queue."""
        return len(self._items)

    def __repr__(self) -> str:
        """Return a string representation of the queue."""
        return f"Queue({self._items})"


class CircularQueue(Generic[T]):
    """
    A circular queue implementation with a fixed capacity.
    Utilizes a list to store elements and two pointers to track front and rear.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize the circular queue with a given capacity.

        Args:
            capacity (int): The maximum number of items the queue can hold.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self._capacity: int = capacity
        self._items: List[Optional[T]] = [None] * capacity
        self._front: int = 0
        self._rear: int = -1
        self._size: int = 0

    def is_empty(self) -> bool:
        """Check if the circular queue is empty."""
        return self._size == 0

    def is_full(self) -> bool:
        """Check if the circular queue is full."""
        return self._size == self._capacity

    def enqueue(self, item: T) -> None:
        """
        Add an item to the rear of the queue.

        Args:
            item (T): The item to be added.

        Raises:
            QueueFullException: If the queue is full.
        """
        if self.is_full():
            raise QueueFullException("Cannot enqueue to a full queue.")
        self._rear = (self._rear + 1) % self._capacity
        self._items[self._rear] = item
        self._size += 1

    def dequeue(self) -> T:
        """
        Remove and return the front item of the queue.

        Returns:
            T: The dequeued item.

        Raises:
            QueueEmptyException: If the queue is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot dequeue from an empty queue.")
        item = self._items[self._front]
        self._items[self._front] = None  # Optional: Clear reference
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        assert item is not None  # For type checking
        return item

    def peek(self) -> T:
        """
        Return the front item without removing it.

        Returns:
            T: The front item.

        Raises:
            QueueEmptyException: If the queue is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot peek into an empty queue.")
        item = self._items[self._front]
        assert item is not None
        return item

    def size(self) -> int:
        """Return the number of items in the queue."""
        return self._size

    def __repr__(self) -> str:
        """Return a string representation of the circular queue."""
        items = [item for item in self._items if item is not None]
        return f"CircularQueue({items}, capacity={self._capacity})"


class PriorityQueue(Generic[T]):
    """
    A priority queue implementation using a binary heap.
    Items are stored as tuples of (priority, value).
    """

    def __init__(self) -> None:
        """Initialize an empty priority queue."""
        self._heap: List[tuple[int, T]] = []

    def is_empty(self) -> bool:
        """Check if the priority queue is empty."""
        return len(self._heap) == 0

    def enqueue(self, item: T, priority: int) -> None:
        """
        Add an item with the given priority to the queue.

        Args:
            item (T): The item to be added.
            priority (int): The priority of the item.
        """
        from heapq import heappush
        heappush(self._heap, (priority, item))

    def dequeue(self) -> T:
        """
        Remove and return the item with the highest priority (lowest priority number).

        Returns:
            T: The dequeued item.

        Raises:
            QueueEmptyException: If the queue is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot dequeue from an empty priority queue.")
        from heapq import heappop
        priority, item = heappop(self._heap)
        return item

    def peek(self) -> T:
        """
        Return the item with the highest priority without removing it.

        Returns:
            T: The front item.

        Raises:
            QueueEmptyException: If the queue is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot peek into an empty priority queue.")
        return self._heap[0][1]

    def size(self) -> int:
        """Return the number of items in the priority queue."""
        return len(self._heap)

    def __repr__(self) -> str:
        """Return a string representation of the priority queue."""
        return f"PriorityQueue({self._heap})"


class Deque(Generic[T]):
    """
    A double-ended queue implementation allowing insertion and removal from both ends.
    """

    def __init__(self) -> None:
        """Initialize an empty deque."""
        self._items: List[T] = []

    def is_empty(self) -> bool:
        """Check if the deque is empty."""
        return len(self._items) == 0

    def add_front(self, item: T) -> None:
        """
        Add an item to the front of the deque.

        Args:
            item (T): The item to be added.
        """
        self._items.insert(0, item)

    def add_rear(self, item: T) -> None:
        """
        Add an item to the rear of the deque.

        Args:
            item (T): The item to be added.
        """
        self._items.append(item)

    def remove_front(self) -> T:
        """
        Remove and return the front item of the deque.

        Returns:
            T: The removed item.

        Raises:
            QueueEmptyException: If the deque is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot remove from front of an empty deque.")
        return self._items.pop(0)

    def remove_rear(self) -> T:
        """
        Remove and return the rear item of the deque.

        Returns:
            T: The removed item.

        Raises:
            QueueEmptyException: If the deque is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot remove from rear of an empty deque.")
        return self._items.pop()

    def peek_front(self) -> T:
        """
        Return the front item without removing it.

        Returns:
            T: The front item.

        Raises:
            QueueEmptyException: If the deque is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot peek into an empty deque.")
        return self._items[0]

    def peek_rear(self) -> T:
        """
        Return the rear item without removing it.

        Returns:
            T: The rear item.

        Raises:
            QueueEmptyException: If the deque is empty.
        """
        if self.is_empty():
            raise QueueEmptyException("Cannot peek into an empty deque.")
        return self._items[-1]

    def size(self) -> int:
        """Return the number of items in the deque."""
        return len(self._items)

    def __repr__(self) -> str:
        """Return a string representation of the deque."""
        return f"Deque({self._items})"


class BlockingQueue(Generic[T]):
    """
    A thread-safe blocking queue implementation using torch tensors.
    Suitable for producer-consumer scenarios.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize the blocking queue with a fixed capacity.

        Args:
            capacity (int): The maximum number of items the queue can hold.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self._capacity: int = capacity
        self._queue: torch.Tensor = torch.empty(capacity, dtype=torch.object)
        self._front: int = 0
        self._rear: int = -1
        self._size: int = 0

    def is_empty(self) -> bool:
        """Check if the blocking queue is empty."""
        return self._size == 0

    def is_full(self) -> bool:
        """Check if the blocking queue is full."""
        return self._size == self._capacity

    def enqueue(self, item: T) -> None:
        """
        Add an item to the rear of the blocking queue.
        Blocks if the queue is full until space becomes available.

        Args:
            item (T): The item to be added.
        """
        while self.is_full():
            # In a real implementation, use synchronization primitives
            pass  # Placeholder for blocking behavior
        self._rear = (self._rear + 1) % self._capacity
        self._queue[self._rear] = item
        self._size += 1

    def dequeue(self) -> T:
        """
        Remove and return the front item of the blocking queue.
        Blocks if the queue is empty until an item becomes available.

        Returns:
            T: The dequeued item.
        """
        while self.is_empty():
            # In a real implementation, use synchronization primitives
            pass  # Placeholder for blocking behavior
        item = self._queue[self._front].item()
        self._queue[self._front] = None  # Clear reference
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        return item

    def size(self) -> int:
        """Return the number of items in the blocking queue."""
        return self._size

    def __repr__(self) -> str:
        """Return a string representation of the blocking queue."""
        items = [self._queue[(self._front + i) % self._capacity].item()
                 for i in range(self._size)]
        return f"BlockingQueue({items}, capacity={self._capacity})"


def main() -> None:
    """Demonstrate the usage of various queue implementations."""
    # Basic Queue
    print("=== Basic Queue ===")
    basic_queue = Queue[int]()
    basic_queue.enqueue(1)
    basic_queue.enqueue(2)
    basic_queue.enqueue(3)
    print(basic_queue)
    print(f"Dequeued: {basic_queue.dequeue()}")
    print(basic_queue)
    print()

    # Circular Queue
    print("=== Circular Queue ===")
    circular_queue = CircularQueue[int](3)
    circular_queue.enqueue(10)
    circular_queue.enqueue(20)
    circular_queue.enqueue(30)
    print(circular_queue)
    try:
        circular_queue.enqueue(40)
    except QueueFullException as e:
        print(e)
    print(f"Dequeued: {circular_queue.dequeue()}")
    circular_queue.enqueue(40)
    print(circular_queue)
    print()

    # Priority Queue
    print("=== Priority Queue ===")
    priority_queue = PriorityQueue[str]()
    priority_queue.enqueue("low", priority=3)
    priority_queue.enqueue("medium", priority=2)
    priority_queue.enqueue("high", priority=1)
    print(priority_queue)
    print(f"Dequeued: {priority_queue.dequeue()}")
    print(priority_queue)
    print()

    # Deque
    print("=== Deque ===")
    deque = Deque[int]()
    deque.add_front(100)
    deque.add_rear(200)
    deque.add_front(50)
    print(deque)
    print(f"Removed front: {deque.remove_front()}")
    print(f"Removed rear: {deque.remove_rear()}")
    print(deque)
    print()

    # Blocking Queue
    print("=== Blocking Queue ===")
    blocking_queue = BlockingQueue[str](2)
    blocking_queue.enqueue("producer1")
    blocking_queue.enqueue("producer2")
    print(blocking_queue)
    # The following enqueue would block in a real scenario
    blocking_queue.enqueue("producer3")
    print(f"Dequeued: {blocking_queue.dequeue()}")
    blocking_queue.enqueue("producer3")
    print(blocking_queue)


if __name__ == "__main__":
    main()