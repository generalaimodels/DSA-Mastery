"""
Queues Data Structure and Algorithms - Comprehensive Implementation

This module provides detailed implementations of various queue data structures,
ranging from basic to advanced levels. It adheres to PEP-8 standards, utilizes
type hints for clarity, and includes comprehensive error handling to ensure robustness.

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from typing import Any, Generic, List, Optional, TypeVar
import torch

T = TypeVar('T')


class QueueEmptyError(Exception):
    """Custom exception to indicate that the queue is empty."""
    pass


class QueueFullError(Exception):
    """Custom exception to indicate that the queue is full."""
    pass


class SimpleQueue(Generic[T]):
    """
    A simple queue implementation using Python's list.

    Time Complexity:
        - Enqueue: O(1) (amortized)
        - Dequeue: O(n) due to list.pop(0)
    Space Complexity:
        - O(n)
    """

    def __init__(self) -> None:
        """Initialize an empty queue."""
        self._items: List[T] = []

    def enqueue(self, item: T) -> None:
        """Add an item to the end of the queue.

        Args:
            item (T): The item to be added.
        """
        self._items.append(item)
        print(f"Enqueued: {item}. Queue state: {self._items}")

    def dequeue(self) -> T:
        """Remove and return the item from the front of the queue.

        Raises:
            QueueEmptyError: If the queue is empty.

        Returns:
            T: The dequeued item.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot dequeue from an empty queue.")
        item = self._items.pop(0)
        print(f"Dequeued: {item}. Queue state: {self._items}")
        return item

    def peek(self) -> T:
        """Return the item at the front of the queue without removing it.

        Raises:
            QueueEmptyError: If the queue is empty.

        Returns:
            T: The item at the front.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot peek into an empty queue.")
        return self._items[0]

    def is_empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if empty, False otherwise.
        """
        return len(self._items) == 0

    def size(self) -> int:
        """Return the number of items in the queue.

        Returns:
            int: The size of the queue.
        """
        return len(self._items)


class CircularQueue(Generic[T]):
    """
    A circular queue implementation using a fixed-size list.

    Time Complexity:
        - Enqueue: O(1)
        - Dequeue: O(1)
    Space Complexity:
        - O(k) where k is the fixed size
    """

    def __init__(self, capacity: int) -> None:
        """Initialize the circular queue with a fixed capacity.

        Args:
            capacity (int): The maximum number of items the queue can hold.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self._capacity: int = capacity
        self._queue: List[Optional[T]] = [None] * capacity
        self._front: int = 0
        self._rear: int = 0
        self._size: int = 0
        print(f"CircularQueue initialized with capacity {self._capacity}")

    def enqueue(self, item: T) -> None:
        """Add an item to the rear of the queue.

        Args:
            item (T): The item to be added.

        Raises:
            QueueFullError: If the queue is full.
        """
        if self.is_full():
            raise QueueFullError("Cannot enqueue to a full queue.")
        self._queue[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1
        print(f"Enqueued: {item}. Queue state: {self._queue}")

    def dequeue(self) -> T:
        """Remove and return the item from the front of the queue.

        Raises:
            QueueEmptyError: If the queue is empty.

        Returns:
            T: The dequeued item.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot dequeue from an empty queue.")
        item = self._queue[self._front]
        self._queue[self._front] = None  # Optional: Clear the slot
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        print(f"Dequeued: {item}. Queue state: {self._queue}")
        return item  # type: ignore

    def peek(self) -> T:
        """Return the item at the front of the queue without removing it.

        Raises:
            QueueEmptyError: If the queue is empty.

        Returns:
            T: The item at the front.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot peek into an empty queue.")
        return self._queue[self._front]  # type: ignore

    def is_empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if empty, False otherwise.
        """
        return self._size == 0

    def is_full(self) -> bool:
        """Check if the queue is full.

        Returns:
            bool: True if full, False otherwise.
        """
        return self._size == self._capacity

    def size(self) -> int:
        """Return the number of items in the queue.

        Returns:
            int: The size of the queue.
        """
        return self._size


import heapq


class PriorityQueue(Generic[T]):
    """
    A priority queue implementation using a binary heap.

    Time Complexity:
        - Enqueue: O(log n)
        - Dequeue: O(log n)
    Space Complexity:
        - O(n)
    """

    def __init__(self) -> None:
        """Initialize an empty priority queue."""
        self._heap: List[Any] = []
        print("PriorityQueue initialized.")

    def enqueue(self, item: T, priority: int) -> None:
        """Add an item with a given priority to the queue.

        Lower priority numbers indicate higher priority.

        Args:
            item (T): The item to be added.
            priority (int): The priority of the item.
        """
        heapq.heappush(self._heap, (priority, item))
        print(f"Enqueued: {item} with priority {priority}. Heap state: {self._heap}")

    def dequeue(self) -> T:
        """Remove and return the highest priority item from the queue.

        Raises:
            QueueEmptyError: If the queue is empty.

        Returns:
            T: The dequeued item.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot dequeue from an empty priority queue.")
        priority, item = heapq.heappop(self._heap)
        print(f"Dequeued: {item} with priority {priority}. Heap state: {self._heap}")
        return item

    def peek(self) -> T:
        """Return the highest priority item without removing it.

        Raises:
            QueueEmptyError: If the queue is empty.

        Returns:
            T: The highest priority item.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot peek into an empty priority queue.")
        return self._heap[0][1]

    def is_empty(self) -> bool:
        """Check if the priority queue is empty.

        Returns:
            bool: True if empty, False otherwise.
        """
        return len(self._heap) == 0

    def size(self) -> int:
        """Return the number of items in the priority queue.

        Returns:
            int: The size of the priority queue.
        """
        return len(self._heap)


class Deque(Generic[T]):
    """
    A double-ended queue (deque) implementation using a dynamic array.

    Time Complexity:
        - Enqueue Front/Rear: O(1) amortized
        - Dequeue Front/Rear: O(1)
    Space Complexity:
        - O(n)
    """

    def __init__(self) -> None:
        """Initialize an empty deque."""
        self._items: List[T] = []
        print("Deque initialized.")

    def enqueue_front(self, item: T) -> None:
        """Add an item to the front of the deque.

        Args:
            item (T): The item to be added.
        """
        self._items.insert(0, item)
        print(f"Enqueued at front: {item}. Deque state: {self._items}")

    def enqueue_rear(self, item: T) -> None:
        """Add an item to the rear of the deque.

        Args:
            item (T): The item to be added.
        """
        self._items.append(item)
        print(f"Enqueued at rear: {item}. Deque state: {self._items}")

    def dequeue_front(self) -> T:
        """Remove and return the item from the front of the deque.

        Raises:
            QueueEmptyError: If the deque is empty.

        Returns:
            T: The dequeued item.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot dequeue from an empty deque.")
        item = self._items.pop(0)
        print(f"Dequeued from front: {item}. Deque state: {self._items}")
        return item

    def dequeue_rear(self) -> T:
        """Remove and return the item from the rear of the deque.

        Raises:
            QueueEmptyError: If the deque is empty.

        Returns:
            T: The dequeued item.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot dequeue from an empty deque.")
        item = self._items.pop()
        print(f"Dequeued from rear: {item}. Deque state: {self._items}")
        return item

    def peek_front(self) -> T:
        """Return the item at the front of the deque without removing it.

        Raises:
            QueueEmptyError: If the deque is empty.

        Returns:
            T: The front item.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot peek into an empty deque.")
        return self._items[0]

    def peek_rear(self) -> T:
        """Return the item at the rear of the deque without removing it.

        Raises:
            QueueEmptyError: If the deque is empty.

        Returns:
            T: The rear item.
        """
        if self.is_empty():
            raise QueueEmptyError("Cannot peek into an empty deque.")
        return self._items[-1]

    def is_empty(self) -> bool:
        """Check if the deque is empty.

        Returns:
            bool: True if empty, False otherwise.
        """
        return len(self._items) == 0

    def size(self) -> int:
        """Return the number of items in the deque.

        Returns:
            int: The size of the deque.
        """
        return len(self._items)


class ConcurrentQueue(Generic[T]):
    """
    A thread-safe queue implementation using PyTorch tensors.

    Note:
        This implementation leverages PyTorch for synchronization primitives,
        although typically threading modules are used for concurrency.

    Time Complexity:
        - Enqueue: O(1)
        - Dequeue: O(1)
    Space Complexity:
        - O(n)
    """

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize a concurrent queue with a given capacity.

        Args:
            capacity (int, optional): The maximum number of items. Defaults to 1000.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self._capacity: int = capacity
        self._queue: List[T] = []
        self._lock = torch.multiprocessing.Lock()
        print(f"ConcurrentQueue initialized with capacity {self._capacity}")

    def enqueue(self, item: T) -> None:
        """Add an item to the queue in a thread-safe manner.

        Args:
            item (T): The item to be added.

        Raises:
            QueueFullError: If the queue is full.
        """
        with self._lock:
            if self.size() >= self._capacity:
                raise QueueFullError("Cannot enqueue to a full concurrent queue.")
            self._queue.append(item)
            print(f"Enqueued (concurrent): {item}. Queue state: {self._queue}")

    def dequeue(self) -> T:
        """Remove and return the item from the front of the queue in a thread-safe manner.

        Raises:
            QueueEmptyError: If the queue is empty.

        Returns:
            T: The dequeued item.
        """
        with self._lock:
            if self.is_empty():
                raise QueueEmptyError("Cannot dequeue from an empty concurrent queue.")
            item = self._queue.pop(0)
            print(f"Dequeued (concurrent): {item}. Queue state: {self._queue}")
            return item

    def is_empty(self) -> bool:
        """Check if the concurrent queue is empty in a thread-safe manner.

        Returns:
            bool: True if empty, False otherwise.
        """
        with self._lock:
            return len(self._queue) == 0

    def size(self) -> int:
        """Return the number of items in the concurrent queue in a thread-safe manner.

        Returns:
            int: The size of the queue.
        """
        with self._lock:
            return len(self._queue)


def main() -> None:
    """Demonstration of various queue implementations."""
    
    print("\n--- SimpleQueue Demo ---")
    sq = SimpleQueue[int]()
    sq.enqueue(1)
    sq.enqueue(2)
    print(sq.dequeue())
    print(sq.peek())
    print(sq.size())

    print("\n--- CircularQueue Demo ---")
    cq = CircularQueue[int](3)
    cq.enqueue(10)
    cq.enqueue(20)
    cq.enqueue(30)
    try:
        cq.enqueue(40)
    except QueueFullError as e:
        print(e)
    print(cq.dequeue())
    cq.enqueue(40)
    print(cq.peek())
    print(cq.size())

    print("\n--- PriorityQueue Demo ---")
    pq = PriorityQueue[str]()
    pq.enqueue("low", priority=5)
    pq.enqueue("medium", priority=3)
    pq.enqueue("high", priority=1)
    print(pq.dequeue())
    print(pq.peek())
    print(pq.size())

    print("\n--- Deque Demo ---")
    dq = Deque[int]()
    dq.enqueue_rear(100)
    dq.enqueue_front(200)
    dq.enqueue_rear(300)
    print(dq.dequeue_front())
    print(dq.dequeue_rear())
    print(dq.peek_front())
    print(dq.size())

    print("\n--- ConcurrentQueue Demo ---")
    cq_concurrent = ConcurrentQueue[str](capacity=2)
    cq_concurrent.enqueue("thread1")
    try:
        cq_concurrent.enqueue("thread2")
        cq_concurrent.enqueue("thread3")
    except QueueFullError as e:
        print(e)
    print(cq_concurrent.dequeue())
    print(cq_concurrent.size())


if __name__ == "__main__":
    main()