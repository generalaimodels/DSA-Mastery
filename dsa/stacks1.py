"""
Stacks Data Structure: From Basics to Advanced Implementations

This module provides a comprehensive overview of the Stack data structure,
covering basic concepts to advanced implementations. It includes various
approaches to implementing stacks in Python, ensuring adherence to best practices,
optimization for performance, and scalability.

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from typing import Any, Generic, List, Optional, TypeVar

# Define a generic type variable for stack elements
T = TypeVar('T')


class StackEmptyError(Exception):
    """Custom exception to indicate that the stack is empty."""
    pass


class StackFullError(Exception):
    """Custom exception to indicate that the stack is full."""
    pass


class ArrayStack(Generic[T]):
    """
    A stack implementation using a dynamic array (Python list).

    This implementation provides average O(1) time complexity for push and pop operations.
    It dynamically resizes to accommodate additional elements, ensuring scalability.
    """

    def __init__(self, initial_capacity: int = 10) -> None:
        """
        Initialize the stack with an optional initial capacity.

        :param initial_capacity: The initial size of the internal storage.
        """
        if initial_capacity <= 0:
            raise ValueError("Initial capacity must be positive.")
        self._data: List[Optional[T]] = [None] * initial_capacity
        self._size: int = 0
        print(f"Initialized ArrayStack with capacity {initial_capacity}.")

    def is_empty(self) -> bool:
        """
        Check if the stack is empty.

        :return: True if stack is empty, False otherwise.
        """
        return self._size == 0

    def push(self, item: T) -> None:
        """
        Push an item onto the top of the stack.

        :param item: The item to be added to the stack.
        """
        if self._size == len(self._data):
            self._resize(2 * len(self._data))  # Double the capacity
            print(f"Resized internal array to {2 * len(self._data)}.")
        self._data[self._size] = item
        self._size += 1
        print(f"Pushed item: {item}. Stack size is now {self._size}.")

    def pop(self) -> T:
        """
        Remove and return the top item of the stack.

        :return: The item at the top of the stack.
        :raises StackEmptyError: If the stack is empty.
        """
        if self.is_empty():
            raise StackEmptyError("Cannot pop from an empty stack.")
        self._size -= 1
        item = self._data[self._size]
        self._data[self._size] = None  # Help garbage collection
        print(f"Popped item: {item}. Stack size is now {self._size}.")
        # Shrink the internal array if necessary
        if 0 < self._size < len(self._data) // 4:
            self._resize(len(self._data) // 2)
            print(f"Resized internal array to {len(self._data)}.")
        assert item is not None  # For type checker
        return item

    def peek(self) -> T:
        """
        Return the top item of the stack without removing it.

        :return: The item at the top of the stack.
        :raises StackEmptyError: If the stack is empty.
        """
        if self.is_empty():
            raise StackEmptyError("Cannot peek from an empty stack.")
        item = self._data[self._size - 1]
        print(f"Peeked item: {item}.")
        assert item is not None  # For type checker
        return item

    def _resize(self, new_capacity: int) -> None:
        """
        Resize the internal storage array to a new capacity.

        :param new_capacity: The new capacity of the internal array.
        """
        new_data: List[Optional[T]] = [None] * new_capacity
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        print(f"Internal array resized to {new_capacity}.")

    def __len__(self) -> int:
        """
        Return the number of items in the stack.

        :return: The size of the stack.
        """
        return self._size

    def __repr__(self) -> str:
        """
        Return a string representation of the stack.

        :return: String representation.
        """
        items = ', '.join(repr(self._data[i]) for i in range(self._size))
        return f"ArrayStack([{items}])"


class LinkedListNode(Generic[T]):
    """
    A node in a singly linked list.
    """

    def __init__(self, data: T, next_node: Optional['LinkedListNode[T]'] = None) -> None:
        """
        Initialize a new node.

        :param data: The data stored in the node.
        :param next_node: Reference to the next node in the list.
        """
        self.data = data
        self.next = next_node
        print(f"Created LinkedListNode with data: {data}.")

    def __repr__(self) -> str:
        """
        Return a string representation of the node.

        :return: String representation.
        """
        return f"LinkedListNode({self.data})"


class LinkedListStack(Generic[T]):
    """
    A stack implementation using a singly linked list.

    This implementation provides O(1) time complexity for push and pop operations.
    It is highly efficient for scenarios with frequent insertions and deletions.
    """

    def __init__(self) -> None:
        """
        Initialize an empty stack.
        """
        self._head: Optional[LinkedListNode[T]] = None
        self._size: int = 0
        print("Initialized LinkedListStack.")

    def is_empty(self) -> bool:
        """
        Check if the stack is empty.

        :return: True if stack is empty, False otherwise.
        """
        return self._head is None

    def push(self, item: T) -> None:
        """
        Push an item onto the top of the stack.

        :param item: The item to be added to the stack.
        """
        new_node = LinkedListNode(item, self._head)
        self._head = new_node
        self._size += 1
        print(f"Pushed item: {item}. Stack size is now {self._size}.")

    def pop(self) -> T:
        """
        Remove and return the top item of the stack.

        :return: The item at the top of the stack.
        :raises StackEmptyError: If the stack is empty.
        """
        if self.is_empty():
            raise StackEmptyError("Cannot pop from an empty stack.")
        assert self._head is not None  # For type checker
        item = self._head.data
        self._head = self._head.next
        self._size -= 1
        print(f"Popped item: {item}. Stack size is now {self._size}.")
        return item

    def peek(self) -> T:
        """
        Return the top item of the stack without removing it.

        :return: The item at the top of the stack.
        :raises StackEmptyError: If the stack is empty.
        """
        if self.is_empty():
            raise StackEmptyError("Cannot peek from an empty stack.")
        assert self._head is not None  # For type checker
        print(f"Peeked item: {self._head.data}.")
        return self._head.data

    def __len__(self) -> int:
        """
        Return the number of items in the stack.

        :return: The size of the stack.
        """
        return self._size

    def __repr__(self) -> str:
        """
        Return a string representation of the stack.

        :return: String representation.
        """
        nodes = []
        current = self._head
        while current:
            nodes.append(repr(current.data))
            current = current.next
        return f"LinkedListStack([{', '.join(nodes)}])"


class OptimizedStack(Generic[T]):
    """
    An optimized stack implementation leveraging both array and linked list advantages.

    This hybrid approach ensures constant time operations while optimizing memory usage.
    Suitable for applications requiring high performance and efficient memory management.
    """

    def __init__(self, initial_capacity: int = 10) -> None:
        """
        Initialize the optimized stack with an initial capacity.

        :param initial_capacity: The initial size of the internal storage.
        """
        self._array: List[Optional[T]] = [None] * initial_capacity
        self._size: int = 0
        print(f"Initialized OptimizedStack with capacity {initial_capacity}.")

    def is_empty(self) -> bool:
        """
        Check if the stack is empty.

        :return: True if stack is empty, False otherwise.
        """
        return self._size == 0

    def push(self, item: T) -> None:
        """
        Push an item onto the stack.

        :param item: The item to add.
        """
        if self._size == len(self._array):
            self._resize(2 * len(self._array))
            print(f"Resized internal array to {2 * len(self._array)}.")
        self._array[self._size] = item
        self._size += 1
        print(f"Pushed item: {item}. Size is now {self._size}.")

    def pop(self) -> T:
        """
        Pop the top item from the stack.

        :return: The popped item.
        :raises StackEmptyError: If the stack is empty.
        """
        if self.is_empty():
            raise StackEmptyError("Cannot pop from an empty stack.")
        self._size -= 1
        item = self._array[self._size]
        self._array[self._size] = None
        print(f"Popped item: {item}. Size is now {self._size}.")
        if 0 < self._size < len(self._array) // 4:
            self._resize(len(self._array) // 2)
            print(f"Resized internal array to {len(self._array)}.")
        assert item is not None
        return item

    def peek(self) -> T:
        """
        Peek at the top item of the stack without removing it.

        :return: The top item.
        :raises StackEmptyError: If the stack is empty.
        """
        if self.is_empty():
            raise StackEmptyError("Cannot peek from an empty stack.")
        item = self._array[self._size - 1]
        print(f"Peeked item: {item}.")
        assert item is not None
        return item

    def _resize(self, new_capacity: int) -> None:
        """
        Resize the internal array to a new capacity.

        :param new_capacity: The new capacity.
        """
        new_array: List[Optional[T]] = [None] * new_capacity
        for i in range(self._size):
            new_array[i] = self._array[i]
        self._array = new_array
        print(f"Internal array resized to {new_capacity}.")

    def __len__(self) -> int:
        """
        Get the number of items in the stack.

        :return: The stack size.
        """
        return self._size

    def __repr__(self) -> str:
        """
        Get the string representation of the stack.

        :return: String representation.
        """
        items = ', '.join(repr(self._array[i]) for i in range(self._size))
        return f"OptimizedStack([{items}])"


# Advanced Applications and Algorithms Utilizing Stacks

def balanced_parentheses(expression: str) -> bool:
    """
    Check if the parentheses in the expression are balanced.

    Supports (), {}, and [].

    :param expression: The string expression to check.
    :return: True if balanced, False otherwise.
    """
    stack = ArrayStack[str]()
    pairs = {')': '(', '}': '{', ']': '['}
    for char in expression:
        if char in pairs.values():
            stack.push(char)
            print(f"Encountered opening bracket: {char}.")
        elif char in pairs:
            if stack.is_empty():
                print(f"Unmatched closing bracket: {char}.")
                return False
            top = stack.pop()
            if top != pairs[char]:
                print(f"Mismatched brackets: {top} != {pairs[char]}.")
                return False
            print(f"Matched closing bracket: {char} with {top}.")
    if stack.is_empty():
        print("All brackets are balanced.")
        return True
    else:
        print("Some brackets are not closed.")
        return False


def evaluate_postfix(expression: str) -> int:
    """
    Evaluate a postfix (Reverse Polish notation) expression.

    Supports +, -, *, / operators.

    :param expression: The postfix expression as a string with space-separated tokens.
    :return: The integer result of the evaluation.
    :raises ValueError: If the expression is invalid.
    """
    stack = ArrayStack[int]()
    operators = {'+', '-', '*', '/'}
    tokens = expression.split()
    print(f"Evaluating postfix expression: {expression}")

    for token in tokens:
        if token.isdigit():
            stack.push(int(token))
            print(f"Pushed number: {token}")
        elif token in operators:
            if len(stack) < 2:
                raise ValueError("Insufficient operands.")
            b = stack.pop()
            a = stack.pop()
            print(f"Popped operands: a={a}, b={b}")
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                if b == 0:
                    raise ZeroDivisionError("Division by zero.")
                result = a // b  # Integer division
            stack.push(result)
            print(f"Performed operation: {a} {token} {b} = {result}")
        else:
            raise ValueError(f"Invalid token: {token}")

    if len(stack) != 1:
        raise ValueError("The expression is invalid.")
    final_result = stack.pop()
    print(f"Final result of postfix expression: {final_result}")
    return final_result


def sort_stack(input_stack: ArrayStack[int]) -> ArrayStack[int]:
    """
    Sort a stack such that the smallest items are on the top using only one additional stack.

    :param input_stack: The stack to sort.
    :return: A new sorted stack.
    """
    auxiliary_stack = ArrayStack[int]()
    print("Starting to sort the stack.")

    while not input_stack.is_empty():
        tmp = input_stack.pop()
        print(f"Temporary popped item: {tmp}")
        while not auxiliary_stack.is_empty() and auxiliary_stack.peek() > tmp:
            moved_item = auxiliary_stack.pop()
            input_stack.push(moved_item)
            print(f"Moved item back to input stack: {moved_item}")

        auxiliary_stack.push(tmp)
        print(f"Pushed item to auxiliary stack: {tmp}")

    print("Stack sorted.")
    return auxiliary_stack


# Test Cases (Can be removed or commented out in production)

def _test_stacks():
    """
    Run test cases to validate stack implementations and related algorithms.
    """
    print("\n--- Testing ArrayStack ---")
    array_stack = ArrayStack[int]()
    for i in range(5):
        array_stack.push(i)
    assert array_stack.peek() == 4
    assert array_stack.pop() == 4
    assert len(array_stack) == 4

    print("\n--- Testing LinkedListStack ---")
    linked_stack = LinkedListStack[str]()
    for char in ['a', 'b', 'c']:
        linked_stack.push(char)
    assert linked_stack.peek() == 'c'
    assert linked_stack.pop() == 'c'
    assert len(linked_stack) == 2

    print("\n--- Testing Balanced Parentheses ---")
    assert balanced_parentheses("(([]){})") is True
    assert balanced_parentheses("([)]") is False

    print("\n--- Testing Postfix Evaluation ---")
    assert evaluate_postfix("3 4 + 2 * 7 /") == 2  # ((3 + 4) * 2) / 7 = 2

    print("\n--- Testing Stack Sorting ---")
    unsorted_stack = ArrayStack[int]()
    for num in [3, 1, 4, 2]:
        unsorted_stack.push(num)
    sorted_stack = sort_stack(unsorted_stack)
    sorted_list = []
    while not sorted_stack.is_empty():
        sorted_list.append(sorted_stack.pop())
    assert sorted_list == [4, 3, 2, 1]

    print("\nAll tests passed successfully.")


if __name__ == "__main__":
    _test_stacks()