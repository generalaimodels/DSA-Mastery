"""
Stacks Module
=============

This module provides comprehensive implementations of the Stack data structure, ranging from basic to advanced levels.
It includes various stack implementations, applications, and algorithms to demonstrate the versatility and robustness
of stacks in computer science.

Author: Generalmodelai-agent
Date: 2024-10-15
"""

from typing import Any, Optional, Generic, TypeVar

T = TypeVar('T')


class StackEmptyError(Exception):
    """Custom exception to indicate that the stack is empty when an operation requiring non-empty stack is attempted."""
    pass


class Node(Generic[T]):
    """
    A node in a singly linked list used for the linked-list implementation of the stack.
    
    Attributes:
        value (T): The value stored in the node.
        next (Optional[Node[T]]): The reference to the next node in the stack.
    """
    
    def __init__(self, value: T) -> None:
        self.value: T = value
        self.next: Optional['Node[T]'] = None


class ArrayStack(Generic[T]):
    """
    Stack implementation using Python's built-in list (dynamic array).
    
    Provides O(1) time complexity for push and pop operations on average.
    """
    
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        """
        Push an item onto the top of the stack.
        
        Args:
            item (T): The item to be pushed.
        """
        self._items.append(item)
        print(f"Pushed {item}. Stack now: {self._items}")
    
    def pop(self) -> T:
        """
        Remove and return the top item of the stack.
        
        Raises:
            StackEmptyError: If the stack is empty.
        
        Returns:
            T: The item at the top of the stack.
        """
        if self.is_empty():
            raise StackEmptyError("Pop from an empty stack.")
        item = self._items.pop()
        print(f"Popped {item}. Stack now: {self._items}")
        return item
    
    def peek(self) -> T:
        """
        Return the top item of the stack without removing it.
        
        Raises:
            StackEmptyError: If the stack is empty.
        
        Returns:
            T: The item at the top of the stack.
        """
        if self.is_empty():
            raise StackEmptyError("Peek from an empty stack.")
        item = self._items[-1]
        print(f"Peeked at {item}. Stack remains: {self._items}")
        return item
    
    def is_empty(self) -> bool:
        """
        Check whether the stack is empty.
        
        Returns:
            bool: True if the stack is empty, False otherwise.
        """
        empty = len(self._items) == 0
        print(f"Stack is_empty check: {empty}")
        return empty
    
    def size(self) -> int:
        """
        Return the number of items in the stack.
        
        Returns:
            int: The size of the stack.
        """
        size = len(self._items)
        print(f"Stack size: {size}")
        return size
    
    def __repr__(self) -> str:
        return f"ArrayStack({self._items})"


class LinkedListStack(Generic[T]):
    """
    Stack implementation using a singly linked list.
    
    Provides O(1) time complexity for push and pop operations.
    """
    
    def __init__(self) -> None:
        self._head: Optional[Node[T]] = None
        self._count: int = 0
    
    def push(self, item: T) -> None:
        """
        Push an item onto the top of the stack.
        
        Args:
            item (T): The item to be pushed.
        """
        new_node = Node(item)
        new_node.next = self._head
        self._head = new_node
        self._count += 1
        print(f"Pushed {item}. Stack size is now {self._count}")
    
    def pop(self) -> T:
        """
        Remove and return the top item of the stack.
        
        Raises:
            StackEmptyError: If the stack is empty.
        
        Returns:
            T: The item at the top of the stack.
        """
        if self.is_empty():
            raise StackEmptyError("Pop from an empty stack.")
        assert self._head is not None  # for type checker
        item = self._head.value
        self._head = self._head.next
        self._count -= 1
        print(f"Popped {item}. Stack size is now {self._count}")
        return item
    
    def peek(self) -> T:
        """
        Return the top item of the stack without removing it.
        
        Raises:
            StackEmptyError: If the stack is empty.
        
        Returns:
            T: The item at the top of the stack.
        """
        if self.is_empty():
            raise StackEmptyError("Peek from an empty stack.")
        assert self._head is not None  # for type checker
        item = self._head.value
        print(f"Peeked at {item}. Stack size remains {self._count}")
        return item
    
    def is_empty(self) -> bool:
        """
        Check whether the stack is empty.
        
        Returns:
            bool: True if the stack is empty, False otherwise.
        """
        empty = self._head is None
        print(f"Stack is_empty check: {empty}")
        return empty
    
    def size(self) -> int:
        """
        Return the number of items in the stack.
        
        Returns:
            int: The size of the stack.
        """
        print(f"Stack size: {self._count}")
        return self._count
    
    def __repr__(self) -> str:
        items = []
        current = self._head
        while current:
            items.append(current.value)
            current = current.next
        return f"LinkedListStack({items})"


class MinStack(Generic[T]):
    """
    Stack implementation that supports retrieving the minimum element in constant time.
    
    Uses an auxiliary stack to keep track of minimum values.
    """
    
    def __init__(self) -> None:
        self._main_stack: list[T] = []
        self._min_stack: list[T] = []
    
    def push(self, item: T) -> None:
        """
        Push an item onto the stack and update the minimum stack.
        
        Args:
            item (T): The item to be pushed.
        """
        self._main_stack.append(item)
        if not self._min_stack or item <= self._min_stack[-1]:
            self._min_stack.append(item)
        print(f"Pushed {item}. Main stack: {self._main_stack}, Min stack: {self._min_stack}")
    
    def pop(self) -> T:
        """
        Remove and return the top item of the stack, updating the minimum stack accordingly.
        
        Raises:
            StackEmptyError: If the stack is empty.
        
        Returns:
            T: The item at the top of the stack.
        """
        if self.is_empty():
            raise StackEmptyError("Pop from an empty stack.")
        item = self._main_stack.pop()
        if item == self._min_stack[-1]:
            self._min_stack.pop()
        print(f"Popped {item}. Main stack: {self._main_stack}, Min stack: {self._min_stack}")
        return item
    
    def peek(self) -> T:
        """
        Return the top item of the stack without removing it.
        
        Returns:
            T: The item at the top of the stack.
        """
        if self.is_empty():
            raise StackEmptyError("Peek from an empty stack.")
        item = self._main_stack[-1]
        print(f"Peeked at {item}. Main stack remains: {self._main_stack}")
        return item
    
    def get_min(self) -> T:
        """
        Retrieve the minimum item in the stack in constant time.
        
        Returns:
            T: The current minimum item in the stack.
        """
        if not self._min_stack:
            raise StackEmptyError("Get min from an empty stack.")
        min_item = self._min_stack[-1]
        print(f"Current minimum: {min_item}")
        return min_item
    
    def is_empty(self) -> bool:
        """
        Check if the stack is empty.
        
        Returns:
            bool: True if empty, False otherwise.
        """
        empty = len(self._main_stack) == 0
        print(f"Stack is_empty check: {empty}")
        return empty
    
    def size(self) -> int:
        """
        Return the number of items in the stack.
        
        Returns:
            int: The size of the stack.
        """
        size = len(self._main_stack)
        print(f"Stack size: {size}")
        return size
    
    def __repr__(self) -> str:
        return f"MinStack(main={self._main_stack}, min={self._min_stack})"


class DynamicArrayStack(Generic[T]):
    """
    Stack implementation using a fixed-size array with dynamic resizing.
    
    Mimics the behavior of dynamic arrays by doubling the capacity when needed.
    """
    
    def __init__(self, initial_capacity: int = 10) -> None:
        if initial_capacity <= 0:
            raise ValueError("Initial capacity must be positive.")
        self._capacity: int = initial_capacity
        self._array: list[Optional[T]] = [None] * self._capacity
        self._top: int = -1
        print(f"Initialized DynamicArrayStack with capacity {self._capacity}")
    
    def push(self, item: T) -> None:
        """
        Push an item onto the stack, resizing the internal array if necessary.
        
        Args:
            item (T): The item to be pushed.
        """
        if self._top + 1 == self._capacity:
            self._resize(2 * self._capacity)
        self._top += 1
        self._array[self._top] = item
        print(f"Pushed {item}. Stack: {self._array[:self._top+1]}")
    
    def pop(self) -> T:
        """
        Remove and return the top item of the stack, resizing if necessary.
        
        Raises:
            StackEmptyError: If the stack is empty.
        
        Returns:
            T: The item at the top of the stack.
        """
        if self.is_empty():
            raise StackEmptyError("Pop from an empty stack.")
        item = self._array[self._top]
        self._array[self._top] = None  # Avoid loitering
        self._top -= 1
        if 0 < self._top + 1 < self._capacity // 4:
            self._resize(self._capacity // 2)
        print(f"Popped {item}. Stack: {self._array[:self._top+1]}")
        return item  # type: ignore
    
    def peek(self) -> T:
        """
        Return the top item of the stack without removing it.
        
        Raises:
            StackEmptyError: If the stack is empty.
        
        Returns:
            T: The item at the top of the stack.
        """
        if self.is_empty():
            raise StackEmptyError("Peek from an empty stack.")
        item = self._array[self._top]
        print(f"Peeked at {item}. Stack remains: {self._array[:self._top+1]}")
        return item  # type: ignore
    
    def is_empty(self) -> bool:
        """
        Check if the stack is empty.
        
        Returns:
            bool: True if empty, False otherwise.
        """
        empty = self._top == -1
        print(f"Stack is_empty check: {empty}")
        return empty
    
    def size(self) -> int:
        """
        Return the number of items in the stack.
        
        Returns:
            int: The size of the stack.
        """
        size = self._top + 1
        print(f"Stack size: {size}")
        return size
    
    def _resize(self, new_capacity: int) -> None:
        """
        Resize the internal array to a new capacity.
        
        Args:
            new_capacity (int): The new capacity of the array.
        """
        print(f"Resizing stack from {self._capacity} to {new_capacity}")
        new_array: list[Optional[T]] = [None] * new_capacity
        for i in range(self._top + 1):
            new_array[i] = self._array[i]
        self._array = new_array
        self._capacity = new_capacity
    
    def __repr__(self) -> str:
        return f"DynamicArrayStack({self._array[:self._top+1]})"


def evaluate_postfix(expression: str) -> int:
    """
    Evaluate a postfix (Reverse Polish Notation) expression using a stack.
    
    Args:
        expression (str): The postfix expression to evaluate.
    
    Raises:
        ValueError: If the expression is invalid.
    
    Returns:
        int: The result of the evaluated expression.
    """
    stack = ArrayStack[int]()
    operators = {'+', '-', '*', '/'}
    tokens = expression.split()
    
    for token in tokens:
        if token.isdigit():
            stack.push(int(token))
        elif token in operators:
            try:
                right = stack.pop()
                left = stack.pop()
                result = 0
                if token == '+':
                    result = left + right
                elif token == '-':
                    result = left - right
                elif token == '*':
                    result = left * right
                elif token == '/':
                    if right == 0:
                        raise ValueError("Division by zero.")
                    result = int(left / right)  # Assuming integer division
                stack.push(result)
                print(f"Applied operator {token}: {left} {token} {right} = {result}")
            except StackEmptyError:
                raise ValueError("Invalid postfix expression.")
        else:
            raise ValueError(f"Unknown token: {token}")
    
    if stack.size() != 1:
        raise ValueError("Invalid postfix expression.")
    
    result = stack.pop()
    print(f"Final result of postfix expression '{expression}': {result}")
    return result


def is_balanced_parentheses(expression: str) -> bool:
    """
    Check if the parentheses in the expression are balanced using a stack.
    
    Args:
        expression (str): The expression to check.
    
    Returns:
        bool: True if balanced, False otherwise.
    """
    stack = ArrayStack[str]()
    pairs = {')': '(', '}': '{', ']': '['}
    
    for char in expression:
        if char in pairs.values():
            stack.push(char)
        elif char in pairs:
            if stack.is_empty() or stack.pop() != pairs[char]:
                print(f"Unbalanced at character: {char}")
                return False
    balanced = stack.is_empty()
    print(f"Parentheses balanced: {balanced}")
    return balanced


def reverse_string(s: str) -> str:
    """
    Reverse a string using a stack.
    
    Args:
        s (str): The string to reverse.
    
    Returns:
        str: The reversed string.
    """
    stack = ArrayStack[str]()
    for char in s:
        stack.push(char)
    reversed_chars = []
    while not stack.is_empty():
        reversed_chars.append(stack.pop())
    reversed_str = ''.join(reversed_chars)
    print(f"Original string: '{s}', Reversed string: '{reversed_str}'")
    return reversed_str


def main() -> None:
    """
    Main function to demonstrate stack implementations and applications.
    """
    print("=== ArrayStack Demonstration ===")
    array_stack = ArrayStack[int]()
    array_stack.push(1)
    array_stack.push(2)
    array_stack.push(3)
    print(array_stack.pop())
    print(array_stack.peek())
    print(f"Is stack empty? {array_stack.is_empty()}")
    print(f"Stack size: {array_stack.size()}")
    print(array_stack)
    
    print("\n=== LinkedListStack Demonstration ===")
    linked_list_stack = LinkedListStack[str]()
    linked_list_stack.push("a")
    linked_list_stack.push("b")
    linked_list_stack.push("c")
    print(linked_list_stack.pop())
    print(linked_list_stack.peek())
    print(f"Is stack empty? {linked_list_stack.is_empty()}")
    print(f"Stack size: {linked_list_stack.size()}")
    print(linked_list_stack)
    
    print("\n=== MinStack Demonstration ===")
    min_stack = MinStack[int]()
    min_stack.push(5)
    min_stack.push(3)
    min_stack.push(7)
    min_stack.push(3)
    print(f"Current min: {min_stack.get_min()}")
    min_stack.pop()
    print(f"Current min after pop: {min_stack.get_min()}")
    
    print("\n=== DynamicArrayStack Demonstration ===")
    dynamic_stack = DynamicArrayStack[int](initial_capacity=2)
    dynamic_stack.push(10)
    dynamic_stack.push(20)
    dynamic_stack.push(30)
    print(dynamic_stack.pop())
    print(dynamic_stack.peek())
    print(f"Stack size: {dynamic_stack.size()}")
    print(dynamic_stack)
    
    print("\n=== Postfix Expression Evaluation ===")
    postfix_expr = "3 4 + 2 * 7 /"
    result = evaluate_postfix(postfix_expr)
    print(f"Postfix evaluation result: {result}")
    
    print("\n=== Balanced Parentheses Check ===")
    expression = "{[(a+b)*(c+d)]}"
    balanced = is_balanced_parentheses(expression)
    print(f"Is the expression '{expression}' balanced? {balanced}")
    
    print("\n=== String Reversal ===")
    original_str = "Hello, World!"
    reversed_str = reverse_string(original_str)
    print(f"Reversed string: {reversed_str}")


if __name__ == "__main__":
    main()