"""
greedy_algorithms.py

A comprehensive guide to Greedy Algorithms in Python, covering basic to advanced topics.
This module adheres to PEP-8 standards, utilizes type hints from the `typing` module,
and ensures optimized, maintainable, and robust code.

"""

from typing import List, Tuple, Optional
import heapq  # Used internally for priority queues in some algorithms


class GreedyAlgorithmError(Exception):
    """Custom exception class for Greedy Algorithm errors."""
    pass


def validate_positive_integer(value: int, name: str) -> None:
    """
    Validates that a given value is a positive integer.

    Args:
        value (int): The value to validate.
        name (str): The name of the parameter (for error messages).

    Raises:
        GreedyAlgorithmError: If the value is not a positive integer.
    """
    if not isinstance(value, int) or value <= 0:
        raise GreedyAlgorithmError(f"{name} must be a positive integer.")


class Activity:
    """
    Represents an activity with a start and finish time.

    Attributes:
        start (int): The start time of the activity.
        finish (int): The finish time of the activity.
    """

    def __init__(self, start: int, finish: int) -> None:
        """
        Initializes an Activity instance.

        Args:
            start (int): Start time.
            finish (int): Finish time.

        Raises:
            GreedyAlgorithmError: If start or finish times are invalid.
        """
        validate_positive_integer(start, "Start time")
        validate_positive_integer(finish, "Finish time")
        if start >= finish:
            raise GreedyAlgorithmError("Start time must be less than finish time.")
        self.start = start
        self.finish = finish

    def __repr__(self) -> str:
        return f"Activity(start={self.start}, finish={self.finish})"


def activity_selection(activities: List[Activity]) -> List[Activity]:
    """
    Selects the maximum number of non-overlapping activities using a greedy approach.

    Args:
        activities (List[Activity]): A list of Activity instances.

    Returns:
        List[Activity]: A list of selected activities.

    Raises:
        GreedyAlgorithmError: If the activities list is empty.
    """
    if not activities:
        raise GreedyAlgorithmError("The activities list cannot be empty.")

    # Sort activities based on their finish times
    sorted_activities = sorted(activities, key=lambda activity: activity.finish)
    selected = [sorted_activities[0]]

    for activity in sorted_activities[1:]:
        if activity.start >= selected[-1].finish:
            selected.append(activity)

    return selected


def fractional_knapsack(weights: List[float], values: List[float], capacity: float) -> float:
    """
    Solves the Fractional Knapsack problem using a greedy approach.

    Args:
        weights (List[float]): List of item weights.
        values (List[float]): List of item values.
        capacity (float): Maximum capacity of the knapsack.

    Returns:
        float: The maximum value achievable.

    Raises:
        GreedyAlgorithmError: If input lists are of unequal length or capacity is invalid.
    """
    if len(weights) != len(values):
        raise GreedyAlgorithmError("Weights and values must be of the same length.")
    if capacity <= 0:
        raise GreedyAlgorithmError("Capacity must be a positive number.")

    # Calculate value per unit weight and sort items accordingly
    index = list(range(len(weights)))
    index.sort(key=lambda i: values[i] / weights[i], reverse=True)

    total_value = 0.0
    for i in index:
        if weights[i] <= capacity:
            capacity -= weights[i]
            total_value += values[i]
        else:
            total_value += values[i] * (capacity / weights[i])
            break

    return total_value


def huffman_encoding(symbols: List[str], frequencies: List[int]) -> dict:
    """
    Constructs Huffman Codes for given symbols and their frequencies.

    Args:
        symbols (List[str]): List of symbols.
        frequencies (List[int]): Corresponding frequencies of the symbols.

    Returns:
        dict: A dictionary mapping symbols to their Huffman codes.

    Raises:
        GreedyAlgorithmError: If input lists are of unequal length or empty.
    """
    if len(symbols) != len(frequencies) or not symbols:
        raise GreedyAlgorithmError("Symbols and frequencies must be of the same non-zero length.")

    heap = [[freq, [symbol, ""]] for symbol, freq in zip(symbols, frequencies)]
    heapq.heapify(heap)

    while len(heap) > 1:
        low1 = heapq.heappop(heap)
        low2 = heapq.heappop(heap)
        for pair in low1[1:]:
            pair[1] = '0' + pair[1]
        for pair in low2[1:]:
            pair[1] = '1' + pair[1]
        merged = [low1[0] + low2[0]] + low1[1:] + low2[1:]
        heapq.heappush(heap, merged)

    huffman_code = sorted(heap[0][1:], key=lambda p: p[0])
    return {symbol: code for symbol, code in huffman_code}


class Graph:
    """
    Represents a graph for Prim's and Dijkstra's algorithms.

    Attributes:
        vertices (int): Number of vertices.
        edges (dict): Adjacency list representing edges and their weights.
    """

    def __init__(self, vertices: int) -> None:
        """
        Initializes a Graph instance.

        Args:
            vertices (int): Number of vertices.

        Raises:
            GreedyAlgorithmError: If the number of vertices is not positive.
        """
        validate_positive_integer(vertices, "Number of vertices")
        self.vertices = vertices
        self.edges = {i: [] for i in range(vertices)}

    def add_edge(self, u: int, v: int, weight: float) -> None:
        """
        Adds an undirected edge to the graph.

        Args:
            u (int): One vertex.
            v (int): The other vertex.
            weight (float): Weight of the edge.

        Raises:
            GreedyAlgorithmError: If vertex indices are out of bounds or weight is non-positive.
        """
        if not (0 <= u < self.vertices) or not (0 <= v < self.vertices):
            raise GreedyAlgorithmError("Vertex index out of bounds.")
        if weight <= 0:
            raise GreedyAlgorithmError("Edge weight must be positive.")
        self.edges[u].append((v, weight))
        self.edges[v].append((u, weight))

    def prim_mst(self) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Computes the Minimum Spanning Tree (MST) using Prim's algorithm.

        Returns:
            Tuple[float, List[Tuple[int, int, float]]]: Total weight of MST and list of edges in MST.

        Raises:
            GreedyAlgorithmError: If the graph is disconnected.
        """
        total_weight = 0.0
        mst_edges: List[Tuple[int, int, float]] = []
        visited = [False] * self.vertices
        min_heap: List[Tuple[float, int, int]] = [(0, 0, -1)]  # (weight, vertex, parent)

        while min_heap:
            weight, u, parent = heapq.heappop(min_heap)
            if not visited[u]:
                visited[u] = True
                total_weight += weight
                if parent != -1:
                    mst_edges.append((parent, u, weight))
                for v, w in self.edges[u]:
                    if not visited[v]:
                        heapq.heappush(min_heap, (w, v, u))

        if not all(visited):
            raise GreedyAlgorithmError("The graph is disconnected; MST does not exist.")

        return total_weight, mst_edges

    def dijkstra_shortest_path(self, start: int) -> List[Optional[float]]:
        """
        Computes the shortest paths from a starting vertex using Dijkstra's algorithm.

        Args:
            start (int): The starting vertex.

        Returns:
            List[Optional[float]]: List of shortest distances from start to each vertex.

        Raises:
            GreedyAlgorithmError: If the starting vertex is out of bounds.
        """
        if not (0 <= start < self.vertices):
            raise GreedyAlgorithmError("Starting vertex is out of bounds.")

        distances = [float('inf')] * self.vertices
        distances[start] = 0
        min_heap: List[Tuple[float, int]] = [(0, start)]

        while min_heap:
            current_dist, u = heapq.heappop(min_heap)
            if current_dist > distances[u]:
                continue
            for v, w in self.edges[u]:
                if distances[v] > distances[u] + w:
                    distances[v] = distances[u] + w
                    heapq.heappush(min_heap, (distances[v], v))

        return distances


def main() -> None:
    """
    Demonstrates the usage of various greedy algorithms.
    """
    # Activity Selection Problem
    try:
        activities = [
            Activity(1, 4),
            Activity(3, 5),
            Activity(0, 6),
            Activity(5, 7),
            Activity(3, 9),
            Activity(5, 9),
            Activity(6, 10),
            Activity(8, 11),
            Activity(8, 12),
            Activity(2, 14),
            Activity(12, 16)
        ]
        selected = activity_selection(activities)
        print("Selected Activities:")
        for act in selected:
            print(act)
    except GreedyAlgorithmError as e:
        print(f"Activity Selection Error: {e}")

    print("\n" + "-"*50 + "\n")

    # Fractional Knapsack Problem
    try:
        weights = [10, 40, 20, 30]
        values = [60, 40, 100, 120]
        capacity = 50
        max_value = fractional_knapsack(weights, values, capacity)
        print(f"Maximum value in Fractional Knapsack: {max_value}")
    except GreedyAlgorithmError as e:
        print(f"Fractional Knapsack Error: {e}")

    print("\n" + "-"*50 + "\n")

    # Huffman Encoding
    try:
        symbols = ['a', 'b', 'c', 'd', 'e', 'f']
        frequencies = [5, 9, 12, 13, 16, 45]
        huffman_codes = huffman_encoding(symbols, frequencies)
        print("Huffman Codes:")
        for symbol in symbols:
            print(f"{symbol}: {huffman_codes[symbol]}")
    except GreedyAlgorithmError as e:
        print(f"Huffman Encoding Error: {e}")

    print("\n" + "-"*50 + "\n")

    # Prim's Minimum Spanning Tree
    try:
        g = Graph(5)
        g.add_edge(0, 1, 2)
        g.add_edge(0, 3, 6)
        g.add_edge(1, 2, 3)
        g.add_edge(1, 3, 8)
        g.add_edge(1, 4, 5)
        g.add_edge(2, 4, 7)
        g.add_edge(3, 4, 9)
        total_weight, mst = g.prim_mst()
        print(f"Total weight of MST: {total_weight}")
        print("Edges in MST:")
        for u, v, w in mst:
            print(f"{u} -- {v} == {w}")
    except GreedyAlgorithmError as e:
        print(f"Prim's MST Error: {e}")

    print("\n" + "-"*50 + "\n")

    # Dijkstra's Shortest Path
    try:
        g = Graph(9)
        g.add_edge(0, 1, 4)
        g.add_edge(0, 7, 8)
        g.add_edge(1, 2, 8)
        g.add_edge(1, 7, 11)
        g.add_edge(2, 3, 7)
        g.add_edge(2, 8, 2)
        g.add_edge(2, 5, 4)
        g.add_edge(3, 4, 9)
        g.add_edge(3, 5, 14)
        g.add_edge(4, 5, 10)
        g.add_edge(5, 6, 2)
        g.add_edge(6, 7, 1)
        g.add_edge(6, 8, 6)
        g.add_edge(7, 8, 7)
        start_vertex = 0
        distances = g.dijkstra_shortest_path(start_vertex)
        print(f"Shortest distances from vertex {start_vertex}:")
        for vertex, distance in enumerate(distances):
            print(f"To vertex {vertex}: {distance}")
    except GreedyAlgorithmError as e:
        print(f"Dijkstra's Shortest Path Error: {e}")


if __name__ == "__main__":
    main()