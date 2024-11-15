"""
greedy_algorithms.py

A comprehensive module exploring Greedy Algorithms from basic to advanced levels.
This module includes detailed implementations, explanations, and examples adhering to
PEP-8 standards, utilizing Python's typing module for clarity and maintainability.

Greedy algorithms build up a solution piece by piece, always choosing the next piece that
offers the most immediate benefit. They are used in optimization problems where finding
a locally optimal choice leads to a globally optimal solution.

This module covers various greedy algorithms, including but not limited to:
- Activity Selection
- Fractional Knapsack
- Huffman Coding
- Prim's and Kruskal's Minimum Spanning Tree
- Dijkstra's Shortest Path

Each algorithm is implemented with considerations for time and space complexity,
robustness, and scalability.
"""

from typing import List, Tuple, Dict, Any
import heapq


class GreedyAlgorithmError(Exception):
    """Custom exception class for Greedy Algorithm errors."""
    pass


def activity_selection(activities: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Selects the maximum number of non-overlapping activities using the Greedy approach.

    Args:
        activities (List[Tuple[int, int]]): A list of tuples where each tuple represents
                                            the start and end times of an activity.

    Returns:
        List[Tuple[int, int]]: A list of selected activities maximizing the number of non-overlapping activities.

    Raises:
        GreedyAlgorithmError: If the input list is empty.
    """
    if not activities:
        raise GreedyAlgorithmError("The list of activities is empty.")

    # Sort activities based on their finish times
    sorted_activities = sorted(activities, key=lambda x: x[1])
    selected = [sorted_activities[0]]

    for current in sorted_activities[1:]:
        last_selected = selected[-1]
        if current[0] >= last_selected[1]:
            selected.append(current)

    return selected


def fractional_knapsack(capacity: float, items: List[Tuple[float, float]]) -> float:
    """
    Solves the Fractional Knapsack problem using the Greedy approach.

    Args:
        capacity (float): The maximum capacity of the knapsack.
        items (List[Tuple[float, float]]): A list of tuples where each tuple contains
                                          the value and weight of an item.

    Returns:
        float: The maximum value achievable with the given capacity.

    Raises:
        GreedyAlgorithmError: If capacity is non-positive or items list is empty.
    """
    if capacity <= 0:
        raise GreedyAlgorithmError("Capacity must be greater than 0.")
    if not items:
        raise GreedyAlgorithmError("The items list is empty.")

    # Calculate value per unit weight and sort items by it in descending order
    items_sorted = sorted(items, key=lambda x: x[0] / x[1], reverse=True)

    total_value = 0.0
    for value, weight in items_sorted:
        if weight == 0:
            continue  # Avoid division by zero
        if weight <= capacity:
            capacity -= weight
            total_value += value
        else:
            total_value += value * (capacity / weight)
            break

    return total_value


def huffman_encoding(char_freq: Dict[str, int]) -> Dict[str, str]:
    """
    Constructs Huffman Codes for characters based on their frequencies.

    Args:
        char_freq (Dict[str, int]): A dictionary mapping characters to their frequencies.

    Returns:
        Dict[str, str]: A dictionary mapping characters to their Huffman codes.

    Raises:
        GreedyAlgorithmError: If the character frequency dictionary is empty.
    """
    if not char_freq:
        raise GreedyAlgorithmError("Character frequency dictionary is empty.")

    heap: List[Tuple[int, Any]] = [[freq, [char, ""]] for char, freq in char_freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        char, code = heap[0][1]
        return {char: "0"}

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        merged = lo + hi
        heapq.heappush(heap, [lo[0] + hi[0]] + merged)

    huffman_codes = sorted(heap[0][1:], key=lambda p: (len(p[-1]), p))
    return {char: code for char, code in huffman_codes}


def kruskal_mst(num_vertices: int, edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """
    Finds the Minimum Spanning Tree (MST) of a graph using Kruskal's algorithm.

    Args:
        num_vertices (int): Number of vertices in the graph.
        edges (List[Tuple[int, int, float]]): A list of tuples representing edges in the form (u, v, weight).

    Returns:
        List[Tuple[int, int, float]]: A list of edges that form the MST.

    Raises:
        GreedyAlgorithmError: If the number of vertices is non-positive or edges list is insufficient to form MST.
    """
    if num_vertices <= 0:
        raise GreedyAlgorithmError("Number of vertices must be positive.")
    if not edges:
        raise GreedyAlgorithmError("Edges list is empty.")

    # Sort edges based on weight
    sorted_edges = sorted(edges, key=lambda x: x[2])
    parent = list(range(num_vertices))

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u: int, v: int) -> None:
        parent[find(u)] = find(v)

    mst = []
    for u, v, weight in sorted_edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, weight))
            if len(mst) == num_vertices - 1:
                break

    if len(mst) != num_vertices - 1:
        raise GreedyAlgorithmError("MST cannot be formed with the given edges.")

    return mst


def prim_mst(num_vertices: int, adjacency_list: Dict[int, List[Tuple[int, float]]]) -> List[Tuple[int, int, float]]:
    """
    Finds the Minimum Spanning Tree (MST) of a graph using Prim's algorithm.

    Args:
        num_vertices (int): Number of vertices in the graph.
        adjacency_list (Dict[int, List[Tuple[int, float]]]): Adjacency list where keys are vertex
                                                             and values are lists of tuples (neighbor, weight).

    Returns:
        List[Tuple[int, int, float]]: A list of edges that form the MST.

    Raises:
        GreedyAlgorithmError: If the number of vertices is non-positive or adjacency list is incomplete.
    """
    if num_vertices <= 0:
        raise GreedyAlgorithmError("Number of vertices must be positive.")
    if not adjacency_list:
        raise GreedyAlgorithmError("Adjacency list is empty.")
    if len(adjacency_list) != num_vertices:
        raise GreedyAlgorithmError("Adjacency list does not match number of vertices.")

    visited = [False] * num_vertices
    min_heap = [(0, 0, -1)]  # (weight, vertex, parent)
    mst = []

    while min_heap and len(mst) < num_vertices:
        weight, u, parent = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        if parent != -1:
            mst.append((parent, u, weight))
        for v, w in adjacency_list.get(u, []):
            if not visited[v]:
                heapq.heappush(min_heap, (w, v, u))

    if len(mst) != num_vertices - 1:
        raise GreedyAlgorithmError("MST cannot be formed with the given adjacency list.")

    return mst


def dijkstra_shortest_path(
    num_vertices: int,
    adjacency_list: Dict[int, List[Tuple[int, float]]],
    start: int
) -> Tuple[List[float], List[int]]:
    """
    Finds the shortest paths from the start vertex to all other vertices using Dijkstra's algorithm.

    Args:
        num_vertices (int): Number of vertices in the graph.
        adjacency_list (Dict[int, List[Tuple[int, float]]]): Adjacency list where keys are vertex
                                                             and values are lists of tuples (neighbor, weight).
        start (int): The starting vertex.

    Returns:
        Tuple[List[float], List[int]]: A tuple containing:
            - List of minimum distances from start to each vertex.
            - List of predecessors for each vertex in the shortest path.

    Raises:
        GreedyAlgorithmError: If the number of vertices is non-positive, the adjacency list is incomplete,
                              or the start vertex is invalid.
    """
    if num_vertices <= 0:
        raise GreedyAlgorithmError("Number of vertices must be positive.")
    if not adjacency_list:
        raise GreedyAlgorithmError("Adjacency list is empty.")
    if len(adjacency_list) != num_vertices:
        raise GreedyAlgorithmError("Adjacency list does not match number of vertices.")
    if start < 0 or start >= num_vertices:
        raise GreedyAlgorithmError("Start vertex is out of bounds.")

    distances = [float('inf')] * num_vertices
    predecessors = [-1] * num_vertices
    distances[start] = 0
    min_heap = [(0, start)]

    while min_heap:
        current_distance, u = heapq.heappop(min_heap)
        if current_distance > distances[u]:
            continue
        for v, weight in adjacency_list.get(u, []):
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
                heapq.heappush(min_heap, (distances[v], v))

    return distances, predecessors


def coin_change_greedy(amount: int, coins: List[int]) -> List[int]:
    """
    Solves the Coin Change problem using the Greedy approach.
    Note: The Greedy approach does not always yield the optimal solution for all coin systems.

    Args:
        amount (int): The total amount to make change for.
        coins (List[int]): A list of coin denominations, sorted in descending order.

    Returns:
        List[int]: A list containing the count of each coin used to make up the amount.

    Raises:
        GreedyAlgorithmError: If amount is negative or coins list is empty.
    """
    if amount < 0:
        raise GreedyAlgorithmError("Amount cannot be negative.")
    if not coins:
        raise GreedyAlgorithmError("Coins list is empty.")

    coins_sorted = sorted(coins, reverse=True)
    result = []
    remaining = amount

    for coin in coins_sorted:
        if coin <= 0:
            continue  # Skip non-positive denominations
        count, remaining = divmod(remaining, coin)
        result.append(count)

    if remaining != 0:
        raise GreedyAlgorithmError("Cannot make the exact amount with the given coins.")

    return result


def main():
    """
    Demonstrates the usage of various greedy algorithms implemented in this module.
    """
    try:
        # Activity Selection Example
        activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 13), (12, 14)]
        selected_activities = activity_selection(activities)
        print("Selected Activities:", selected_activities)

        # Fractional Knapsack Example
        capacity = 50
        items = [(60, 10), (100, 20), (120, 30)]
        max_value = fractional_knapsack(capacity, items)
        print("Maximum value in Knapsack:", max_value)

        # Huffman Encoding Example
        char_freq = {'a': 45, 'b': 13, 'c': 12, 'd': 16, 'e': 9, 'f': 5}
        huffman_codes = huffman_encoding(char_freq)
        print("Huffman Codes:", huffman_codes)

        # Kruskal's MST Example
        num_vertices = 4
        edges = [
            (0, 1, 10),
            (0, 2, 6),
            (0, 3, 5),
            (1, 3, 15),
            (2, 3, 4)
        ]
        mst_kruskal = kruskal_mst(num_vertices, edges)
        print("Kruskal's MST:", mst_kruskal)

        # Prim's MST Example
        adjacency_list = {
            0: [(1, 10), (2, 6), (3, 5)],
            1: [(0, 10), (3, 15)],
            2: [(0, 6), (3, 4)],
            3: [(0, 5), (1, 15), (2, 4)]
        }
        mst_prim = prim_mst(num_vertices, adjacency_list)
        print("Prim's MST:", mst_prim)

        # Dijkstra's Shortest Path Example
        adjacency_list_dijkstra = {
            0: [(1, 4), (2, 1)],
            1: [(3, 1)],
            2: [(1, 2), (3, 5)],
            3: []
        }
        distances, predecessors = dijkstra_shortest_path(4, adjacency_list_dijkstra, 0)
        print("Dijkstra's Shortest Distances:", distances)
        print("Dijkstra's Predecessors:", predecessors)

        # Coin Change Example
        amount = 30
        coins = [25, 10, 5]
        coin_counts = coin_change_greedy(amount, coins)
        print("Coin Counts:", coin_counts)

    except GreedyAlgorithmError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()