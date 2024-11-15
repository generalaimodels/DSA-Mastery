"""
Graphs Module
=============
This module provides comprehensive implementations of graph data structures and algorithms,
ranging from basic to advanced levels. It adheres to PEP-8 standards, utilizes type hints
for clarity, and includes comprehensive error handling to ensure robustness and scalability.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import heapq


class GraphError(Exception):
    """Custom exception class for Graph-related errors."""
    pass


class Graph:
    """
    Represents a directed or undirected graph using an adjacency list.
    
    Attributes:
        is_directed (bool): Indicates if the graph is directed.
        adjacency_list (Dict[Any, List[Tuple[Any, float]]]): Adjacency list representation.
    """

    def __init__(self, is_directed: bool = False) -> None:
        """Initializes the Graph."""
        self.is_directed = is_directed
        self.adjacency_list: Dict[Any, List[Tuple[Any, float]]] = {}

    def add_vertex(self, vertex: Any) -> None:
        """
        Adds a vertex to the graph.
        
        Args:
            vertex (Any): The vertex to add.
        
        Raises:
            GraphError: If the vertex already exists.
        """
        if vertex in self.adjacency_list:
            raise GraphError(f"Vertex '{vertex}' already exists.")
        self.adjacency_list[vertex] = []

    def add_edge(self, source: Any, destination: Any, weight: float = 1.0) -> None:
        """
        Adds an edge to the graph.
        
        Args:
            source (Any): The source vertex.
            destination (Any): The destination vertex.
            weight (float, optional): The weight of the edge. Defaults to 1.0.
        
        Raises:
            GraphError: If either vertex does not exist.
        """
        if source not in self.adjacency_list or destination not in self.adjacency_list:
            raise GraphError("Both vertices must exist in the graph.")
        self.adjacency_list[source].append((destination, weight))
        if not self.is_directed:
            self.adjacency_list[destination].append((source, weight))

    def remove_vertex(self, vertex: Any) -> None:
        """
        Removes a vertex and all associated edges from the graph.
        
        Args:
            vertex (Any): The vertex to remove.
        
        Raises:
            GraphError: If the vertex does not exist.
        """
        if vertex not in self.adjacency_list:
            raise GraphError(f"Vertex '{vertex}' does not exist.")
        del self.adjacency_list[vertex]
        for edges in self.adjacency_list.values():
            edges[:] = [edge for edge in edges if edge[0] != vertex]

    def remove_edge(self, source: Any, destination: Any) -> None:
        """
        Removes an edge from the graph.
        
        Args:
            source (Any): The source vertex.
            destination (Any): The destination vertex.
        
        Raises:
            GraphError: If the edge does not exist.
        """
        if source not in self.adjacency_list:
            raise GraphError(f"Source vertex '{source}' does not exist.")
        if not any(edge[0] == destination for edge in self.adjacency_list[source]):
            raise GraphError(f"Edge from '{source}' to '{destination}' does not exist.")
        self.adjacency_list[source] = [
            edge for edge in self.adjacency_list[source] if edge[0] != destination
        ]
        if not self.is_directed:
            self.adjacency_list[destination] = [
                edge for edge in self.adjacency_list[destination] if edge[0] != source
            ]

    def get_vertices(self) -> List[Any]:
        """Returns a list of all vertices in the graph."""
        return list(self.adjacency_list.keys())

    def get_edges(self) -> List[Tuple[Any, Any, float]]:
        """Returns a list of all edges in the graph."""
        edges = []
        for source, destinations in self.adjacency_list.items():
            for destination, weight in destinations:
                if not self.is_directed and source < destination:
                    edges.append((source, destination, weight))
                elif self.is_directed:
                    edges.append((source, destination, weight))
        return edges

    def __str__(self) -> str:
        """Returns a string representation of the graph."""
        graph_str = ""
        for vertex, edges in self.adjacency_list.items():
            edges_str = ", ".join([f"{dest}(weight={w})" for dest, w in edges])
            graph_str += f"{vertex} -> {edges_str}\n"
        return graph_str.strip()


def bfs(graph: Graph, start: Any) -> List[Any]:
    """
    Performs Breadth-First Search (BFS) on the graph.
    
    Args:
        graph (Graph): The graph to traverse.
        start (Any): The starting vertex.
    
    Returns:
        List[Any]: The list of vertices in BFS order.
    
    Raises:
        GraphError: If the start vertex does not exist.
    """
    if start not in graph.adjacency_list:
        raise GraphError(f"Start vertex '{start}' does not exist.")

    visited: Set[Any] = set()
    queue: List[Any] = [start]
    order: List[Any] = []

    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            queue.extend([neighbor for neighbor, _ in graph.adjacency_list[vertex] if neighbor not in visited])
    return order


def dfs(graph: Graph, start: Any) -> List[Any]:
    """
    Performs Depth-First Search (DFS) on the graph.
    
    Args:
        graph (Graph): The graph to traverse.
        start (Any): The starting vertex.
    
    Returns:
        List[Any]: The list of vertices in DFS order.
    
    Raises:
        GraphError: If the start vertex does not exist.
    """
    if start not in graph.adjacency_list:
        raise GraphError(f"Start vertex '{start}' does not exist.")

    visited: Set[Any] = set()
    stack: List[Any] = [start]
    order: List[Any] = []

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            stack.extend([neighbor for neighbor, _ in reversed(graph.adjacency_list[vertex]) if neighbor not in visited])
    return order


def dijkstra(graph: Graph, start: Any) -> Dict[Any, float]:
    """
    Finds the shortest paths from the start vertex to all other vertices using Dijkstra's algorithm.
    
    Args:
        graph (Graph): The graph to traverse.
        start (Any): The starting vertex.
    
    Returns:
        Dict[Any, float]: A dictionary mapping vertices to their shortest distance from start.
    
    Raises:
        GraphError: If the start vertex does not exist.
    """
    if start not in graph.adjacency_list:
        raise GraphError(f"Start vertex '{start}' does not exist.")

    distances: Dict[Any, float] = {vertex: float('inf') for vertex in graph.adjacency_list}
    distances[start] = 0.0
    priority_queue: List[Tuple[float, Any]] = [(0.0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph.adjacency_list[current_vertex]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


def bellman_ford(graph: Graph, start: Any) -> Dict[Any, float]:
    """
    Finds the shortest paths from the start vertex to all other vertices using Bellman-Ford algorithm.
    
    Args:
        graph (Graph): The graph to traverse.
        start (Any): The starting vertex.
    
    Returns:
        Dict[Any, float]: A dictionary mapping vertices to their shortest distance from start.
    
    Raises:
        GraphError: If the start vertex does not exist or a negative cycle is detected.
    """
    if start not in graph.adjacency_list:
        raise GraphError(f"Start vertex '{start}' does not exist.")

    distances: Dict[Any, float] = {vertex: float('inf') for vertex in graph.adjacency_list}
    distances[start] = 0.0

    vertices = graph.get_vertices()
    for _ in range(len(vertices) - 1):
        for u, v, w in graph.get_edges():
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w

    # Check for negative-weight cycles
    for u, v, w in graph.get_edges():
        if distances[u] + w < distances[v]:
            raise GraphError("Graph contains a negative-weight cycle.")

    return distances


def floyd_warshall(graph: Graph) -> Dict[Any, Dict[Any, float]]:
    """
    Computes the shortest paths between all pairs of vertices using Floyd-Warshall algorithm.
    
    Args:
        graph (Graph): The graph to analyze.
    
    Returns:
        Dict[Any, Dict[Any, float]]: A nested dictionary mapping vertex pairs to their shortest distance.
    
    Raises:
        GraphError: If the graph contains a negative-weight cycle.
    """
    vertices = graph.get_vertices()
    distances: Dict[Any, Dict[Any, float]] = {u: {v: float('inf') for v in vertices} for u in vertices}

    for u in vertices:
        distances[u][u] = 0.0
        for v, w in graph.adjacency_list[u]:
            distances[u][v] = min(distances[u][v], w)

    for k in vertices:
        for i in vertices:
            for j in vertices:
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]

    # Check for negative-weight cycles
    for u in vertices:
        if distances[u][u] < 0:
            raise GraphError("Graph contains a negative-weight cycle.")

    return distances


def kruskal_mst(graph: Graph) -> List[Tuple[Any, Any, float]]:
    """
    Finds the Minimum Spanning Tree (MST) of an undirected graph using Kruskal's algorithm.
    
    Args:
        graph (Graph): The graph to analyze.
    
    Returns:
        List[Tuple[Any, Any, float]]: A list of edges in the MST.
    
    Raises:
        GraphError: If the graph is directed.
    """
    if graph.is_directed:
        raise GraphError("Kruskal's algorithm requires an undirected graph.")

    parent: Dict[Any, Any] = {}
    rank: Dict[Any, int] = {}

    def find(u: Any) -> Any:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u: Any, v: Any) -> None:
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return
        if rank[root_u] < rank[root_v]:
            parent[root_u] = root_v
        else:
            parent[root_v] = root_u
            if rank[root_u] == rank[root_v]:
                rank[root_u] += 1

    for vertex in graph.get_vertices():
        parent[vertex] = vertex
        rank[vertex] = 0

    sorted_edges = sorted(graph.get_edges(), key=lambda edge: edge[2])
    mst: List[Tuple[Any, Any, float]] = []

    for u, v, w in sorted_edges:
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, w))

    return mst


def prim_mst(graph: Graph, start: Any) -> List[Tuple[Any, Any, float]]:
    """
    Finds the Minimum Spanning Tree (MST) of a connected undirected graph using Prim's algorithm.
    
    Args:
        graph (Graph): The graph to analyze.
        start (Any): The starting vertex.
    
    Returns:
        List[Tuple[Any, Any, float]]: A list of edges in the MST.
    
    Raises:
        GraphError: If the graph is directed or the start vertex does not exist.
    """
    if graph.is_directed:
        raise GraphError("Prim's algorithm requires an undirected graph.")
    if start not in graph.adjacency_list:
        raise GraphError(f"Start vertex '{start}' does not exist.")

    visited: Set[Any] = set()
    mst: List[Tuple[Any, Any, float]] = []
    min_heap: List[Tuple[float, Any, Any]] = []

    def add_edges(vertex: Any) -> None:
        visited.add(vertex)
        for neighbor, weight in graph.adjacency_list[vertex]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (weight, vertex, neighbor))

    add_edges(start)

    while min_heap and len(visited) < len(graph.adjacency_list):
        weight, u, v = heapq.heappop(min_heap)
        if v not in visited:
            mst.append((u, v, weight))
            add_edges(v)

    if len(visited) != len(graph.adjacency_list):
        raise GraphError("Graph is not connected.")

    return mst


def topological_sort(graph: Graph) -> List[Any]:
    """
    Performs a topological sort on a directed acyclic graph (DAG).
    
    Args:
        graph (Graph): The graph to sort.
    
    Returns:
        List[Any]: The vertices in topologically sorted order.
    
    Raises:
        GraphError: If the graph is not directed or contains a cycle.
    """
    if not graph.is_directed:
        raise GraphError("Topological sort requires a directed graph.")

    in_degree: Dict[Any, int] = {u: 0 for u in graph.get_vertices()}
    for edges in graph.adjacency_list.values():
        for v, _ in edges:
            in_degree[v] += 1

    queue: List[Any] = [u for u, degree in in_degree.items() if degree == 0]
    sorted_order: List[Any] = []

    while queue:
        u = queue.pop(0)
        sorted_order.append(u)
        for v, _ in graph.adjacency_list[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(sorted_order) != len(graph.adjacency_list):
        raise GraphError("Graph contains a cycle; topological sort not possible.")

    return sorted_order


def strongly_connected_components(graph: Graph) -> List[Set[Any]]:
    """
    Finds all strongly connected components in a directed graph using Kosaraju's algorithm.
    
    Args:
        graph (Graph): The graph to analyze.
    
    Returns:
        List[Set[Any]]: A list of sets, each containing vertices of a strongly connected component.
    
    Raises:
        GraphError: If the graph is undirected.
    """
    if not graph.is_directed:
        raise GraphError("Strongly connected components require a directed graph.")

    visited: Set[Any] = set()
    stack: List[Any] = []

    def fill_order(v: Any) -> None:
        visited.add(v)
        for neighbor, _ in graph.adjacency_list[v]:
            if neighbor not in visited:
                fill_order(neighbor)
        stack.append(v)

    for vertex in graph.get_vertices():
        if vertex not in visited:
            fill_order(vertex)

    transpose = Graph(is_directed=True)
    for vertex in graph.get_vertices():
        transpose.add_vertex(vertex)
    for u, v, w in graph.get_edges():
        transpose.add_edge(v, u, w)

    visited.clear()
    scc: List[Set[Any]] = []

    def dfs_transpose(v: Any, component: Set[Any]) -> None:
        visited.add(v)
        component.add(v)
        for neighbor, _ in transpose.adjacency_list[v]:
            if neighbor not in visited:
                dfs_transpose(neighbor, component)

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            component: Set[Any] = set()
            dfs_transpose(vertex, component)
            scc.append(component)

    return scc