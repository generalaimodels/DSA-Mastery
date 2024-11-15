"""
graphs.py

A comprehensive module for graph data structures and algorithms, ranging from basic
to advanced implementations. This module adheres to PEP-8 standards, utilizes type
hints for clarity, and incorporates robust error handling. Torch is used for optimized
tensor operations where applicable.
"""

from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Generic,
    Callable,
)
import torch

T = TypeVar('T')


class GraphError(Exception):
    """Custom exception class for Graph-related errors."""
    pass


class Graph(Generic[T]):
    """
    A general graph class supporting both directed and undirected graphs.
    
    Attributes:
        directed (bool): Indicates if the graph is directed.
        adjacency_list (Dict[T, Set[T]]): Adjacency list representing the graph.
    """

    def __init__(self, directed: bool = False) -> None:
        """
        Initializes the Graph.

        Args:
            directed (bool, optional): If True, creates a directed graph. Defaults to False.
        """
        self.directed: bool = directed
        self.adjacency_list: Dict[T, Set[T]] = {}

    def add_vertex(self, vertex: T) -> None:
        """
        Adds a vertex to the graph.

        Args:
            vertex (T): The vertex to add.

        Raises:
            GraphError: If the vertex already exists.
        """
        if vertex in self.adjacency_list:
            raise GraphError(f"Vertex '{vertex}' already exists.")
        self.adjacency_list[vertex] = set()

    def add_edge(self, source: T, destination: T) -> None:
        """
        Adds an edge to the graph.

        Args:
            source (T): The source vertex.
            destination (T): The destination vertex.

        Raises:
            GraphError: If either vertex does not exist.
        """
        if source not in self.adjacency_list:
            raise GraphError(f"Source vertex '{source}' does not exist.")
        if destination not in self.adjacency_list:
            raise GraphError(f"Destination vertex '{destination}' does not exist.")
        self.adjacency_list[source].add(destination)
        if not self.directed:
            self.adjacency_list[destination].add(source)

    def remove_vertex(self, vertex: T) -> None:
        """
        Removes a vertex and all associated edges from the graph.

        Args:
            vertex (T): The vertex to remove.

        Raises:
            GraphError: If the vertex does not exist.
        """
        if vertex not in self.adjacency_list:
            raise GraphError(f"Vertex '{vertex}' does not exist.")
        # Remove all edges to this vertex
        for adj in self.adjacency_list.values():
            adj.discard(vertex)
        del self.adjacency_list[vertex]

    def remove_edge(self, source: T, destination: T) -> None:
        """
        Removes an edge from the graph.

        Args:
            source (T): The source vertex.
            destination (T): The destination vertex.

        Raises:
            GraphError: If the edge does not exist.
        """
        if source not in self.adjacency_list or destination not in self.adjacency_list:
            raise GraphError("One or both vertices do not exist.")
        if destination not in self.adjacency_list[source]:
            raise GraphError(f"Edge from '{source}' to '{destination}' does not exist.")
        self.adjacency_list[source].remove(destination)
        if not self.directed:
            self.adjacency_list[destination].remove(source)

    def get_vertices(self) -> Set[T]:
        """
        Returns all vertices in the graph.

        Returns:
            Set[T]: A set of all vertices.
        """
        return set(self.adjacency_list.keys())

    def get_edges(self) -> Set[Tuple[T, T]]:
        """
        Returns all edges in the graph.

        Returns:
            Set[Tuple[T, T]]: A set of edges represented as tuples.
        """
        edges = set()
        for source, destinations in self.adjacency_list.items():
            for destination in destinations:
                edge = (source, destination)
                if not self.directed:
                    edge = tuple(sorted(edge))
                edges.add(edge)
        return edges

    def __str__(self) -> str:
        """
        Returns a string representation of the graph.

        Returns:
            str: The string representation.
        """
        graph_type = "Directed" if self.directed else "Undirected"
        return f"{graph_type} Graph with vertices: {self.get_vertices()} and edges: {self.get_edges()}"


class GraphAlgorithms:
    """
    A collection of graph algorithms.
    """

    @staticmethod
    def bfs(graph: Graph[Any], start: Any) -> List[Any]:
        """
        Performs Breadth-First Search (BFS) on the graph.

        Args:
            graph (Graph[Any]): The graph to traverse.
            start (Any): The starting vertex.

        Raises:
            GraphError: If the starting vertex does not exist.

        Returns:
            List[Any]: The list of vertices in BFS order.
        """
        if start not in graph.adjacency_list:
            raise GraphError(f"Start vertex '{start}' does not exist.")

        visited: Set[Any] = set()
        queue: List[Any] = [start]
        bfs_order: List[Any] = []

        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                bfs_order.append(vertex)
                queue.extend(graph.adjacency_list[vertex] - visited)
        return bfs_order

    @staticmethod
    def dfs(graph: Graph[Any], start: Any) -> List[Any]:
        """
        Performs Depth-First Search (DFS) on the graph.

        Args:
            graph (Graph[Any]): The graph to traverse.
            start (Any): The starting vertex.

        Raises:
            GraphError: If the starting vertex does not exist.

        Returns:
            List[Any]: The list of vertices in DFS order.
        """
        if start not in graph.adjacency_list:
            raise GraphError(f"Start vertex '{start}' does not exist.")

        visited: Set[Any] = set()
        stack: List[Any] = [start]
        dfs_order: List[Any] = []

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                dfs_order.append(vertex)
                stack.extend(graph.adjacency_list[vertex] - visited)
        return dfs_order

    @staticmethod
    def dijkstra(
        graph: Graph[Any], start: Any, weights: Optional[Dict[Tuple[Any, Any], float]] = None
    ) -> Dict[Any, float]:
        """
        Finds the shortest paths from the start vertex to all other vertices using Dijkstra's algorithm.

        Args:
            graph (Graph[Any]): The graph.
            start (Any): The starting vertex.
            weights (Optional[Dict[Tuple[Any, Any], float]], optional): Edge weights. Defaults to None.

        Raises:
            GraphError: If the start vertex does not exist.

        Returns:
            Dict[Any, float]: A dictionary mapping vertices to their shortest distance from start.
        """
        if start not in graph.adjacency_list:
            raise GraphError(f"Start vertex '{start}' does not exist.")

        distances: Dict[Any, float] = {vertex: float('inf') for vertex in graph.get_vertices()}
        distances[start] = 0.0

        visited: Set[Any] = set()
        tensor_distances = torch.tensor([float('inf') for _ in graph.get_vertices()])
        vertex_list = list(graph.get_vertices())
        start_index = vertex_list.index(start)
        tensor_distances[start_index] = 0.0

        while len(visited) < len(graph.get_vertices()):
            # Select the unvisited vertex with the smallest distance
            min_distance = float('inf')
            current_vertex = None
            for vertex in graph.get_vertices():
                if vertex not in visited and distances[vertex] < min_distance:
                    min_distance = distances[vertex]
                    current_vertex = vertex

            if current_vertex is None:
                break

            visited.add(current_vertex)
            for neighbor in graph.adjacency_list[current_vertex]:
                edge = (current_vertex, neighbor)
                weight = weights[edge] if weights and edge in weights else 1.0
                new_distance = distances[current_vertex] + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    neighbor_index = vertex_list.index(neighbor)
                    tensor_distances[neighbor_index] = new_distance

        return distances

    @staticmethod
    def a_star(
        graph: Graph[Any],
        start: Any,
        goal: Any,
        heuristic: Callable[[Any, Any], float],
        weights: Optional[Dict[Tuple[Any, Any], float]] = None,
    ) -> List[Any]:
        """
        Finds the shortest path from start to goal using the A* algorithm.

        Args:
            graph (Graph[Any]): The graph.
            start (Any): The starting vertex.
            goal (Any): The goal vertex.
            heuristic (Callable[[Any, Any], float]): Heuristic function estimating distance from current to goal.
            weights (Optional[Dict[Tuple[Any, Any], float]], optional): Edge weights. Defaults to None.

        Raises:
            GraphError: If start or goal vertices do not exist.

        Returns:
            List[Any]: The list of vertices representing the shortest path.
        """
        if start not in graph.adjacency_list:
            raise GraphError(f"Start vertex '{start}' does not exist.")
        if goal not in graph.adjacency_list:
            raise GraphError(f"Goal vertex '{goal}' does not exist.")

        open_set: Set[Any] = {start}
        came_from: Dict[Any, Any] = {}

        g_score: Dict[Any, float] = {vertex: float('inf') for vertex in graph.get_vertices()}
        g_score[start] = 0.0

        f_score: Dict[Any, float] = {vertex: float('inf') for vertex in graph.get_vertices()}
        f_score[start] = heuristic(start, goal)

        while open_set:
            current = min(open_set, key=lambda vertex: f_score[vertex])
            if current == goal:
                return GraphAlgorithms.reconstruct_path(came_from, current)

            open_set.remove(current)
            for neighbor in graph.adjacency_list[current]:
                tentative_g_score = g_score[current]
                edge = (current, neighbor)
                weight = weights[edge] if weights and edge in weights else 1.0
                tentative_g_score += weight

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.add(neighbor)

        raise GraphError("No path found from start to goal.")

    @staticmethod
    def reconstruct_path(came_from: Dict[Any, Any], current: Any) -> List[Any]:
        """
        Reconstructs the path from start to current using the came_from dictionary.

        Args:
            came_from (Dict[Any, Any]): The mapping of navigated vertices.
            current (Any): The current vertex.

        Returns:
            List[Any]: The reconstructed path.
        """
        path: List[Any] = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def detect_cycle(graph: Graph[Any]) -> bool:
        """
        Detects if there is a cycle in the graph using DFS.

        Args:
            graph (Graph[Any]): The graph to check.

        Returns:
            bool: True if a cycle is detected, False otherwise.
        """
        visited: Set[Any] = set()
        recursion_stack: Set[Any] = set()

        def dfs(vertex: Any) -> bool:
            visited.add(vertex)
            recursion_stack.add(vertex)
            for neighbor in graph.adjacency_list[vertex]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            recursion_stack.remove(vertex)
            return False

        for vertex in graph.get_vertices():
            if vertex not in visited:
                if dfs(vertex):
                    return True
        return False

    @staticmethod
    def topological_sort(graph: Graph[Any]) -> List[Any]:
        """
        Performs Topological Sort on a Directed Acyclic Graph (DAG).

        Args:
            graph (Graph[Any]): The graph to sort.

        Raises:
            GraphError: If the graph is not directed or contains a cycle.

        Returns:
            List[Any]: A list of vertices in topologically sorted order.
        """
        if not graph.directed:
            raise GraphError("Topological sort is only applicable to directed graphs.")

        if GraphAlgorithms.detect_cycle(graph):
            raise GraphError("Graph contains a cycle; topological sort not possible.")

        in_degree: Dict[Any, int] = {vertex: 0 for vertex in graph.get_vertices()}
        for source in graph.get_vertices():
            for destination in graph.adjacency_list[source]:
                in_degree[destination] += 1

        queue: List[Any] = [vertex for vertex in graph.get_vertices() if in_degree[vertex] == 0]
        topo_order: List[Any] = []

        while queue:
            vertex = queue.pop(0)
            topo_order.append(vertex)
            for neighbor in graph.adjacency_list[vertex]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo_order) != len(graph.get_vertices()):
            raise GraphError("Graph contains a cycle; topological sort not possible.")

        return topo_order

    @staticmethod
    def bellman_ford(
        graph: Graph[Any], start: Any, weights: Optional[Dict[Tuple[Any, Any], float]] = None
    ) -> Dict[Any, float]:
        """
        Finds the shortest paths from start to all other vertices using Bellman-Ford algorithm.

        Args:
            graph (Graph[Any]): The graph.
            start (Any): The starting vertex.
            weights (Optional[Dict[Tuple[Any, Any], float]], optional): Edge weights. Defaults to None.

        Raises:
            GraphError: If a negative-weight cycle is detected or start vertex does not exist.

        Returns:
            Dict[Any, float]: A dictionary mapping vertices to their shortest distance from start.
        """
        if start not in graph.adjacency_list:
            raise GraphError(f"Start vertex '{start}' does not exist.")

        distances: Dict[Any, float] = {vertex: float('inf') for vertex in graph.get_vertices()}
        distances[start] = 0.0

        edges: List[Tuple[Any, Any, float]] = []
        for source in graph.get_vertices():
            for destination in graph.adjacency_list[source]:
                weight = weights[(source, destination)] if weights and (source, destination) in weights else 1.0
                edges.append((source, destination, weight))

        for _ in range(len(graph.get_vertices()) - 1):
            for source, destination, weight in edges:
                if distances[source] + weight < distances[destination]:
                    distances[destination] = distances[source] + weight

        # Check for negative-weight cycles
        for source, destination, weight in edges:
            if distances[source] + weight < distances[destination]:
                raise GraphError("Graph contains a negative-weight cycle.")

        return distances

    @staticmethod
    def floyd_warshall(graph: Graph[Any]) -> Dict[Any, Dict[Any, float]]:
        """
        Computes the shortest paths between all pairs of vertices using Floyd-Warshall algorithm.

        Args:
            graph (Graph[Any]): The graph.

        Raises:
            GraphError: If the graph is empty.

        Returns:
            Dict[Any, Dict[Any, float]]: A nested dictionary with shortest distances.
        """
        vertices = list(graph.get_vertices())
        if not vertices:
            raise GraphError("Graph is empty.")

        distance: Dict[Any, Dict[Any, float]] = {
            vertex: {v: float('inf') for v in vertices} for vertex in vertices
        }

        for vertex in vertices:
            distance[vertex][vertex] = 0.0

        for source in vertices:
            for destination in graph.adjacency_list[source]:
                distance[source][destination] = 1.0  # Default weight
                # Modify if weights are provided

        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if distance[i][k] + distance[k][j] < distance[i][j]:
                        distance[i][j] = distance[i][k] + distance[k][j]

        return distance

    @staticmethod
    def krusky_mst(graph: Graph[Any]) -> Set[Tuple[Any, Any]]:
        """
        Finds the Minimum Spanning Tree (MST) using Krusky's algorithm.

        Args:
            graph (Graph[Any]): The graph.

        Raises:
            GraphError: If the graph is not undirected.

        Returns:
            Set[Tuple[Any, Any]]: A set of edges representing the MST.
        """
        if graph.directed:
            raise GraphError("Minimum Spanning Tree is only defined for undirected graphs.")

        parent: Dict[Any, Any] = {vertex: vertex for vertex in graph.get_vertices()}

        def find(vertex: Any) -> Any:
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]  # Path compression
                vertex = parent[vertex]
            return vertex

        def union(v1: Any, v2: Any) -> None:
            root1 = find(v1)
            root2 = find(v2)
            if root1 != root2:
                parent[root2] = root1

        edges = list(graph.get_edges())
        # Sorting edges by weight is required; assuming all weights are 1 if not provided
        edges.sort(key=lambda edge: 1.0)  # Modify if weights are provided

        mst: Set[Tuple[Any, Any]] = set()
        for edge in edges:
            v1, v2 = edge
            if find(v1) != find(v2):
                union(v1, v2)
                mst.add(edge)

        return mst

    @staticmethod
    def strong_connected_components(graph: Graph[Any]) -> List[Set[Any]]:
        """
        Finds all strongly connected components using Kosaraju's algorithm.

        Args:
            graph (Graph[Any]): The graph.

        Raises:
            GraphError: If the graph is undirected.

        Returns:
            List[Set[Any]]: A list of sets, each representing a strongly connected component.
        """
        if not graph.directed:
            raise GraphError("Strongly connected components are only defined for directed graphs.")

        visited: Set[Any] = set()
        stack: List[Any] = []

        def fill_order(v: Any) -> None:
            visited.add(v)
            for neighbor in graph.adjacency_list[v]:
                if neighbor not in visited:
                    fill_order(neighbor)
            stack.append(v)

        for vertex in graph.get_vertices():
            if vertex not in visited:
                fill_order(vertex)

        transposed = Graph[Any](directed=True)
        for vertex in graph.get_vertices():
            transposed.add_vertex(vertex)
        for source in graph.get_vertices():
            for destination in graph.adjacency_list[source]:
                transposed.add_edge(destination, source)

        visited.clear()
        scc: List[Set[Any]] = []

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                component: Set[Any] = set()
                GraphAlgorithms.dfs_util(transposed, vertex, visited, component)
                scc.append(component)

        return scc

    @staticmethod
    def dfs_util(
        graph: Graph[Any], vertex: Any, visited: Set[Any], component: Set[Any]
    ) -> None:
        """
        Utility function for DFS traversal.

        Args:
            graph (Graph[Any]): The graph.
            vertex (Any): The current vertex.
            visited (Set[Any]): The set of visited vertices.
            component (Set[Any]): The current component being built.
        """
        visited.add(vertex)
        component.add(vertex)
        for neighbor in graph.adjacency_list[vertex]:
            if neighbor not in visited:
                GraphAlgorithms.dfs_util(graph, neighbor, visited, component)

    @staticmethod
    def all_pairs_shortest_path_dijkstra(
        graph: Graph[Any], weights: Optional[Dict[Tuple[Any, Any], float]] = None
    ) -> Dict[Any, Dict[Any, float]]:
        """
        Computes all pairs shortest paths using Dijkstra's algorithm.

        Args:
            graph (Graph[Any]): The graph.
            weights (Optional[Dict[Tuple[Any, Any], float]], optional): Edge weights. Defaults to None.

        Returns:
            Dict[Any, Dict[Any, float]]: A nested dictionary with shortest distances.
        """
        all_distances: Dict[Any, Dict[Any, float]] = {}
        for vertex in graph.get_vertices():
            all_distances[vertex] = GraphAlgorithms.dijkstra(graph, vertex, weights)
        return all_distances

    @staticmethod
    def maximum_flow(
        graph: Graph[Any],
        source: Any,
        sink: Any,
        capacities: Dict[Tuple[Any, Any], float]
    ) -> float:
        """
        Computes the maximum flow from source to sink using the Ford-Fulkerson method.

        Args:
            graph (Graph[Any]): The graph.
            source (Any): The source vertex.
            sink (Any): The sink vertex.
            capacities (Dict[Tuple[Any, Any], float]): Capacities of edges.

        Raises:
            GraphError: If source or sink vertices do not exist.

        Returns:
            float: The value of the maximum flow.
        """
        if source not in graph.adjacency_list:
            raise GraphError(f"Source vertex '{source}' does not exist.")
        if sink not in graph.adjacency_list:
            raise GraphError(f"Sink vertex '{sink}' does not exist.")

        residual = {edge: capacity for edge, capacity in capacities.items()}
        flow = 0.0

        while True:
            parent = {}
            queue = [source]
            while queue:
                u = queue.pop(0)
                for v in graph.adjacency_list[u]:
                    if (u, v) in residual and residual[(u, v)] > 0 and v not in parent:
                        parent[v] = u
                        queue.append(v)
                        if v == sink:
                            break
            if sink not in parent:
                break

            # Find minimum residual capacity along the path
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, residual[(parent[s], s)])
                s = parent[s]

            # Update residual capacities
            v = sink
            while v != source:
                u = parent[v]
                residual[(u, v)] -= path_flow
                residual.setdefault((v, u), 0.0)
                residual[(v, u)] += path_flow
                v = u

            flow += path_flow

        return flow

    @staticmethod
    def bipartite_matching(
        graph: Graph[Any],
        left_partition: Set[Any],
        right_partition: Set[Any],
        capacities: Optional[Dict[Tuple[Any, Any], float]] = None
    ) -> Set[Tuple[Any, Any]]:
        """
        Finds the maximum bipartite matching using the Hopcroft-Karp algorithm.

        Args:
            graph (Graph[Any]): The bipartite graph.
            left_partition (Set[Any]): The left partition of vertices.
            right_partition (Set[Any]): The right partition of vertices.
            capacities (Optional[Dict[Tuple[Any, Any], float]], optional): Edge capacities. Defaults to None.

        Raises:
            GraphError: If the graph is not bipartite as per the partitions.

        Returns:
            Set[Tuple[Any, Any]]: A set of matched pairs.
        """
        # Implementation placeholder: Hopcroft-Karp for maximum matching
        # Detailed implementation would follow similar robustness and optimization
        raise NotImplementedError("Hopcroft-Karp algorithm is not implemented yet.")


# Example Usage:
if __name__ == "__main__":
    try:
        # Create an undirected graph
        graph = Graph[str](directed=False)
        for vertex in ['A', 'B', 'C', 'D', 'E']:
            graph.add_vertex(vertex)
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'C')
        graph.add_edge('B', 'D')
        graph.add_edge('C', 'D')
        graph.add_edge('D', 'E')

        print(graph)

        # Perform BFS
        bfs_order = GraphAlgorithms.bfs(graph, 'A')
        print(f"BFS Order: {bfs_order}")

        # Perform DFS
        dfs_order = GraphAlgorithms.dfs(graph, 'A')
        print(f"DFS Order: {dfs_order}")

        # Detect Cycle
        has_cycle = GraphAlgorithms.detect_cycle(graph)
        print(f"Graph has cycle: {has_cycle}")

        # Topological Sort (will raise error as the graph is undirected)
        try:
            topo_order = GraphAlgorithms.topological_sort(graph)
            print(f"Topological Order: {topo_order}")
        except GraphError as e:
            print(e)

        # Dijkstra's Algorithm
        weights = {('A', 'B'): 2, ('A', 'C'): 5, ('B', 'C'): 1, ('B', 'D'): 2, ('C', 'D'): 3, ('D', 'E'): 1}
        distances = GraphAlgorithms.dijkstra(graph, 'A', weights)
        print(f"Dijkstra's shortest distances from A: {distances}")

    except GraphError as ge:
        print(f"Graph Error: {ge}")
    except Exception as ex:
        print(f"An error occurred: {ex}")