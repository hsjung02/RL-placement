import numpy as np
import heapq

def routing(cell_positions, adj_i, adj_j) -> int:


    capacity = 1000
    grid_size = 32
    penalty = 50

    graph = {}
    for i in range(grid_size):
        for j in range(grid_size):
            neighbors = {}
            if i>0:
                neighbors[grid_size*(i-1)+j] = capacity
            if i<grid_size-1:
                neighbors[grid_size*(i+1)+j] = capacity
            if j>0:
                neighbors[grid_size*i+j-1] = capacity
            if j<grid_size-1:
                neighbors[grid_size*i+j+1] = capacity
            graph[grid_size*i+j] = neighbors

    edge_num = len(adj_i)
    wirelength = 0
    for edge in range(edge_num):

        start = cell_positions[adj_i[edge],0]*grid_size+cell_positions[adj_i[edge],1]
        end = cell_positions[adj_j[edge],0]*grid_size+cell_positions[adj_j[edge],1]
        if adj_j[edge] <= adj_i[edge]:
            continue

        distances = dijkstra(graph, start, end, penalty)
        graph = track_path(graph, distances, end, penalty)
        wirelength += distances[end]
    
    return wirelength


def dijkstra(graph, start, end, penalty):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = []
    heapq.heappush(queue, [distances[start], start])

    while queue:
        current_distance, current_destination = heapq.heappop(queue)
        if current_destination == end:
            return distances

        if distances[current_destination] < current_distance:
            continue

        for new_destination, capacity in graph[current_destination].items():
            distance = (current_distance + 1 if capacity > 0 else current_distance + penalty)
            if distance < distances[new_destination]:
                distances[new_destination] = distance
                heapq.heappush(queue, [distance, new_destination])

    return distances

def track_path(graph, distances, end, penalty):
    path = []
    now = end
    d = distances[end]
    while d>0:
        for neighbor, capacity in graph[now].items():
            dis = (1 if capacity > 0 else penalty)
            if d == distances[neighbor] + dis:
                path.append([now, neighbor])
                now = neighbor
                d -= dis
                break
    for [x1, x2] in path:
        graph[x1][x2] -= 1
        graph[x2][x1] -= 1
    return graph