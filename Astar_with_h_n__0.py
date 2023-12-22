import heapq
import numpy as np
import time

class State:
    def __init__(self, current_city, visited, path_cost, path):
        self.current_city = current_city
        self.visited = visited
        self.path_cost = path_cost
        self.path = path

    def __lt__(self, other):
        return self.path_cost < other.path_cost

def a_star_tsp(cost_matrix):
    N = len(cost_matrix)
    start_state = State(0, 1 << 0, 0, [0])  # Start at city 0
    frontier = [start_state]  # Priority queue
    expanded_nodes = 0
    generated_nodes = 0

    while frontier:
        current_state = heapq.heappop(frontier)
        expanded_nodes += 1

        # Check if all cities are visited and the current city is 0
        if current_state.visited == (1 << N) - 1 and current_state.current_city == 0:
            return current_state.path, current_state.path_cost, expanded_nodes, generated_nodes

        for next_city in range(N):
            if current_state.current_city != next_city and (not current_state.visited & (1 << next_city) or (next_city == 0 and current_state.visited == (1 << N) - 1)):
                new_visited = current_state.visited | (1 << next_city)
                new_cost = current_state.path_cost + cost_matrix[current_state.current_city][next_city]
                new_path = current_state.path + [next_city]
                heapq.heappush(frontier, State(next_city, new_visited, new_cost, new_path))
                generated_nodes += 1

    return None, float('inf'), expanded_nodes, generated_nodes

def generate_random_cost_matrix(N, seed=None):
    np.random.seed(seed)
    matrix = np.random.randint(1, 101, size=(N, N))
    np.fill_diagonal(matrix, 0)
    return matrix
# Example execution
# N = 5  # Number of cities
# seed = 1
# cost_matrix = generate_random_cost_matrix(N, seed)

# start_time = time.time()  # Start timing
# path, cost, expanded_nodes, generated_nodes = a_star_tsp(cost_matrix)
# end_time = time.time()  # End timing

# print("Path:", path)
# print("Cost:", cost)
# print("Expanded Nodes:", expanded_nodes)
# print("Generated Nodes:", generated_nodes)
# print("Running Time:", end_time - start_time, "seconds")
for N in [10,11,12]:
  print("With N = ",N)
  for seed in range(1,6):
    cost_matrix = generate_random_cost_matrix(N, seed)

    start_time = time.time()  # Start timing
    path, cost, expanded_nodes, generated_nodes = a_star_tsp(cost_matrix)
    end_time = time.time()  # End timing

    print("Path:", path)
    print("Cost:", cost)
    print("Expanded Nodes:", expanded_nodes)
    print("Generated Nodes:", generated_nodes)
    print("Running Time:", end_time - start_time, "seconds")
    print("---------------------------------------------------")
