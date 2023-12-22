import heapq
import random
import time
import numpy as np
class State:
    def __init__(self, N, path=None, current_cost=0, visited=None):
        if path is None:
            self.path = [0]  # Start at city 0
        else:
            self.path = path
        self.current_cost = current_cost
        self.N = N
        if visited is None:
            self.visited = [False] * N
            self.visited[0] = True
        else:
            self.visited = visited

    def is_goal(self):
        return len(self.path) == self.N + 1 and self.path[-1] == 0

    def successors(self, cost_matrix):
        succs = []
        last_city = self.path[-1]
        if len(self.path) == self.N:
            if cost_matrix[last_city][0] > 0:
                succs.append(State(self.N, self.path + [0], self.current_cost + cost_matrix[last_city][0], self.visited.copy()))
        else:
            for next_city in range(self.N):
                if not self.visited[next_city] and cost_matrix[last_city][next_city] > 0:
                    new_visited = self.visited.copy()
                    new_visited[next_city] = True
                    new_path = self.path + [next_city]
                    new_cost = self.current_cost + cost_matrix[last_city][next_city]
                    succs.append(State(self.N, new_path, new_cost, new_visited))
        return succs

def min_out_heuristic(current_id, cost_matrix, visited):
    min_cost = float('inf')
    for city in range(len(cost_matrix)):
        if city != current_id and not visited[city]:
            min_cost = min(min_cost, cost_matrix[current_id][city])
    return min_cost if min_cost != float('inf') else 0

def a_star(cost_matrix):
    N = len(cost_matrix)
    root = State(N)
    open_list = []
    counter = 0  # Counter for tie-breaking in the heap
    heapq.heappush(open_list, (min_out_heuristic(0, cost_matrix, root.visited), counter, root))
    counter += 1
    generated_nodes = 0
    expanded_nodes = 0

    while open_list:
        _, _, current_state = heapq.heappop(open_list)
        expanded_nodes += 1
        if current_state.is_goal():
            return current_state, expanded_nodes, generated_nodes
        for succ in current_state.successors(cost_matrix):
            generated_nodes += 1
            f = succ.current_cost + min_out_heuristic(succ.path[-1], cost_matrix, succ.visited)
            heapq.heappush(open_list, (f, counter, succ))
            counter += 1

    return None, expanded_nodes, generated_nodes

def generate_random_cost_matrix(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    matrix = np.random.randint(1, 101, size=(N, N))
    np.fill_diagonal(matrix, 0)  # Zero out diagonal
    return matrix


# Example usage
# N = 5  # Number of cities
# seed = 2
for N in [5,10,11,12]:
  print("with N = ",N)
  for seed in range(1,6):
    print("with seed = ",seed)
    cost_matrix = generate_random_cost_matrix(N,seed)

    start_time = time.time()
    solution, expanded_nodes, generated_nodes = a_star(cost_matrix)
    end_time = time.time()

    if solution:
        print("Path:", solution.path)
        print("Cost:", solution.current_cost)
    else:
        print("No solution found")

    print("Number of Expanded Nodes:", expanded_nodes)
    print("Number of Generated Nodes:", generated_nodes)
    print("Time taken:", end_time - start_time, "seconds")
    print("-----------------------------------------------")
