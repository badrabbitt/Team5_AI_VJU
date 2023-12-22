import numpy as np
import itertools
import time

class State:
    def __init__(self, N, path=None, current_cost=0):
        if path is None:
            self.path = [0]  # Start at city 0
        else:
            self.path = path
        self.current_cost = current_cost
        self.N = N

    def is_goal(self):
        # Goal is reached when all cities are visited and return to city 0
        return len(self.path) == self.N + 1 and self.path[-1] == 0

    def successors(self, cost_matrix):
        succs = []
        last_city = self.path[-1]
        if len(self.path) == self.N:
            # If all cities have been visited, return to start
            if cost_matrix[last_city][0] > 0:  # Ensure there is a path back to start
                succs.append(State(self.N, self.path + [0], self.current_cost + cost_matrix[last_city][0]))
        else:
            for next_city in range(self.N):
                if next_city not in self.path and cost_matrix[last_city][next_city] > 0:
                    new_path = self.path + [next_city]
                    new_cost = self.current_cost + cost_matrix[last_city][next_city]
                    succs.append(State(self.N, new_path, new_cost))
        return succs

# def ida_star(root, cost_matrix):
#     threshold = root.current_cost
#     while True:
#         distance, state = search(root, 0, threshold, cost_matrix)
#         if distance == 'FOUND':
#             return state
#         if distance == float('inf'):
#             return None
#         threshold = distance

def ida_star(root, cost_matrix):
    threshold = root.current_cost
    expanded_nodes = 0
    generated_nodes = 0
    while True:
        distance, state, expanded, generated = search(root, 0, threshold, cost_matrix, 0, 0)
        expanded_nodes += expanded
        generated_nodes += generated
        if distance == 'FOUND':
            return state, expanded_nodes, generated_nodes
        if distance == float('inf'):
            return None, expanded_nodes, generated_nodes
        threshold = distance

# def search(node, g, threshold, cost_matrix):
#     f = g + node.current_cost
#     if f > threshold:
#         return f, None
#     if node.is_goal():
#         return 'FOUND', node
#     min_threshold = float('inf')
#     for succ in node.successors(cost_matrix):
#         temp, state = search(succ, g + cost_matrix[node.path[-1]][succ.path[-1]], threshold, cost_matrix)
#         if temp == 'FOUND':
#             return 'FOUND', state
#         if temp < min_threshold:
#             min_threshold = temp
#     return min_threshold, None

def search(node, g, threshold, cost_matrix, expanded_nodes, generated_nodes):
    f = g + node.current_cost
    if f > threshold:
        return f, None, expanded_nodes, generated_nodes
    if node.is_goal():
        return 'FOUND', node, expanded_nodes, generated_nodes
    expanded_nodes += 1
    min_threshold = float('inf')
    for succ in node.successors(cost_matrix):
        generated_nodes += 1
        temp, state, exp, gen = search(succ, g + cost_matrix[node.path[-1]][succ.path[-1]], threshold, cost_matrix, expanded_nodes, generated_nodes)
        expanded_nodes = exp
        generated_nodes = gen
        if temp == 'FOUND':
            return 'FOUND', state, expanded_nodes, generated_nodes
        if temp < min_threshold:
            min_threshold = temp
    return min_threshold, None, expanded_nodes, generated_nodes


def generate_random_cost_matrix(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(1, 101, size=(N, N))

# Example Execution
# N = int(input())  # Number of cities
# seed = int(input())  # Seed for reproducibility
N = 12
# seed = 1
for seed in range(2,6):
    cost_matrix = generate_random_cost_matrix(N, seed)
    cost_matrix[np.arange(N), np.arange(N)] = 0  # Zero out diagonal

    start_time = time.time()
    initial_state = State(N)
    # solution = ida_star(initial_state, cost_matrix)
    solution, expanded_nodes, generated_nodes = ida_star(initial_state, cost_matrix)
    end_time = time.time()

    if solution:
        print("Path:", solution.path)
        print("Cost:", solution.current_cost)
    else:
        print("No solution found")

    print("Time taken:", end_time - start_time, "seconds")

    print("Number of Expanded Nodes:", expanded_nodes)
    print("Number of Generated Nodes:", generated_nodes)


# cost_matrix = generate_random_cost_matrix(N, seed)
# cost_matrix[np.arange(N), np.arange(N)] = 0  # Zero out diagonal

# start_time = time.time()
# initial_state = State(N)
# # solution = ida_star(initial_state, cost_matrix)
# solution, expanded_nodes, generated_nodes = ida_star(initial_state, cost_matrix)
# end_time = time.time()

# if solution:
#     print("Path:", solution.path)
#     print("Cost:", solution.current_cost)
# else:
#     print("No solution found")

# print("Time taken:", end_time - start_time, "seconds")

# print("Number of Expanded Nodes:", expanded_nodes)
# print("Number of Generated Nodes:", generated_nodes)