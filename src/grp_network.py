import os
import random
import scipy
import numpy as np


def generate_adjacency_matrix(num_cities, sparsity=0.2, mixing_rate_range=(0.1, 1.0)):
    """
    Generate an adjacency matrix for a graph.
    """
    A = scipy.sparse.csr_matrix((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):  # Only consider upper triangular part
            if random.random() < sparsity:
                mixing_rate = random.uniform(*mixing_rate_range)
                A[i, j] = mixing_rate
                A[j, i] = mixing_rate  # Symmetric for undirected graph
    ensure_connected(A)
    return A


def ensure_connected(A):
    """
    Ensure the graph described by the adjacency matrix is connected.
    """
    num_cities = A.shape[0]
    visited = [False] * num_cities

    def dfs(city):
        visited[city] = True
        for neighbor in range(num_cities):
            if A[city, neighbor] > 0 and not visited[neighbor]:
                dfs(neighbor)

    dfs(0)

    for i in range(1, num_cities):
        if not visited[i]:
            connected_city = random.choice([j for j in range(num_cities) if visited[j]])
            mixing_rate = random.uniform(0.1, 1.0)
            A[i, connected_city] = mixing_rate
            A[connected_city, i] = mixing_rate
            dfs(i)


def generate_population_matrix(num_cities, population_range=(100, 1000), initial_infected=1, initial_infected_city_rate=0.1):
    """
    Generate a population matrix with SIRN values for each city.
    """
    population_matrix = np.zeros((num_cities, 4))

    for city in range(num_cities):
        total_population = random.randint(*population_range)
        susceptible = total_population - initial_infected
        if np.random.uniform() < initial_infected_city_rate:
            infected = initial_infected
        else:
            infected = 0
        recovered = 0

        population_matrix[city] = [susceptible, infected, recovered, total_population]

    return population_matrix


def save_dataset(adjacency_matrix, population_matrix, dataset_folder):
    """
    Save adjacency matrix and population matrix as files in a dataset folder.
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Save adjacency matrix with one line per city
    adjacency_matrix_path = os.path.join(dataset_folder, "adjacency_matrix.txt")
    with open(adjacency_matrix_path, "w") as f:
        for row in adjacency_matrix:
            # Format each value to remove trailing zeros and align columns
            pretty_row = ["{:6}".format(f"{value:.3g}" if value > 0 else "0") for value in row]
            f.write(" ".join(pretty_row) + "\n")
    print(f"Pretty-printed adjacency matrix saved to {adjacency_matrix_path}")

    # Save population matrix with one line per city
    population_matrix_path = os.path.join(dataset_folder, "population_matrix.json")
    with open(population_matrix_path, "w") as f:
        f.write("{\n")  # Start of the JSON object
        for i, (city, values) in enumerate(population_matrix.items()):
            # Write each city on one line
            line = f'    "{city}": {values}'
            if i != len(population_matrix) - 1:
                line += ","  # Add a comma except for the last line
            f.write(line + "\n")
        f.write("}\n")  # End of the JSON object
    print(f"Population matrix saved to {population_matrix_path}")


def generate_random_data(node_range, sparsity_range, mixing_rate_range, population_range):
    """
    """
    scale = np.random.choice(range(*node_range))
    sparsity = np.random.choice(np.linspace(sparsity_range[0], sparsity_range[1], 100))
    A = generate_adjacency_matrix(scale, sparsity, mixing_rate_range)
    SIRN = generate_population_matrix(scale, population_range, 1, 0.2)
    return A, SIRN


if __name__ == "__main__":
    node_range = [5, 20]
    sparsity_range = [0.2, 0.4]
    mixing_rate_range = (0.1, 0.9)
    population_range = (100, 500)

    A, SIRN = generate_random_data(node_range, sparsity_range, mixing_rate_range, population_range)
