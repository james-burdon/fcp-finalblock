import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# Define a function to create the ring adjacency matrix
def create_ring_adjacency_matrix(n):
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[i, (i - 1) % n] = 1
    return A

# Define a function to rewire connections in the adjacency matrices of both nodes
def rewire_adjacency_matrices(A1, A2, p):
    n = A1.shape[0]
    for i in range(n):
        for j in range(i+1, n):  # Only consider connections once to avoid duplication
            if A1[i, j] == 1 and random.random() < p:  # If there's a connection and probability is met
                new_connection = random.choice([x for x in range(n) if x != i and x != j and A1[i, x] == 0 and A2[i, x] == 0])
                A1[i, j] = 0
                A1[j, i] = 0  # Remove old connection
                A2[i, j] = 0
                A2[j, i] = 0  # Remove old connection in the other adjacency matrix
                A1[i, new_connection] = 1
                A1[new_connection, i] = 1  # Add new connection
                A2[i, new_connection] = 1
                A2[new_connection, i] = 1  # Add new connection in the other adjacency matrix

# Parameters
n_nodes = 20
rewiring_probability = 0.2

# Create the ring adjacency matrices for both nodes
adjacency_matrix_1 = create_ring_adjacency_matrix(n_nodes)
adjacency_matrix_2 = np.copy(adjacency_matrix_1)  # Copy the original adjacency matrix for the second node

# Convert adjacency matrices into NetworkX graphs
G1 = nx.from_numpy_array(adjacency_matrix_1)
G2 = nx.from_numpy_array(adjacency_matrix_2)

# Rewire connections
rewire_adjacency_matrices(adjacency_matrix_2, adjacency_matrix_1, rewiring_probability)  # Swap the matrices for rewiring

# Convert rewired adjacency matrix into NetworkX graph
G2_rewired = nx.from_numpy_array(adjacency_matrix_2)

# Draw graphs
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
nx.draw_circular(G1, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
plt.title('Original Graph 1')

plt.subplot(1, 2, 2)
nx.draw_circular(G2_rewired, with_labels=True, node_color='lightcoral', node_size=500, font_size=10)
plt.title('Rewired Graph 2')

plt.tight_layout()
plt.show()
